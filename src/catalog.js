import { basename, delimiter, join } from 'node:path';
import { aliasesOf } from './config.js';
import { detectEngine, ENGINES } from './engines/index.js';
import {
  cachedRepos,
  formatModelId,
  isDir,
  parseModelId,
  repoDir,
  resolveRevision,
} from './hf/cache.js';
import { glob, modelFiles } from './hf/files.js';
import { badRequest, notFound } from './http/errors.js';

const SAFE = /^[A-Za-z0-9._/:@=-]+$/;
const LAYOUTS = 'https://github.com/ai-ecoverse/slicc-ortllama#engines';
const NAMED = 3;

const quote = (word) => `'${word.replaceAll("'", "'\\''")}'`;

export function shellWord(word) {
  return SAFE.test(word) ? word : quote(word);
}

export function downloadHint(repo, { revision = null, include = [] } = {}) {
  const rev = revision ? ` --revision ${shellWord(revision)}` : '';
  const inc = include.length > 0 ? ` --include ${include.map(quote).join(' ')}` : '';
  return `hf download ${shellWord(repo)}${rev}${inc}`;
}

function prefixed(subfolder, patterns) {
  return subfolder ? patterns.map((p) => `${subfolder}/${p}`) : patterns;
}

function snapshotFiles(cache, repo, snapshot) {
  return modelFiles(snapshot, { allow: [snapshot, join(repoDir(cache, repo), 'blobs')] });
}

async function onnxIn(files, dir) {
  const names = (await files.list(dir)).filter((n) => n.endsWith('.onnx')).sort();
  return names.map((n) => (dir ? `${dir}/${n}` : n));
}

async function strayOnnx(files, runnable) {
  const out = [...(await onnxIn(files, '')), ...(await onnxIn(files, 'onnx'))];
  for (const dir of await files.dirs()) {
    if (dir === 'onnx') continue;
    const found = [...(await onnxIn(files, dir)), ...(await onnxIn(files, `${dir}/onnx`))];
    if (found.length > 0 && !(await runnable(dir))) out.push(...found);
  }
  return out;
}

function unrunnable(id, where, onnx) {
  if (onnx.length === 0)
    return notFound(`model ${id} is ${where}, but has no ONNX model ortllama can run`);
  const more = onnx.length > NAMED ? ` and ${onnx.length - NAMED} more` : '';
  return notFound(
    `model ${id} is ${where}, but ortllama does not recognize the layout of its ONNX files (${onnx.slice(0, NAMED).join(', ')}${more}); the layouts it runs are listed in ${LAYOUTS}`
  );
}

export function localRoots(env = process.env) {
  return (env.ORTLLAMA_MODEL_PATH ?? '').split(delimiter).filter(Boolean);
}

function hintedProviders(id, hint) {
  const list = hint?.executionProviders;
  if (list === undefined) return null;
  if (
    !Array.isArray(list) ||
    list.length === 0 ||
    !list.every((ep) => ep && typeof ep === 'string')
  )
    throw new TypeError(
      `config.json hints.${id}.executionProviders must be a list of execution provider names`
    );
  return list;
}

async function specialized(engine, files, base, hint, variant) {
  const info = { ...base, hint: hint ?? {} };
  if (!engine.specialize) return { info };
  return engine.specialize(files, info, variant);
}

function hintedVariant(info, hint) {
  return info.variants.includes(hint?.variant) ? hint.variant : info.defaultVariant;
}

async function listEntry({ id, files, found, revision }, hint) {
  const defaultVariant = hintedVariant(found.info, hint);
  const { info, details } = await specialized(
    found.engine,
    files,
    found.info,
    hint,
    defaultVariant
  );
  return {
    id,
    engine: found.engine.name,
    capability: found.engine.capability,
    variants: info.variants,
    defaultVariant,
    revision,
    inputModalities: info.inputModalities,
    outputModalities: info.outputModalities,
    contextLength: info.contextLength,
    ...(details ? { details } : {}),
  };
}

const detectedIn = (files, engines) => (dir) => detectEngine(files.sub(dir), engines);

async function explore(engines, { id, files, subId, revision, where }) {
  const found = await detectEngine(files, engines);
  if (found) return [{ id, files, found, revision, where, root: true }];
  const runnable = new Set();
  const out = [
    { id, files, found, revision, where, root: true, runnable: async (d) => runnable.has(d) },
  ];
  for (const dir of await files.dirs()) {
    const sub = files.sub(dir);
    const hit = await detectEngine(sub, engines);
    if (hit) runnable.add(dir);
    const entry = { id: subId(dir), files: sub, found: hit, revision, where, root: false };
    out.push({ ...entry, runnable: detectedIn(sub, engines) });
  }
  return out;
}

async function localDirs(modelPath, engines) {
  const out = [];
  for (const root of modelPath) {
    if (!(await isDir(root))) continue;
    const id = `local/${basename(root)}`;
    const where = 'in ORTLLAMA_MODEL_PATH';
    const subId = (dir) => `local/${dir}`;
    out.push(
      ...(await explore(engines, { id, files: modelFiles(root), subId, revision: null, where }))
    );
  }
  return out;
}

async function hubDirs(cache, engines) {
  const out = [];
  for (const repo of await cachedRepos(cache)) {
    const rev = await resolveRevision(cache, repo);
    if (!rev) continue;
    const files = snapshotFiles(cache, repo, rev.snapshot);
    const subId = (dir) => `${repo}/${dir}`;
    out.push(
      ...(await explore(engines, {
        id: repo,
        files,
        subId,
        revision: rev.sha,
        where: 'in the cache',
      }))
    );
  }
  return out;
}

export function createCatalog({ cache, home, config, modelPath = [], engines = ENGINES }) {
  const aliases = aliasesOf(config);
  const hints = config.hints ?? {};

  async function resolveLocal(id, variant) {
    const dirs = (await localDirs(modelPath, engines)).filter((m) => m.id === id);
    const match = dirs.find((m) => m.found) ?? dirs.find((m) => m.root) ?? dirs[0];
    if (!match) throw notFound(`model ${id} is not in ORTLLAMA_MODEL_PATH`);
    if (!match.found)
      throw unrunnable(id, match.where, await strayOnnx(match.files, match.runnable));
    return finish({
      id,
      files: match.files,
      found: match.found,
      variant,
      revision: null,
      repo: null,
      subfolder: '',
    });
  }

  async function survey() {
    const dirs = [...(await hubDirs(cache, engines)), ...(await localDirs(modelPath, engines))];
    const models = await Promise.all(
      dirs.filter((m) => m.found).map((m) => listEntry(m, hints[m.id]))
    );
    const unrecognized = [];
    for (const m of dirs) {
      if (!m.root || m.found) continue;
      const onnx = await strayOnnx(m.files, m.runnable);
      if (onnx.length > 0) unrecognized.push(unrunnable(m.id, m.where, onnx).message);
    }
    return { models, unrecognized };
  }

  async function finish({
    id,
    files,
    found,
    variant,
    revision,
    repo,
    subfolder,
    requestedRevision = null,
  }) {
    const { engine } = found;
    const chosen = variant ?? hintedVariant(found.info, hints[id]);
    if (!found.info.variants.includes(chosen)) {
      throw notFound(
        `model ${id} has no variant ${chosen}; it has ${found.info.variants.join(', ')}`,
        'variant_not_found'
      );
    }
    const providers = hintedProviders(id, hints[id]);
    const { info, key } = await specialized(engine, files, found.info, hints[id], chosen);
    const missing = await engine.missing(files, info, chosen);
    const patterns = engine.include(info, chosen);
    const uncovered = missing.filter((rel) => !patterns.some((p) => glob(p).test(rel)));
    const download = repo
      ? downloadHint(repo, {
          revision: requestedRevision,
          include: prefixed(subfolder, [...patterns, ...uncovered]),
        })
      : null;
    if (missing.length > 0) {
      const hint = download ? `; run: ${download}` : '';
      throw notFound(
        `model ${id}:${chosen} is missing ${missing.length} file(s), such as ${missing[0]}${hint}`,
        'model_incomplete'
      );
    }
    return {
      id,
      variant: chosen,
      revision,
      engine,
      info,
      files,
      download,
      providers,
      key: `${id}:${chosen}@${revision ?? 'local'}${key ? `+${key}` : ''}${providers ? `+ep=${providers.join(',')}` : ''}`,
    };
  }

  return {
    home,
    cache,
    aliases,
    async resolve(requested) {
      if (typeof requested !== 'string' || requested.trim() === '')
        throw badRequest('model is required', 'model');
      const target = aliases[requested] ?? requested;
      const [localId, localVariant = null] = target.split(':');
      if (localId.startsWith('local/'))
        return {
          ...(await resolveLocal(localId, localVariant)),
          alias: target === requested ? null : requested,
        };
      let parsed;
      try {
        parsed = parseModelId(target);
      } catch (error) {
        throw badRequest(error.message, 'model');
      }
      const { repo, subfolder, revision, variant } = parsed;
      const id = formatModelId({ repo, subfolder });
      const rev = await resolveRevision(cache, repo, revision ?? 'main');
      const include = subfolder ? [`${subfolder}/*`] : [];
      if (!rev)
        throw notFound(
          `model ${id} is not in the Hugging Face cache; run: ${downloadHint(repo, { revision, include })}`
        );
      const root = snapshotFiles(cache, repo, rev.snapshot);
      const files = subfolder ? root.sub(subfolder) : root;
      if (!(await isDir(files.root)))
        throw notFound(
          `model ${id} has no files ortllama can run in the cache; run: ${downloadHint(repo, { revision, include })}`
        );
      const found = await detectEngine(files, engines);
      if (!found)
        throw unrunnable(id, 'in the cache', await strayOnnx(files, detectedIn(files, engines)));
      const resolved = await finish({
        id,
        files,
        found,
        variant,
        revision: rev.sha,
        repo,
        subfolder,
        requestedRevision: revision,
      });
      return { ...resolved, alias: target === requested ? null : requested };
    },
    async list() {
      return (await survey()).models;
    },
    survey,
  };
}

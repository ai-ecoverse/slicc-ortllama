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
import { modelFiles } from './hf/files.js';
import { badRequest, notFound } from './http/errors.js';

const SAFE = /^[A-Za-z0-9._/:@=-]+$/;

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
  if (!engine.specialize || !info.variants.includes(variant)) return { info };
  return engine.specialize(files, info, variant);
}

async function listEntry({ id, files, found, revision }, hint) {
  const defaultVariant = hint?.variant ?? found.info.defaultVariant;
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

export function createCatalog({ cache, home, config, modelPath = [], engines = ENGINES }) {
  const aliases = aliasesOf(config);
  const hints = config.hints ?? {};

  async function localModels() {
    const out = [];
    for (const root of modelPath) {
      if (!(await isDir(root))) continue;
      const files = modelFiles(root);
      const found = await detectEngine(files, engines);
      if (found) {
        out.push({ id: `local/${basename(root)}`, files, found, revision: null });
        continue;
      }
      for (const dir of await files.dirs()) {
        const sub = files.sub(dir);
        const hit = await detectEngine(sub, engines);
        if (hit) out.push({ id: `local/${dir}`, files: sub, found: hit, revision: null });
      }
    }
    return out;
  }

  async function hubModels() {
    const out = [];
    for (const repo of await cachedRepos(cache)) {
      const rev = await resolveRevision(cache, repo);
      if (!rev) continue;
      const files = snapshotFiles(cache, repo, rev.snapshot);
      const found = await detectEngine(files, engines);
      if (found) {
        out.push({ id: repo, files, found, revision: rev.sha });
        continue;
      }
      for (const dir of await files.dirs()) {
        const sub = files.sub(dir);
        const hit = await detectEngine(sub, engines);
        if (hit) out.push({ id: `${repo}/${dir}`, files: sub, found: hit, revision: rev.sha });
      }
    }
    return out;
  }

  async function resolveLocal(id, variant) {
    const match = (await localModels()).find((m) => m.id === id);
    if (!match) throw notFound(`model ${id} is not in ORTLLAMA_MODEL_PATH`);
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
    const chosen = variant ?? hints[id]?.variant ?? found.info.defaultVariant;
    if (!found.info.variants.includes(chosen)) {
      throw notFound(
        `model ${id} has no variant ${chosen}; it has ${found.info.variants.join(', ')}`,
        'variant_not_found'
      );
    }
    const providers = hintedProviders(id, hints[id]);
    const { info, key } = await specialized(engine, files, found.info, hints[id], chosen);
    const missing = await engine.missing(files, info, chosen);
    const download = repo
      ? downloadHint(repo, {
          revision: requestedRevision,
          include: prefixed(subfolder, engine.include(info, chosen)),
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
      const found = (await isDir(files.root)) ? await detectEngine(files, engines) : null;
      if (!found)
        throw notFound(
          `model ${id} has no files ortllama can run in the cache; run: ${downloadHint(repo, { revision, include })}`
        );
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
      const models = [...(await hubModels()), ...(await localModels())];
      return Promise.all(models.map((m) => listEntry(m, hints[m.id])));
    },
  };
}

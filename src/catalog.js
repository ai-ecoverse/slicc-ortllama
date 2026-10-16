import { basename, delimiter } from 'node:path';
import { aliasesOf } from './config.js';
import { detectEngine, ENGINES } from './engines/index.js';
import { cachedRepos, formatModelId, isDir, parseModelId, resolveRevision } from './hf/cache.js';
import { modelFiles } from './hf/files.js';
import { badRequest, notFound } from './http/errors.js';

const quote = (s) => `'${s}'`;

export function downloadHint(repo, { revision = null, include = [] } = {}) {
  const rev = revision ? ` --revision ${revision}` : '';
  const inc = include.length > 0 ? ` --include ${include.map(quote).join(' ')}` : '';
  return `hf download ${repo}${rev}${inc}`;
}

function prefixed(subfolder, patterns) {
  return subfolder ? patterns.map((p) => `${subfolder}/${p}`) : patterns;
}

export function localRoots(env = process.env) {
  return (env.ORTLLAMA_MODEL_PATH ?? '').split(delimiter).filter(Boolean);
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
      const files = modelFiles(rev.snapshot);
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
    const { engine, info } = found;
    const chosen = variant ?? hints[id]?.variant ?? info.defaultVariant;
    if (!info.variants.includes(chosen)) {
      throw notFound(
        `model ${id} has no variant ${chosen}; it has ${info.variants.join(', ')}`,
        'variant_not_found'
      );
    }
    const missing = await engine.missing(files, info, chosen);
    if (missing.length > 0) {
      const hint = repo
        ? `; run: ${downloadHint(repo, { revision: requestedRevision, include: prefixed(subfolder, engine.include(info, chosen)) })}`
        : '';
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
      key: `${id}:${chosen}@${revision ?? 'local'}`,
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
      const root = modelFiles(rev.snapshot);
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
      return models.map(({ id, found, revision }) => ({
        id,
        engine: found.engine.name,
        capability: found.engine.capability,
        variants: found.info.variants,
        defaultVariant: hints[id]?.variant ?? found.info.defaultVariant,
        revision,
        inputModalities: found.info.inputModalities,
        outputModalities: found.info.outputModalities,
        contextLength: found.info.contextLength,
      }));
    },
  };
}

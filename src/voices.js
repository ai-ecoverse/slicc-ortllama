import { HttpError } from './http/errors.js';
import { resolveSpeech } from './speech.js';

async function voicesOf(resolved, model, cache) {
  const { engine } = resolved;
  const voices = engine.voices ? await engine.voices(resolved.files, { cache }) : [];
  return voices.map(({ id, language, gender }) => ({
    id,
    model,
    engine: engine.name,
    language,
    gender,
  }));
}

async function speaking(catalog, name) {
  try {
    return await resolveSpeech(catalog, name);
  } catch (error) {
    if (error instanceof HttpError) return null;
    throw error;
  }
}

async function everyModel(catalog) {
  const listed = (await catalog.list()).filter((m) => m.capability === 'speech').map((m) => m.id);
  const seen = new Set();
  const out = [];
  for (const name of [...listed, ...Object.keys(catalog.aliases)]) {
    const resolved = await speaking(catalog, name);
    const key = resolved && `${resolved.id}@${resolved.revision}`;
    if (!resolved || seen.has(key)) continue;
    seen.add(key);
    out.push({ resolved, name });
  }
  return out;
}

export function voicesRoute({ catalog }) {
  return {
    async handle({ url }) {
      const model = url.searchParams.get('model');
      const models =
        model === null
          ? await everyModel(catalog)
          : [{ resolved: await resolveSpeech(catalog, model), name: model }];
      const data = [];
      for (const { resolved, name } of models)
        data.push(...(await voicesOf(resolved, name, catalog.cache)));
      return { object: 'list', data };
    },
  };
}

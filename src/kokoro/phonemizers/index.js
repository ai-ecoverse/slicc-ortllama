import { HttpError } from '../../http/errors.js';
import { espeakNg } from './espeak.js';
import { misaki } from './misaki.js';
import { phonemizeJs } from './phonemize.js';
import { xenova } from './xenova.js';

export const PHONEMIZERS = [misaki, espeakNg, phonemizeJs, xenova];

function unavailable(message) {
  return new HttpError(501, message, { code: 'engine_unavailable' });
}

function installs(language) {
  return PHONEMIZERS.filter((c) => !language || c.languages.includes(language))
    .map((c) => `${c.name} (${c.install})`)
    .join(', ');
}

export function phonemizerOrder(env = process.env, candidates = PHONEMIZERS) {
  const names = (env.ORTLLAMA_PHONEMIZERS ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (names.length === 0) return candidates;
  return names.map((name) => {
    const found = candidates.find((c) => c.name === name);
    if (!found)
      throw new RangeError(
        `ORTLLAMA_PHONEMIZERS names ${name}; use ${candidates.map((c) => c.name).join(', ')}`
      );
    return found;
  });
}

export async function loadPhonemizers({ candidates = phonemizerOrder(), ...context } = {}) {
  const loaded = [];
  const failures = [];
  const pick = (language, skip = null) =>
    loaded.find((p) => p !== skip && p.languages.includes(language)) ?? null;
  for (const candidate of candidates) {
    let backend;
    const fallback = (language) => pick(language, backend);
    try {
      backend = await candidate.load({ ...context, fallback });
    } catch (error) {
      failures.push(error);
      continue;
    }
    loaded.push(backend);
  }
  if (loaded.length === 0)
    throw (
      failures.find((e) => e?.code !== 'engine_unavailable') ??
      unavailable(`Kokoro needs a phonemizer; install one of: ${installs()}`)
    );
  return {
    names: loaded.map((p) => p.name),
    for(language) {
      const found = pick(language);
      if (!found)
        throw unavailable(
          `no phonemizer for ${language} is installed; install one of: ${installs(language)}`
        );
      return found;
    },
    async phonemize(parts, language) {
      return this.for(language).phonemize(parts, language);
    },
  };
}

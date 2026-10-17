import { HttpError } from '../../http/errors.js';
import { espeakNg } from './espeak.js';
import { misaki } from './misaki.js';
import { phonemizeJs } from './phonemize.js';
import { xenova } from './xenova.js';

export const PHONEMIZERS = [misaki, phonemizeJs, espeakNg, xenova];

export const BROKEN_MS = 60_000;

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
      throw new HttpError(
        400,
        `ORTLLAMA_PHONEMIZERS names ${name}; use ${candidates.map((c) => c.name).join(', ')}`,
        { code: 'invalid_phonemizer' }
      );
    return found;
  });
}

export async function loadPhonemizers({
  candidates = phonemizerOrder(),
  now = Date.now,
  ...context
} = {}) {
  const loaded = [];
  const failures = [];
  const broken = new Map();
  const usable = (language, skip) =>
    loaded.filter(
      (p) => !skip.includes(p) && p.languages.includes(language) && !(broken.get(p) > now())
    );

  async function run(parts, language, skip = []) {
    const choices = usable(language, skip);
    if (choices.length === 0)
      throw unavailable(
        `no phonemizer for ${language} is installed; install one of: ${installs(language)}`
      );
    let last;
    for (const backend of choices) {
      try {
        return await backend.phonemize(parts, language);
      } catch (error) {
        if (error instanceof HttpError) throw error;
        broken.set(backend, now() + BROKEN_MS);
        last = error;
      }
    }
    throw last;
  }

  for (const candidate of candidates) {
    let backend;
    const fallback = (language) =>
      usable(language, [backend]).length > 0
        ? { phonemize: (parts, lang) => run(parts, lang, [backend]) }
        : null;
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
    for: (language) => usable(language, [])[0] ?? null,
    phonemize: (parts, language) => run(parts, language),
  };
}

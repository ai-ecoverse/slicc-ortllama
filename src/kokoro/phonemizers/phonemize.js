import { createRequire } from 'node:module';
import { join } from 'node:path';
import { moduleRoots, optional } from '../../engines/optional.js';

const LANGUAGES = ['en-US', 'en-GB'];

export function toKokoro(ipa) {
  return ipa
    .replace(/ɫ/g, 'l')
    .replace(/ɝ/g, 'ɜɹ')
    .replace(/([ˈˌ])([^aeiouæɑɒɔəɚɛɜɪʊʌ\sˈˌ]+)/g, '$2$1');
}

export function createPhonemize(lib) {
  return {
    name: 'phonemize',
    languages: LANGUAGES,
    async phonemize(parts, language) {
      return parts.map((part) =>
        toKokoro(lib.phonemize(part, { language, affricates: 'ligature' }))
      );
    },
  };
}

async function required(env, roots, make) {
  const bases = [import.meta.url, ...(await roots(env)).map((root) => join(root, 'package.json'))];
  for (const base of bases) {
    try {
      return make(base)('phonemize');
    } catch {}
  }
  return null;
}

export const phonemizeJs = {
  name: 'phonemize',
  install: 'npm install phonemize',
  languages: LANGUAGES,
  async load({ load = optional, env, roots = moduleRoots, make = createRequire } = {}) {
    let lib;
    try {
      lib = await load('phonemize', { env });
    } catch (error) {
      if (error?.code === 'engine_unavailable') throw error;
      lib = await required(env, roots, make);
      if (!lib) throw error;
    }
    return createPhonemize(lib.phonemize ? lib : lib.default);
  },
};

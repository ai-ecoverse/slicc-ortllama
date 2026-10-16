import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { locate, optional } from '../../engines/optional.js';

const NAME = 'phonemize';
const LANGUAGES = ['en-US', 'en-GB'];

export function toKokoro(ipa, language = 'en-US') {
  const kokoro = ipa
    .replace(/ɫ/g, 'l')
    .replace(/ɝ/g, 'ɜɹ')
    .replace(/([ˈˌ])([^aeiouæɑɒɔəɚɛɜɪʊʌ\sˈˌ]+)/g, '$2$1');
  if (language !== 'en-GB') return kokoro;
  return kokoro
    .replace(/oʊ/g, 'Q')
    .replace(/ɑ(?!ː)/g, 'ɒ')
    .replace(/ɔ(?![ːɪ])/g, 'ɔː');
}

export function createPhonemize(lib) {
  return {
    name: NAME,
    languages: LANGUAGES,
    async phonemize(parts, language) {
      return parts.map((part) =>
        toKokoro(lib.phonemize(part, { language, affricates: 'ligature' }), language)
      );
    },
  };
}

export function requireEntry(pkg) {
  const root = pkg.exports?.['.'] ?? pkg.exports;
  if (typeof root === 'string') return root;
  return root?.require ?? pkg.main ?? 'index.js';
}

async function packageRoot(file, read) {
  for (let dir = dirname(file); dir !== dirname(dir); dir = dirname(dir)) {
    try {
      const pkg = JSON.parse(await read(join(dir, 'package.json'), 'utf8'));
      if (pkg.name === NAME) return { dir, pkg };
    } catch {}
  }
  return null;
}

async function required({ env, resolve, find, make, read }) {
  let file;
  try {
    file = fileURLToPath(resolve(NAME));
  } catch {
    file = await find(NAME, env);
  }
  const root = file ? await packageRoot(file, read) : null;
  if (!root) return null;
  return make(join(root.dir, 'package.json'))(join(root.dir, requireEntry(root.pkg)));
}

export const phonemizeJs = {
  name: NAME,
  install: 'npm install phonemize',
  languages: LANGUAGES,
  async load({
    load = optional,
    env,
    resolve = (specifier) => import.meta.resolve(specifier),
    find = locate,
    make = createRequire,
    read = readFile,
  } = {}) {
    let lib;
    try {
      lib = await load(NAME, { env });
    } catch (error) {
      if (error?.code === 'engine_unavailable') throw error;
      lib = await required({ env, resolve, find, make, read }).catch(() => null);
      if (!lib) throw error;
    }
    return createPhonemize(lib.phonemize ? lib : lib.default);
  },
};

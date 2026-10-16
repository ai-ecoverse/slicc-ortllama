import { readFile } from 'node:fs/promises';
import { optional } from '../../engines/optional.js';

const IN = 'input.txt';
const OUT = 'phonemes.txt';

const VOICES = {
  'en-US': 'en-us',
  'en-GB': 'en-gb',
  'es-ES': 'es',
  'fr-FR': 'fr-fr',
  'hi-IN': 'hi',
  'it-IT': 'it',
  'pt-BR': 'pt-br',
};

export function createEspeak(factory, { readBinary = readFile } = {}) {
  let compiled = null;
  let located = null;
  const compile = () => {
    const path = /^[a-z]+:/i.test(located) ? new URL(located) : located;
    compiled ??= readBinary(path).then((bytes) => WebAssembly.compile(bytes));
    compiled.catch(() => {
      compiled = null;
    });
    return compiled;
  };
  async function run(text, voice) {
    const errors = [];
    let fail;
    const failed = new Promise((_, reject) => {
      fail = reject;
    });
    const module = await Promise.race([
      factory({
        arguments: ['-q', '-b', '1', '--ipa', '-v', voice, '--phonout', OUT, '-f', IN],
        preRun: [(m) => m.FS.writeFile(IN, text)],
        locateFile(path, prefix) {
          located ??= prefix + path;
          return located;
        },
        instantiateWasm(imports, done) {
          compile()
            .then((wasm) => WebAssembly.instantiate(wasm, imports))
            .then(done)
            .catch(fail);
          return {};
        },
        print() {},
        printErr: (line) => errors.push(line),
        quit(_status, error) {
          throw error;
        },
      }),
      failed,
    ]);
    let raw;
    try {
      raw = module.FS.readFile(OUT, { encoding: 'utf8' });
    } catch {
      throw new Error(`espeak-ng could not phonemize for ${voice}: ${errors.join(' ')}`);
    }
    return raw
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
  }
  return {
    name: 'espeak-ng',
    languages: Object.keys(VOICES),
    async phonemize(parts, language) {
      if (parts.length === 0) return [];
      const voice = VOICES[language];
      const lines = await run(parts.join('\n\n'), voice);
      if (lines.length === parts.length) return lines;
      const out = [];
      for (const part of parts) out.push((await run(part, voice)).join(' '));
      return out;
    },
  };
}

export const espeakNg = {
  name: 'espeak-ng',
  install: 'npm install espeak-ng',
  languages: Object.keys(VOICES),
  async load({ load = optional, env } = {}) {
    const { default: factory } = await load('espeak-ng', { env });
    return createEspeak(factory);
  },
};

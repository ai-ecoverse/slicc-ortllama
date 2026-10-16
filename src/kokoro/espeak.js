import { readFile } from 'node:fs/promises';

const IN = 'input.txt';
const OUT = 'phonemes.txt';

export async function loadEspeak({
  importGlue = () => import('espeak-ng'),
  resolve = (specifier) => import.meta.resolve(specifier),
  readBinary = readFile,
} = {}) {
  const [{ default: factory }, bytes] = await Promise.all([
    importGlue(),
    readBinary(new URL('espeak-ng.wasm', resolve('espeak-ng'))),
  ]);
  return createEspeak(factory, await WebAssembly.compile(bytes));
}

export function createEspeak(factory, compiled) {
  async function run(text, voice) {
    const errors = [];
    let fail;
    const failed = new Promise((_, reject) => {
      fail = reject;
    });
    const module = await Promise.race([
      factory({
        arguments: ['-q', '-b', '1', '--ipa', '-v', voice, '--phonout', OUT, '-f', IN],
        preRun: [(module) => module.FS.writeFile(IN, text)],
        instantiateWasm(imports, done) {
          WebAssembly.instantiate(compiled, imports).then(done).catch(fail);
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
    async phonemize(parts, voice) {
      if (parts.length === 0) return [];
      const lines = await run(parts.join('\n\n'), voice);
      if (lines.length === parts.length) return lines;
      const out = [];
      for (const part of parts) out.push((await run(part, voice)).join(' '));
      return out;
    },
  };
}

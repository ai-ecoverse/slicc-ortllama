import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { describe, test } from 'node:test';
import { decodeFile } from '../../src/audio/decode.js';
import { fixture } from './audio.mjs';

const ROOT = new URL('../../', import.meta.url);

function snr(reference, decoded) {
  let signal = 0;
  let noise = 0;
  for (let i = 0; i < reference.length; i++) {
    signal += reference[i] ** 2;
    noise += (reference[i] - decoded[i]) ** 2;
  }
  return 10 * Math.log10(signal / noise);
}

function similarity(a, b) {
  let dot = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < b.length; i++) {
    dot += a[i] * b[i];
    x += a[i] ** 2;
    y += b[i] ** 2;
  }
  return dot / Math.sqrt(x * y);
}

const blocked = `
import { registerHooks } from 'node:module';
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === 'minimp3-wasm') throw Object.assign(new Error('blocked'), { code: 'ERR_MODULE_NOT_FOUND' });
    return next(specifier, context);
  },
});
const { decodeFile } = await import(${JSON.stringify(new URL('src/audio/decode.js', ROOT).href)});
const { readFileSync } = await import('node:fs');
const bytes = new Uint8Array(readFileSync(${JSON.stringify(new URL('test/integration/fixtures/fox.mp3', ROOT).pathname)}));
console.log((await decodeFile(bytes, { rate: 16000 })).length);
`;

describe('mp3 decoding in JavaScript', () => {
  test('needs no minimp3-wasm', async () => {
    const pkg = JSON.parse(await readFile(new URL('package.json', ROOT), 'utf8'));
    assert.equal(pkg.dependencies['minimp3-wasm'], undefined);
    const child = spawnSync(process.execPath, ['--input-type=module', '-e', blocked], {
      encoding: 'utf8',
    });
    assert.equal(child.status, 0, child.stderr);
    assert.equal(child.stdout.trim(), '42732');
  });

  test("matches ffmpeg's decoder on MPEG-1 MS stereo, MPEG-2 and MPEG-2.5", async () => {
    for (const [name, rate] of [
      ['mix-ms', 44100],
      ['mix-mpeg2', 22050],
      ['mix-mpeg25', 11025],
    ]) {
      const decoded = await decodeFile(await fixture('mp3', name), { rate });
      const reference = await decodeFile(await fixture('flac', name), { rate });
      assert.equal(decoded.length, reference.length, name);
      const quality = snr(reference, decoded);
      assert.ok(quality > 75, `${name}: ${quality.toFixed(1)} dB`);
      const worst = decoded.reduce((m, x, i) => Math.max(m, Math.abs(x - reference[i]) * 32768), 0);
      assert.ok(worst < 0.51, `${name}: ${worst.toFixed(2)} LSB off the 16-bit ffmpeg output`);
    }
  });

  test('decodes free-format mp3 and trims its LAME delay', async () => {
    const decoded = await decodeFile(await fixture('mp3', 'mix-free'), { rate: 44100 });
    const source = await decodeFile(await fixture('flac', 'mix-free'), { rate: 44100 });
    assert.equal(decoded.length, source.length);
    assert.ok(similarity(decoded, source) > 0.99);
  });

  test('answers clear errors for layer II and for streams without frames', async () => {
    await assert.rejects(
      decodeFile(await fixture('mp2', 'tone'), { type: 'audio/mpeg' }),
      /MPEG audio layer II is not supported/
    );
    const free = new Uint8Array(4000);
    free.set([0xff, 0xfb, 0x04, 0xc4]);
    const started = performance.now();
    await assert.rejects(decodeFile(free, { type: 'audio/mpeg' }), /the mp3 stream is corrupt/);
    await assert.rejects(
      decodeFile(new Uint8Array(3000).fill(0x55), { type: 'audio/mpeg' }),
      /the mp3 stream is corrupt/
    );
    assert.ok(performance.now() - started < 2000);
  });
});

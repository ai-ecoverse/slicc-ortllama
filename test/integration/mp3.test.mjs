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

  test('answers 400 when the sample rate changes mid-stream', async () => {
    const first = await fixture('mp3', 'mix-mpeg25');
    const second = await fixture('mp3', 'tone-plain');
    const joined = new Uint8Array(first.length + second.length);
    joined.set(first);
    joined.set(second, first.length);
    await assert.rejects(decodeFile(joined), /the audio changes its sample rate/);
  });
});

const unshipped = `
import { registerHooks } from 'node:module';
registerHooks({
  resolve(specifier, context, next) {
    if (specifier === '@breezystack/lamejs') throw Object.assign(new Error("Cannot find package '@breezystack/lamejs'"), { code: 'ERR_MODULE_NOT_FOUND' });
    return next(specifier, context);
  },
});
const { ENCODERS } = await import(${JSON.stringify(new URL('src/audio/encode.js', ROOT).href)});
const open = await ENCODERS.mp3.open();
const encoder = open(24000);
const tone = Float32Array.from({ length: 24000 }, (_, i) => 0.3 * Math.sin(i * 0.1));
const bytes = [...encoder.push(tone), ...encoder.end()];
console.log(bytes.length);
`;

function encode(samples, rate, open) {
  const encoder = open(rate);
  const parts = [];
  for (let at = 0; at < samples.length; at += 3000)
    parts.push(encoder.push(samples.subarray(at, at + 3000)));
  parts.push(encoder.end());
  return new Uint8Array(Buffer.concat(parts));
}

function aligned(source, decoded) {
  let best = { snr: -Infinity, delay: 0 };
  for (let delay = 0; delay < 2000; delay++) {
    let signal = 0;
    let noise = 0;
    for (let i = 0; i < source.length; i += 4) {
      signal += source[i] ** 2;
      noise += (source[i] - (decoded[i + delay] ?? 0)) ** 2;
    }
    const value = 10 * Math.log10(signal / noise);
    if (value > best.snr) best = { snr: value, delay };
  }
  return best;
}

const ffmpeg = spawnSync('ffmpeg', ['-version']).status === 0;

describe('mp3 encoding in JavaScript', () => {
  test('needs no lamejs', async () => {
    const pkg = JSON.parse(await readFile(new URL('package.json', ROOT), 'utf8'));
    for (const field of ['dependencies', 'devDependencies', 'peerDependencies'])
      assert.equal(pkg[field]?.['@breezystack/lamejs'], undefined, field);
    const home = new URL('test/integration/fixtures/', ROOT).pathname;
    const child = spawnSync(process.execPath, ['--input-type=module', '-e', unshipped], {
      encoding: 'utf8',
      env: { ...process.env, HOME: home, PNPM_HOME: home, ORTLLAMA_MODULE_PATH: '' },
    });
    assert.equal(child.status, 0, child.stderr);
    assert.ok(Number(child.stdout) > 7000, child.stdout);
  });

  test('round-trips speech at every mp3 sample rate', async () => {
    const { ENCODERS } = await import('../../src/audio/encode.js');
    const { decodeWav, resample } = await import('../../src/audio/pcm.js');
    const open = await ENCODERS.mp3.open();
    const { samples, rate } = decodeWav(await fixture('wav'));
    for (const target of [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]) {
      const source = resample(samples, rate, target);
      const bytes = encode(source, target, open);
      const kbps = (bytes.length * 8) / 1000 / (source.length / target);
      assert.ok(kbps > 63 && kbps < 70, `${target}: ${kbps.toFixed(1)} kbit/s`);
      const decoded = await decodeFile(bytes, { rate: target });
      assert.ok(decoded.length >= source.length + 1057, `${target}: ${decoded.length}`);
      const { snr: quality, delay } = aligned(source, decoded);
      assert.equal(delay, 1057, `${target}`);
      assert.ok(quality > 30, `${target}: ${quality.toFixed(1)} dB`);
    }
  });

  test('decodes in ffmpeg to the same samples', { skip: !ffmpeg }, async () => {
    const { ENCODERS } = await import('../../src/audio/encode.js');
    const { decodeWav, resample } = await import('../../src/audio/pcm.js');
    const { samples, rate } = decodeWav(await fixture('wav'));
    for (const target of [8000, 24000, 44100]) {
      const bytes = encode(resample(samples, rate, target), target, await ENCODERS.mp3.open());
      const raw = spawnSync('ffmpeg', ['-v', 'error', '-i', '-', '-f', 'f32le', '-'], {
        input: bytes,
        maxBuffer: 1 << 26,
      }).stdout;
      const reference = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
      const decoded = await decodeFile(bytes, { rate: target });
      assert.equal(decoded.length, reference.length);
      assert.ok(snr(reference, decoded) > 70, `${target}: ${snr(reference, decoded)}`);
    }
  });
});

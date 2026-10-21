import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { describe, test } from 'node:test';
import { decodeFile } from '../../src/audio/decode.js';
import { ENCODERS } from '../../src/audio/encode.js';
import { decodeWav, resample } from '../../src/audio/pcm.js';
import { fixture } from './audio.mjs';

const ffmpeg = spawnSync('ffmpeg', ['-version']).status === 0;

function encode(samples, rate, open, chunk = 3000) {
  const encoder = open(rate);
  const parts = [];
  for (let at = 0; at < samples.length; at += chunk)
    parts.push(encoder.push(samples.subarray(at, at + chunk)));
  parts.push(encoder.end());
  return { bytes: new Uint8Array(Buffer.concat(parts)), parts, rate: encoder.rate };
}

function frames(bytes) {
  const out = [];
  for (let at = 0; at < bytes.length; ) {
    assert.equal(bytes[at], 0xff, `sync at ${at}`);
    assert.equal(bytes[at + 1] & 0xf6, 0xf0, `sync at ${at}`);
    const length = ((bytes[at + 3] & 3) << 11) | (bytes[at + 4] << 3) | (bytes[at + 5] >> 5);
    const head = (bytes[at + 7] << 16) | (bytes[at + 8] << 8) | bytes[at + 9];
    out.push({ length, sequence: (head >> 6) & 3, profile: bytes[at + 2] >> 6 });
    at += length;
  }
  return out;
}

function snr(reference, decoded, delay = 0) {
  let signal = 0;
  let noise = 0;
  for (let i = 0; i < reference.length; i++) {
    signal += reference[i] ** 2;
    noise += (reference[i] - (decoded[i + delay] ?? 0)) ** 2;
  }
  return 10 * Math.log10(signal / noise);
}

function aligned(source, decoded) {
  let best = { snr: -Infinity, delay: 0 };
  for (let delay = 0; delay < 2048; delay++) {
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

async function fox(rate) {
  const { samples, rate: from } = decodeWav(await fixture('wav'));
  return resample(samples, from, rate);
}

function attack() {
  let seed = 1;
  const noise = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
    return seed / 0x7fffffff - 0.5;
  };
  const onset = 12300;
  const samples = Float32Array.from({ length: 36000 }, (_, i) =>
    i < onset ? 0 : 0.5 * Math.exp(-(i - onset) / 2400) * (noise() + Math.sin(i * 0.3))
  );
  return { samples, onset };
}

describe('AAC-LC encoding in JavaScript', () => {
  test('answers audio/aac', async () => {
    assert.equal(ENCODERS.aac.type, 'audio/aac');
    assert.equal(typeof (await ENCODERS.aac.open()), 'function');
  });

  test('round-trips speech at every AAC-LC sample rate within 64 kbit/s', async () => {
    const open = await ENCODERS.aac.open();
    for (const rate of [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]) {
      const source = await fox(rate);
      const { bytes } = encode(source, rate, open);
      const list = frames(bytes);
      assert.ok(
        list.every((f) => f.profile === 1),
        `${rate}`
      );
      assert.equal(list.length, Math.ceil(source.length / 1024) + 1, `${rate}`);
      const kbps = (bytes.length * 8) / 1000 / ((list.length * 1024) / rate);
      assert.ok(kbps > 40 && kbps <= 64, `${rate}: ${kbps.toFixed(1)} kbit/s`);
      const decoded = await decodeFile(bytes, { rate, type: 'audio/aac' });
      assert.equal(decoded.length, list.length * 1024, `${rate}`);
      const { snr: quality, delay } = aligned(source, decoded);
      assert.equal(delay, 1024, `${rate}`);
      assert.ok(quality > 30, `${rate}: ${quality.toFixed(1)} dB`);
    }
  });

  test('takes other sample rates at the nearest AAC rate', async () => {
    const open = await ENCODERS.aac.open();
    const source = await fox(7350);
    const { bytes, rate } = encode(source, 7350, open);
    assert.equal(rate, 8000);
    assert.equal((bytes[2] >> 2) & 15, 11);
    const decoded = await decodeFile(bytes, { rate: 8000, type: 'audio/aac' });
    const { snr: quality, delay } = aligned(resample(source, 7350, 8000), decoded);
    assert.equal(delay, 1024);
    assert.ok(quality > 30, `${quality.toFixed(1)} dB`);
  });

  test('decodes in ffmpeg to the same samples', { skip: !ffmpeg }, async () => {
    const open = await ENCODERS.aac.open();
    for (const rate of [8000, 24000, 44100]) {
      const { bytes } = encode(await fox(rate), rate, open);
      const probe = spawnSync(
        'ffprobe',
        ['-v', 'error', '-show_entries', 'stream=codec_name,profile,sample_rate,channels', '-'],
        { input: bytes, encoding: 'buffer' }
      );
      const info = probe.stdout.toString();
      assert.match(info, /codec_name=aac/);
      assert.match(info, /profile=LC/);
      assert.match(info, new RegExp(`sample_rate=${rate}`));
      assert.match(info, /channels=1/);
      const run = spawnSync('ffmpeg', ['-v', 'error', '-i', '-', '-f', 'f32le', '-'], {
        input: bytes,
        maxBuffer: 1 << 26,
      });
      assert.equal(run.stderr.toString(), '');
      const raw = run.stdout;
      const reference = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
      const decoded = await decodeFile(bytes, { rate });
      assert.equal(decoded.length, reference.length);
      assert.ok(snr(reference, decoded) > 100, `${rate}: ${snr(reference, decoded)}`);
    }
  });

  test('switches to short blocks on an attack and keeps the silence before it clean', async () => {
    const open = await ENCODERS.aac.open();
    const { samples, onset } = attack();
    const { bytes } = encode(samples, 24000, open);
    const sequences = frames(bytes).map((f) => f.sequence);
    const at = sequences.indexOf(2);
    assert.ok(at > 0, sequences.join(''));
    assert.equal(sequences[at - 1], 1);
    assert.equal(sequences[sequences.lastIndexOf(2) + 1], 3);
    const decoded = await decodeFile(bytes, { rate: 24000 });
    let before = 0;
    for (let i = onset - 2048; i < onset - 48; i++) before += decoded[i + 1024] ** 2;
    let burst = 0;
    for (let i = onset; i < onset + 2000; i++) burst += samples[i] ** 2;
    const echo = 10 * Math.log10(before / burst);
    assert.ok(echo < -50, `pre-echo ${echo.toFixed(1)} dB`);
  });

  test("keeps every frame inside the decoder's 6144-bit buffer on loud noise", async () => {
    const open = await ENCODERS.aac.open();
    let seed = 7;
    const loud = Float32Array.from({ length: 48000 }, () => {
      seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
      return (seed / 0x7fffffff) * 1.8 - 0.9;
    });
    const { bytes } = encode(loud, 24000, open);
    const list = frames(bytes);
    assert.ok(list.every((f) => (f.length - 7) * 8 <= 6144));
    const kbps = (bytes.length * 8) / 1000 / ((list.length * 1024) / 24000);
    assert.ok(kbps > 60 && kbps <= 64, `${kbps.toFixed(1)} kbit/s`);
    const decoded = await decodeFile(bytes, { rate: 24000 });
    assert.ok(snr(loud, decoded, 1024) > 3);
  });

  test('streams whole ADTS frames as samples arrive', async () => {
    const open = await ENCODERS.aac.open();
    const source = await fox(24000);
    const { parts } = encode(source, 24000, open, 1500);
    assert.equal(parts[0].length, 0);
    assert.ok(parts[1].length > 0);
    for (const part of parts.filter((p) => p.length)) frames(part);
    const silent = open(24000);
    assert.equal(frames(silent.end()).length, 1);
  });
});

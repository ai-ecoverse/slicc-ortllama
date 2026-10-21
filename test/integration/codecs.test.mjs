import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { audioSpecificConfig } from '../../src/audio/aac.js';
import { decodeFile } from '../../src/audio/decode.js';
import { createFlacEncoder, decodeFlac } from '../../src/audio/flac.js';
import { concat, decodeWav, encodePcm16 } from '../../src/audio/pcm.js';
import { fixture, foxWav } from './audio.mjs';

const TONE = 2 * Math.PI * (440 / 8000);

function rms(samples) {
  let sum = 0;
  for (const x of samples) sum += x * x;
  return Math.sqrt(sum / samples.length);
}

function correlation(samples) {
  let dot = 0;
  let energy = 0;
  samples.forEach((x, i) => {
    dot += x * Math.sin(TONE * i);
    energy += Math.sin(TONE * i) ** 2;
  });
  return dot / Math.sqrt(energy) / Math.sqrt(samples.reduce((n, x) => n + x * x, 0));
}

function flacOf(samples, rate) {
  const encoder = createFlacEncoder(rate);
  const parts = [];
  for (let at = 0; at < samples.length; at += 3000)
    parts.push(encoder.push(samples.subarray(at, at + 3000)));
  return concat([...parts, encoder.end()]);
}

const pcm16 = (samples) => new Int16Array(encodePcm16(samples).buffer);

const flacTool = spawnSync('flac', ['--version']).status === 0;

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

function pitch(samples, rate, hz) {
  let re = 0;
  let im = 0;
  samples.forEach((x, i) => {
    re += x * Math.cos((2 * Math.PI * hz * i) / rate);
    im += x * Math.sin((2 * Math.PI * hz * i) / rate);
  });
  return (
    Math.hypot(re, im) / Math.sqrt((samples.length / 2) * samples.reduce((n, x) => n + x * x, 0))
  );
}

function patched(bytes, from, to) {
  const out = bytes.slice();
  const at = Buffer.from(out).indexOf(from);
  out.set(Buffer.from(to), at);
  return out;
}

const refused = (bytes, options = {}) => decodeFile(bytes, { rate: 16000, ...options });

describe('mp3 decoding is gapless', () => {
  test('trims the encoder delay and padding the LAME header names', async () => {
    const fox = await decodeFile(await fixture('mp3'), { rate: 16000 });
    assert.equal(fox.length, 42732);
    assert.equal(fox.length / 16000, 2.67075);
    const tone = await decodeFile(await fixture('mp3', 'tone'), { rate: 8000 });
    assert.equal(tone.length, 2000);
    assert.ok(rms(tone.subarray(0, 80)) > 0.02, 'the tone starts at once');
    assert.ok(correlation(tone) > 0.95, `correlation ${correlation(tone)}`);
  });

  test('decodes an mp3 without the header as before', async () => {
    const plain = await decodeFile(await fixture('mp3', 'tone-plain'), { rate: 8000 });
    assert.equal(plain.length, 6 * 576);
    assert.ok(rms(plain.subarray(0, 1000)) < 0.005);
    assert.ok(rms(plain.subarray(1200, 2800)) > 0.02);
  });
});

describe('FLAC encoding uses LPC', () => {
  const chord = Float32Array.from(
    { length: 24000 },
    (_, i) => 0.3 * Math.sin(i * 0.05) + 0.2 * Math.sin(i * 0.131) + 0.1 * Math.sin(i * 0.377)
  );

  test('round-trips speech and tones bit for bit and shrinks tonal audio', async () => {
    const { samples, rate } = decodeWav(await foxWav());
    for (const [input, at] of [
      [samples, rate],
      [chord, 24000],
    ]) {
      const flac = flacOf(input, at);
      const back = concat(
        [...decodeFlac(flac)].map((c) => c.samples),
        Float32Array
      );
      assert.deepEqual(
        Int16Array.from(back, (x) => Math.round(x * 32768)),
        pcm16(input)
      );
      assert.ok(flac.length < input.length * 2 * 0.65, `${flac.length} bytes`);
    }
    const ratio = flacOf(chord, 24000).length / (chord.length * 2);
    assert.ok(ratio < 0.3, `chord at ${ratio}`);
  });

  test('passes flac -t and decodes to the same samples', { skip: !flacTool }, async () => {
    const dir = await mkdtemp(join(tmpdir(), 'ortllama-flac-'));
    try {
      const file = join(dir, 'chord.flac');
      await writeFile(file, flacOf(chord, 24000));
      assert.equal(spawnSync('flac', ['-t', '-s', file]).status, 0);
      const raw = spawnSync('flac', [
        '-d',
        '-s',
        '-c',
        '--force-raw-format',
        '--endian=little',
        '--sign=signed',
        file,
      ]).stdout;
      assert.deepEqual(new Int16Array(raw.buffer, raw.byteOffset, raw.length / 2), pcm16(chord));
    } finally {
      await rm(dir, { recursive: true });
    }
  });
});

describe('AAC in MP4 and ADTS, and Ogg Vorbis', () => {
  test('decode the fox to the WAV, aligned by the MP4 gapless data and the Ogg granule', async () => {
    const { samples, rate } = decodeWav(await foxWav());
    for (const [ext, name] of [
      ['m4a', 'fox'],
      ['m4a', 'fox-he'],
      ['ogg', 'fox'],
    ]) {
      const decoded = await decodeFile(await fixture(ext, name), { rate });
      assert.equal(decoded.length, samples.length, name);
      assert.ok(similarity(decoded, samples) > 0.99, `${name}.${ext}`);
    }
    const adts = await decodeFile(await fixture('aac'), { rate, type: 'audio/aac' });
    assert.equal(adts.length, 43 * 1024);
    assert.ok(similarity(adts.subarray(1024), samples) > 0.99);
  });

  test('keep the sample rate and length of 44.1 kHz stereo', async () => {
    for (const [ext, name, length] of [
      ['m4a', 'tone', 22050],
      ['ogg', 'tone', 22050],
      ['mp4', 'tone-frag', 23 * 1024],
    ]) {
      const decoded = await decodeFile(await fixture(ext, name), { rate: 44100 });
      assert.equal(decoded.length, length, name);
      assert.ok(pitch(decoded.subarray(2048, 20480), 44100, 440) > 0.99, `${name}.${ext}`);
      assert.ok(pitch(decoded.subarray(2048, 20480), 44100, 400) < 0.1, `${name}.${ext}`);
    }
  });

  test('read explicit AAC sample rates', () => {
    assert.deepEqual(audioSpecificConfig(Uint8Array.of(0x17, 0x80, 0x56, 0x22, 0x10)), {
      object: 2,
      index: 4,
      rate: 44100,
      channels: 2,
    });
  });

  test('answer clear errors for other codecs and broken files', async () => {
    const m4a = await fixture('m4a');
    await assert.rejects(refused(patched(m4a, 'mp4a', 'alac')), /MP4 codec alac is not supported/);
    await assert.rejects(refused(m4a.subarray(0, 4000)), /the MP4 file is corrupt/);
    await assert.rejects(
      refused(new Uint8Array([0xff, 0xf1, 0x50]), { type: 'audio/aac' }),
      /the AAC stream is corrupt/
    );
    const adts = await fixture('aac');
    await assert.rejects(refused(adts.subarray(0, adts.length - 100)), /the AAC stream is corrupt/);
    const ogg = await fixture('ogg');
    await assert.rejects(refused(patched(ogg, 'BCV', 'BCX')), /the Vorbis stream is corrupt/);
  });
});

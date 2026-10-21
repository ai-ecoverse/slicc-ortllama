import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { after, before, describe, test } from 'node:test';
import { transcribe } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { createAacCodec, PEERS } from '../../src/audio/aac-codec.js';
import { ENCODERS } from '../../src/audio/encode.js';
import { concat, decodeWav, resample } from '../../src/audio/pcm.js';
import { wordErrorRate } from '../../src/engines/whisper.js';
import { openCatalog, startServer } from '../../src/index.js';
import { transcriptionModel } from '../../src/pi-audio.js';
import { FOX, fixture, foxWav, pcm24 } from './audio.mjs';

const WHISPER = process.env.ORTLLAMA_TEST_WHISPER ?? 'onnx-community/whisper-tiny';

async function cached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(
      WHISPER
    );
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${WHISPER} is not in the Hugging Face cache`;

describe('Whisper transcription through pi-audio', { skip }, () => {
  let server;
  let wav;
  let model;
  let unregister;

  before(async () => {
    wav = await foxWav();
    server = await startServer({ env: { ...process.env, ORTLLAMA_API_KEY: '' }, port: 0 });
    unregister = registerOpenAI();
    model = transcriptionModel({ id: WHISPER, baseUrl: `${server.url}/v1` });
  });
  after(async () => {
    unregister?.();
    await server?.close();
  });

  test('transcribes a WAV clip', async () => {
    const result = await transcribe(model, { data: wav, mimeType: 'audio/wav' }, { apiKey: 'x' });
    assert.match(result.text.toLowerCase(), FOX);
    assert.equal(result.language, 'english');
    assert.ok(result.segments.length >= 1);
    assert.ok(result.segments[0].end > 2 && result.segments[0].end < 3.2);
    assert.equal(result.usage.audio.inputSeconds, 3);
  });

  test('transcribes raw 24 kHz PCM through the transcribe role', async () => {
    const role = transcriptionModel({ baseUrl: `${server.url}/v1` });
    const result = await transcribe(
      role,
      { data: pcm24(wav), mimeType: 'audio/pcm' },
      { apiKey: 'x' }
    );
    assert.match(result.text.toLowerCase(), FOX);
  });

  test('transcribes mp3, flac, Ogg Opus and WebM Opus', async () => {
    for (const [ext, mimeType] of [
      ['mp3', 'audio/mpeg'],
      ['flac', 'audio/flac'],
      ['opus', 'audio/opus'],
    ]) {
      const result = await transcribe(
        model,
        { data: await fixture(ext), mimeType },
        { apiKey: 'x' }
      );
      assert.match(result.text.toLowerCase(), FOX, ext);
    }
    const form = new FormData();
    form.set('file', new Blob([await fixture('webm')]), 'clip.webm');
    form.set('model', WHISPER);
    const webm = await fetch(`${server.url}/v1/audio/transcriptions`, {
      method: 'POST',
      body: form,
    });
    assert.match((await webm.json()).text.toLowerCase(), FOX);
  });

  test('transcribes M4A, HE-AAC, ADTS AAC and Ogg Vorbis to the same text as the WAV', async () => {
    const heard = async (data, mimeType) =>
      (await transcribe(model, { data, mimeType }, { apiKey: 'x' })).text;
    const text = await heard(wav, 'audio/wav');
    for (const [ext, name, mimeType] of [
      ['m4a', 'fox', 'audio/mp4'],
      ['m4a', 'fox-he', 'audio/m4a'],
      ['ogg', 'fox', 'audio/ogg'],
    ])
      assert.equal(await heard(await fixture(ext, name), mimeType), text, `${name}.${ext}`);
    assert.match((await heard(await fixture('aac'), 'audio/aac')).toLowerCase(), FOX);
    const form = new FormData();
    form.set('file', new Blob([await fixture('m4a')]), 'memo.m4a');
    form.set('model', WHISPER);
    form.set('response_format', 'verbose_json');
    const memo = await fetch(`${server.url}/v1/audio/transcriptions`, {
      method: 'POST',
      body: form,
    });
    assert.equal((await memo.json()).duration, 2.671);
  });

  test('transcribes the fox encoded to aac by each encoder peer to the same text as the WAV', async () => {
    const heard = async (data, mimeType) =>
      (await transcribe(model, { data, mimeType }, { apiKey: 'x' })).text;
    const text = await heard(wav, 'audio/wav');
    const { samples, rate } = decodeWav(wav);
    const fox = resample(samples, rate, 24000);
    for (const set of PEERS.encode) {
      const load = (name) =>
        set.includes(name)
          ? import(name)
          : Promise.reject(Object.assign(new Error(name), { code: 'engine_unavailable' }));
      const aac = createAacCodec({ scope: {}, load });
      const encoder = await (await ENCODERS.aac.open({ aac }))(24000);
      const adts = concat([await encoder.push(fox), await encoder.end()]);
      assert.equal((await heard(adts, 'audio/aac')).toLowerCase(), text.toLowerCase(), set[0]);
    }
  });

  test('decodes a clip longer than 30 seconds in windows and detects languages', async () => {
    const shipped = new URL('../../fixtures/', import.meta.url);
    const { fixtures } = JSON.parse(
      await readFile(new URL('onnx-community/whisper-tiny.json', shipped), 'utf8')
    );
    const clip = async (name) => {
      const form = new FormData();
      const data = await readFile(new URL(`audio/${name}.wav`, shipped));
      form.set('file', new Blob([data], { type: 'audio/wav' }), `${name}.wav`);
      form.set('model', WHISPER);
      form.set('response_format', 'verbose_json');
      const response = await fetch(`${server.url}/v1/audio/transcriptions`, {
        method: 'POST',
        body: form,
      });
      return response.json();
    };
    const long = await clip('lighthouse');
    const text = fixtures.find((f) => f.name === 'lighthouse').text;
    assert.ok(long.duration > 30);
    assert.ok(long.segments.at(-1).end > 30);
    assert.ok(long.segments.some((s) => s.seek > 0));
    assert.ok(wordErrorRate(text, long.text) < 0.1);
    assert.equal((await clip('french')).language, 'french');
    assert.equal((await clip('spanish')).language, 'spanish');
  });

  test('answers verbose_json and text, and refuses formats it cannot decode', async () => {
    const post = (fields, type = 'audio/wav', data = wav) => {
      const form = new FormData();
      form.set('file', new Blob([data], { type }), 'clip');
      for (const [k, v] of Object.entries(fields)) form.set(k, v);
      return fetch(`${server.url}/v1/audio/transcriptions`, { method: 'POST', body: form });
    };
    const verbose = await (
      await post({ model: WHISPER, response_format: 'verbose_json', language: 'en' })
    ).json();
    assert.equal(verbose.task, 'transcribe');
    assert.ok(verbose.duration > 2.5 && verbose.duration < 2.8);
    assert.ok(Array.isArray(verbose.segments[0].tokens));
    const text = await (await post({ model: WHISPER, response_format: 'text' })).text();
    assert.match(text.toLowerCase(), FOX);
    const amr = await post({ model: WHISPER }, 'audio/amr', new TextEncoder().encode('#!AMR\n'));
    assert.equal(amr.status, 400);
    const { error } = await amr.json();
    assert.equal(error.param, 'file');
    assert.match(error.message, /audio\/amr is not supported; send wav, mp3, flac, m4a/);
    const aac = await post({ model: WHISPER }, 'audio/aac', new Uint8Array([0xff, 0xf1, 0x50]));
    assert.equal(aac.status, 400);
    assert.match((await aac.json()).error.message, /the AAC stream is corrupt/);
    const klingon = await post({ model: WHISPER, language: 'tlh' });
    assert.equal(klingon.status, 400);
    assert.equal((await klingon.json()).error.param, 'language');
  });
});

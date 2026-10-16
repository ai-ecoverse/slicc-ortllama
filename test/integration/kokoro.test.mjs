import assert from 'node:assert/strict';
import { mkdtemp, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { speak } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { hubCache, openCatalog, startServer } from '../../src/index.js';
import { speechModel } from '../../src/pi-audio.js';

const KOKORO = 'onnx-community/Kokoro-82M-v1.0-ONNX';
const TEXT =
  'The quick brown fox jumps over the lazy dog. Then it runs into the forest, where nobody can find it.';

async function cached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(KOKORO);
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${KOKORO} is not in the Hugging Face cache`;

function pcmStats(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let sum = 0;
  for (let i = 0; i + 1 < bytes.length; i += 2) sum += (view.getInt16(i, true) / 32768) ** 2;
  const samples = Math.floor(bytes.length / 2);
  return { seconds: samples / 24000, rms: Math.sqrt(sum / Math.max(1, samples)) };
}

function plausible(seconds, text) {
  const expected = text.length / 15;
  assert.ok(
    seconds > expected / 2 && seconds < expected * 2,
    `${seconds.toFixed(2)} s for ${text.length} characters`
  );
}

async function collect(stream) {
  const events = [];
  for await (const event of stream) events.push(event);
  const failed = events.find((e) => e.type === 'error');
  assert.equal(failed, undefined, failed?.error);
  return events;
}

describe('Kokoro speech through pi-audio', { skip }, () => {
  let server;
  let unregister;

  before(async () => {
    server = await startServer({ env: { ...process.env, ORTLLAMA_API_KEY: '' }, port: 0 });
    unregister = registerOpenAI();
  });
  after(async () => {
    unregister?.();
    await server?.close();
  });

  test('speaks 24 kHz PCM16 that is about as long as the text', async () => {
    const model = speechModel({ baseUrl: `${server.url}/v1` });
    const format = model.outputFormats[0];
    const events = await collect(
      speak(model, { text: TEXT, voice: 'af_heart', format }, { apiKey: 'ortllama' })
    );
    assert.deepEqual(events[0], { type: 'start', format });
    const audio = events.filter((e) => e.type === 'audio');
    assert.ok(audio.length >= 2);
    const bytes = Buffer.concat(audio.map((e) => e.chunk.data));
    const { seconds, rms } = pcmStats(bytes);
    plausible(seconds, TEXT);
    assert.ok(rms > 0.02, `rms ${rms}`);
  });

  test('speaks WAV in a British voice', async () => {
    const model = speechModel({ baseUrl: `${server.url}/v1`, id: KOKORO });
    const format = model.outputFormats.find((f) => f.encoding === 'wav');
    const text = 'Good evening, and welcome to the news.';
    const events = await collect(
      speak(model, { text, voice: 'bf_emma', format }, { apiKey: 'ortllama' })
    );
    const [wav] = events.filter((e) => e.type === 'audio').map((e) => e.chunk.data);
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
    assert.equal(Buffer.from(wav.subarray(0, 4)).toString(), 'RIFF');
    assert.deepEqual(
      [view.getUint16(22, true), view.getUint32(24, true), view.getUint16(34, true)],
      [1, 24000, 16]
    );
    const { seconds, rms } = pcmStats(wav.subarray(44));
    plausible(seconds, text);
    assert.ok(rms > 0.02, `rms ${rms}`);
  });

  test('streams each sentence as it is ready', async () => {
    const started = performance.now();
    const response = await fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ input: TEXT, voice: 'am_michael', response_format: 'pcm' }),
    });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'audio/pcm');
    let first;
    let bytes = 0;
    for await (const chunk of response.body) {
      first ??= performance.now() - started;
      bytes += chunk.length;
    }
    const total = performance.now() - started;
    assert.ok(first < total * 0.8, `first audio after ${first} of ${total} ms`);
    plausible(bytes / 48000, TEXT);
  });

  const post = (body, signal) =>
    fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ input: 'Hi.', response_format: 'pcm', ...body }),
      signal,
    });

  test('speaks sentences that start with a dash and long ones without punctuation', async () => {
    for (const input of ['- The kettle is boiling.', '--help me, please.']) {
      const { seconds, rms } = pcmStats(
        new Uint8Array(await (await post({ input })).arrayBuffer())
      );
      plausible(seconds, input);
      assert.ok(rms > 0.02, `${input}: rms ${rms}`);
    }
    const input = 'and then we walked on '.repeat(40);
    const response = await post({ input });
    assert.equal(response.status, 200);
    plausible(new Uint8Array(await response.arrayBuffer()).length / 48000, input);
  });

  test('a hung-up client frees the queue', async () => {
    const controller = new AbortController();
    const long = await post({ input: 'and then we walked on '.repeat(180) }, controller.signal);
    await long.body.getReader().read();
    controller.abort();
    const started = performance.now();
    assert.equal((await post({ input: 'Next.' })).status, 200);
    const waited = performance.now() - started;
    assert.ok(waited < 10_000, `waited ${waited} ms`);
  });

  test('names the supported formats and voices', async () => {
    const mp3 = await post({ response_format: 'mp3' });
    assert.equal(mp3.status, 400);
    assert.match((await mp3.json()).error.message, /use one of pcm, wav/);
    const voice = await post({ voice: '../../etc/passwd' });
    assert.equal(voice.status, 400);
    assert.equal((await voice.json()).error.param, 'voice');
    const long = await post({ input: 'a'.repeat(80_000) });
    assert.equal(long.status, 400);
    assert.match((await long.json()).error.message, /at most 4096 characters/);
  });
});

describe('Kokoro without a phonemizer', { skip }, () => {
  let server;

  before(async () => {
    const real = hubCache(process.env);
    const cache = await mkdtemp(join(tmpdir(), 'ortllama-nophonemizer-'));
    const repo = 'models--onnx-community--Kokoro-82M-v1.0-ONNX';
    await symlink(join(real, repo), join(cache, repo));
    server = await startServer({
      env: {
        ...process.env,
        HF_HUB_CACHE: cache,
        ORTLLAMA_PHONEMIZERS: 'misaki',
        ORTLLAMA_API_KEY: '',
      },
      port: 0,
    });
  });
  after(() => server?.close());

  test('still lists Kokoro and answers speech with 501 naming what to install', async () => {
    const { data } = await (await fetch(`${server.url}/v1/models`)).json();
    assert.ok(data.some((m) => m.id === KOKORO && m.ortllama.engine === 'kokoro'));
    const response = await fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ input: 'Hello.', response_format: 'pcm' }),
    });
    assert.equal(response.status, 501);
    const { error } = await response.json();
    assert.equal(error.code, 'engine_unavailable');
    assert.match(
      error.message,
      /install one of: misaki \(hf download hexgrad\/misaki --repo-type dataset/
    );
    assert.match(error.message, /espeak-ng \(npm install espeak-ng\)/);
  });
});

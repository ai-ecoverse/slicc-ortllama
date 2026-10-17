import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { after, before, describe, test } from 'node:test';
import { transcribe } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { wordErrorRate } from '../../src/engines/whisper.js';
import { openCatalog, startServer } from '../../src/index.js';
import { transcriptionModel } from '../../src/pi-audio.js';
import { FOX, foxWav, pcm24 } from './audio.mjs';

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
    const mp3 = await post({ model: WHISPER }, 'audio/mpeg', new Uint8Array([0xff, 0xfb, 0x90]));
    assert.equal(mp3.status, 400);
    assert.equal((await mp3.json()).error.param, 'file');
    const klingon = await post({ model: WHISPER, language: 'tlh' });
    assert.equal(klingon.status, 400);
    assert.equal((await klingon.json()).error.param, 'language');
  });
});

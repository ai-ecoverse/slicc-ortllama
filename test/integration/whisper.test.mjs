import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { after, before, describe, test } from 'node:test';
import { transcribe } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { decodeWav, resample } from '../../src/audio/pcm.js';
import { openCatalog, startServer } from '../../src/index.js';
import { transcriptionModel } from '../../src/pi-audio.js';

const WHISPER = 'onnx-community/whisper-tiny';
const FOX = /the quick brown fox jumps? over the lazy dog/;

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

function pcm24(wav) {
  const { samples, rate } = decodeWav(wav);
  const out = resample(samples, rate, 24000);
  const bytes = new Uint8Array(out.length * 2);
  const view = new DataView(bytes.buffer);
  out.forEach((x, i) => {
    view.setInt16(i * 2, Math.max(-32768, Math.min(32767, Math.round(x * 32768))), true);
  });
  return bytes;
}

describe('Whisper transcription through pi-audio', { skip }, () => {
  let server;
  let wav;
  let model;
  let unregister;

  before(async () => {
    wav = new Uint8Array(await readFile(new URL('fixtures/fox.wav', import.meta.url)));
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

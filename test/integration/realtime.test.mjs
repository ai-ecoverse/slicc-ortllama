import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { listen } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { openCatalog, startServer } from '../../src/index.js';
import { realtimeTranscriptionModel } from '../../src/pi-audio.js';
import { FOX, foxWav, pcm24 } from './audio.mjs';

const WHISPER = 'onnx-community/whisper-tiny';
const VAD = 'onnx-community/silero-vad';
const KEY = 'sekrit';
const CHUNK = 960;

async function cached() {
  try {
    const catalog = await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } });
    await catalog.resolve(WHISPER);
    await catalog.resolve(VAD);
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have &&
  !process.env.ORTLLAMA_REQUIRE_MODEL &&
  `${WHISPER} or ${VAD} is not in the Hugging Face cache`;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function* chunks(pcm, { realtime = false, marks = {} } = {}) {
  const format = { encoding: 'pcm16', sampleRate: 24000, channels: 1 };
  const started = performance.now();
  for (let at = 0; at < pcm.length; at += CHUNK) {
    const data = pcm.subarray(at, at + CHUNK);
    const timestamp = at / 2 / 24000;
    if (realtime) await sleep(Math.max(0, timestamp * 1000 - (performance.now() - started)));
    if (
      marks.speechEnd !== undefined &&
      marks.sentEnd === undefined &&
      timestamp >= marks.speechEnd
    )
      marks.sentEnd = performance.now();
    yield { data, format, timestamp };
  }
}

function socket(url, protocols) {
  const ws = new WebSocket(url, protocols);
  const events = [];
  const waiters = [];
  ws.addEventListener('message', (m) => {
    events.push(JSON.parse(m.data));
    for (const w of waiters.splice(0)) w();
  });
  const next = async (type) => {
    for (;;) {
      const found = events.find((e) => e.type === type);
      if (found) {
        events.splice(events.indexOf(found), 1);
        return found;
      }
      await new Promise((resolve) => waiters.push(resolve));
    }
  };
  const opened = new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve);
    ws.addEventListener('error', reject);
  });
  return { ws, next, opened, send: (e) => ws.send(JSON.stringify(e)) };
}

describe('realtime transcription through pi-audio', { skip }, () => {
  let server;
  let wav;
  let model;
  let unregister;

  before(async () => {
    wav = await foxWav();
    server = await startServer({ env: { ...process.env, ORTLLAMA_API_KEY: KEY }, port: 0 });
    unregister = registerOpenAI();
    model = realtimeTranscriptionModel({ id: WHISPER, baseUrl: `${server.url}/v1` });
  });
  after(async () => {
    unregister?.();
    await server?.close();
  });

  test('listen() streams a turn with server VAD in real time', async () => {
    const pcm = pcm24(wav, { before: 0.6, after: 1.5 });
    const marks = { speechEnd: 0.6 + 2.67 };
    const stream = listen(model, chunks(pcm, { realtime: true, marks }), { apiKey: KEY });
    const seen = [];
    for await (const event of stream) {
      seen.push({ ...event, ms: performance.now() });
      if (event.type === 'final') marks.final = performance.now();
    }
    const types = seen.map((e) => e.type);
    assert.ok(types.includes('speech_start'), types.join());
    const final = seen.find((e) => e.type === 'final');
    assert.ok(final, JSON.stringify(seen.at(-1)));
    assert.match(final.text.toLowerCase(), FOX);
    const done = seen.find((e) => e.type === 'done');
    assert.match(done.transcript.text.toLowerCase(), FOX);
    const latency = Math.round(marks.final - marks.sentEnd);
    console.log(`end of the clip to the final transcript: ${latency} ms`);
    const partials = seen.filter((e) => e.type === 'partial');
    if (partials.length > 0)
      assert.ok(final.text.toLowerCase().startsWith(partials[0].text.toLowerCase().split(' ')[0]));
  });

  test('listen() keeps up when audio arrives faster than real time', async () => {
    const pcm = pcm24(wav, { before: 0.3, after: 1 });
    const result = await listen(model, chunks(pcm), { apiKey: KEY }).result();
    assert.match(result.text.toLowerCase(), FOX);
    const { models } = await (
      await fetch(`${server.url}/ortllama/ps`, { headers: { authorization: `Bearer ${KEY}` } })
    ).json();
    assert.ok(models.some((m) => m.id === WHISPER));
  });

  test('manual commits, clears and protocol errors over a raw socket', async () => {
    const url = `${server.url.replace('http', 'ws')}/v1/realtime?intent=transcription`;
    const s = socket(url, ['realtime', `openai-insecure-api-key.${KEY}`]);
    await s.opened;
    assert.equal(s.ws.protocol, 'realtime');
    const created = await s.next('session.created');
    assert.equal(created.session.type, 'transcription');
    s.send({
      type: 'session.update',
      session: {
        type: 'transcription',
        audio: {
          input: {
            format: { type: 'audio/pcm', rate: 24000 },
            transcription: { model: WHISPER, language: 'en' },
            turn_detection: null,
          },
        },
      },
    });
    const updated = await s.next('session.updated');
    assert.equal(updated.session.audio.input.turn_detection, null);
    s.send({ type: 'input_audio_buffer.commit' });
    assert.equal((await s.next('error')).error.code, 'input_audio_buffer_commit_empty');
    const pcm = pcm24(wav);
    s.send({
      type: 'input_audio_buffer.append',
      audio: Buffer.from(pcm.subarray(0, 4800)).toString('base64'),
    });
    s.send({ type: 'input_audio_buffer.clear' });
    await s.next('input_audio_buffer.cleared');
    for (let at = 0; at < pcm.length; at += 48000)
      s.send({
        type: 'input_audio_buffer.append',
        audio: Buffer.from(pcm.subarray(at, at + 48000)).toString('base64'),
      });
    s.send({ type: 'input_audio_buffer.commit' });
    const committed = await s.next('input_audio_buffer.committed');
    const completed = await s.next('conversation.item.input_audio_transcription.completed');
    assert.equal(completed.item_id, committed.item_id);
    assert.match(completed.transcript.toLowerCase(), FOX);
    s.send({ type: 'response.create' });
    assert.equal((await s.next('error')).error.code, 'invalid_event');
    s.ws.send('{nope');
    assert.equal((await s.next('error')).error.code, 'invalid_json');
    s.ws.close();
  });

  test('refuses sockets without the key, from foreign origins or for other intents', async () => {
    const base = `${server.url.replace('http', 'ws')}/v1/realtime`;
    const refused = async (url, protocols) => {
      const s = socket(url, protocols);
      await assert.rejects(s.opened);
    };
    await refused(`${base}?intent=transcription`, ['realtime']);
    await refused(`${base}?intent=conversation`, ['realtime', `openai-insecure-api-key.${KEY}`]);
    await refused(`${base}x?intent=transcription`, ['realtime', `openai-insecure-api-key.${KEY}`]);
  });
});

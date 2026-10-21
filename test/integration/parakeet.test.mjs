import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { after, before, describe, test } from 'node:test';
import { listen } from '@ai-ecoverse/pi-audio';
import { registerOpenAI } from '@ai-ecoverse/pi-audio/openai';
import { decodeFile } from '../../src/audio/decode.js';
import { wordErrorRate } from '../../src/engines/whisper.js';
import { createRuntime, openCatalog, startServer } from '../../src/index.js';
import { createFeatures, slaneyFilters } from '../../src/parakeet/features.js';
import { realtimeTranscriptionModel } from '../../src/pi-audio.js';
import { FOX, foxWav, pcm24 } from './audio.mjs';
import { readNpy } from './npy.mjs';

const PARAKEET = process.env.ORTLLAMA_TEST_PARAKEET ?? 'ai-ecoverse/Underdog-Bark-3.0-ONNX';
const VAD = 'onnx-community/silero-vad';
const CPU = { ...process.env, ORTLLAMA_MODEL_PATH: '', ORTLLAMA_EP: 'cpu' };
const HERE = new URL('fixtures/parakeet/', import.meta.url);
const SHIPPED = new URL('../../fixtures/', import.meta.url);
const WER = { q8: 0.025, q4f16: 0.03 };

const catalog = await openCatalog({ env: CPU });
const resolve = (id) => catalog.resolve(id).catch(() => null);
const model = await resolve(PARAKEET);
const vad = await resolve(VAD);
const skip =
  !model && !process.env.ORTLLAMA_REQUIRE_MODEL && `${PARAKEET} is not in the Hugging Face cache`;
const expected = JSON.parse(await readFile(new URL('expected.json', HERE), 'utf8'));
const clipNames = Object.keys(expected.clips);

const flac = async (bytes) => decodeFile(new Uint8Array(bytes), { type: 'audio/flac' });
const inModel = (rel) => model.files.path(rel);
const hasAll = async (rels) => {
  for (const rel of rels) if (!(await model.files.exists(rel))) return false;
  return true;
};

function maxDiff(a, b) {
  assert.equal(a.length, b.length);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

function corpusWer(references, hypotheses) {
  let errors = 0;
  let words = 0;
  references.forEach((ref, i) => {
    const n = ref.split(/\s+/).filter(Boolean).length;
    errors += wordErrorRate(ref, hypotheses[i]) * n;
    words += n;
  });
  return errors / words;
}

async function withVariant(variant, run) {
  const runtime = createRuntime({ env: CPU, keepAlive: '0' });
  try {
    const entry = await runtime.acquire(await catalog.resolve(`${PARAKEET}:${variant}`));
    return await run(entry.instance);
  } finally {
    await runtime.close();
  }
}

describe('Parakeet TDT transcription', { skip }, () => {
  let clips;

  before(async () => {
    const rels = clipNames.map((n) => `export/clips/${n}.flac`);
    clips = (await hasAll(rels))
      ? Object.fromEntries(
          await Promise.all(
            clipNames.map(async (n, i) => [n, await flac(await readFile(inModel(rels[i])))])
          )
        )
      : null;
  });

  test('lists the export on the parakeet engine with q8 as its default', async () => {
    const entry = (await catalog.list()).find((m) => m.id === PARAKEET);
    assert.equal(entry.engine, 'parakeet');
    assert.equal(entry.capability, 'transcription');
    assert.equal(entry.defaultVariant, 'q8');
    for (const v of ['fp32', 'fp16', 'q8', 'q4f16']) assert.ok(entry.variants.includes(v), v);
    assert.deepEqual(entry.inputModalities, ['audio']);
    assert.ok(model.engine.include(model.info, 'q8').includes('NOTICE'));
  });

  test('computes the Slaney filters and log-mel features of the export fixture', async (t) => {
    const rels = ['mel_filters.npy', 'input_features.npy', 'attention_mask.npy'];
    if (!(await hasAll(rels.map((r) => `fixture/${r}`))))
      return t.skip('no fixture/ in the export');
    const preprocessor = await model.files.json('preprocessor_config.json');
    const filters = await readNpy(inModel('fixture/mel_filters.npy'));
    assert.ok(maxDiff(slaneyFilters(512, 128, 16000, 0, 8000), filters.data) <= 1e-9);
    const audio = await flac(await readFile(new URL('s01.flac', HERE)));
    const got = createFeatures(preprocessor).compute(audio);
    const want = await readNpy(inModel('fixture/input_features.npy'));
    const mask = await readNpy(inModel('fixture/attention_mask.npy'));
    assert.deepEqual(want.shape, [1, got.frames, 128]);
    assert.ok(maxDiff(got.features, want.data) <= 2e-4);
    assert.deepEqual([...got.mask], [...mask.data]);
  });

  test('encodes the fixture clip as the fp32 export did', async (t) => {
    if (!(await hasAll(['fixture/encoder_outputs.npy'])))
      return t.skip('no fixture/ in the export');
    const want = await readNpy(inModel('fixture/encoder_outputs.npy'));
    const audio = await flac(await readFile(new URL('s01.flac', HERE)));
    await withVariant('fp32', async (instance) => {
      const got = await instance.encode(audio);
      assert.deepEqual([1, got.frames, 640], want.shape);
      assert.ok(maxDiff(got.data, want.data) <= 1e-4);
    });
  });

  for (const variant of ['fp32', 'fp16'])
    test(`decodes every parity clip token for token on ${variant}`, async (t) => {
      if (!clips) return t.skip('no export/clips/ in the export');
      await withVariant(variant, async (instance) => {
        for (const name of clipNames) {
          const steps = await instance.steps(clips[name]);
          const want = expected.clips[name][variant];
          assert.deepEqual(
            steps.map((s) => s.token),
            want.tokens,
            `${name} tokens`
          );
          assert.deepEqual(
            steps.map((s) => s.duration),
            want.durations,
            `${name} durations`
          );
          assert.equal((await instance.transcribe(clips[name])).text, want.text, name);
        }
      });
    });

  test('stamps every token inside the window it was heard in', async (t) => {
    if (!clips) return t.skip('no export/clips/ in the export');
    const trimmed = clipNames.map((n) => {
      const c = clips[n];
      let from = 0;
      let to = c.length;
      while (from < to && Math.abs(c[from]) < 0.02) from++;
      while (to > from && Math.abs(c[to - 1]) < 0.02) to--;
      return c.subarray(from, to);
    });
    const audio = new Float32Array(trimmed.reduce((n, c) => n + c.length, 0));
    let at = 0;
    for (const c of trimmed) {
      audio.set(c, at);
      at += c.length;
    }
    for (let i = 0; i < audio.length; i++) audio[i] += 0.01 * Math.sin(i * 1.7);
    await withVariant('q8', async (instance) => {
      const { segments } = await instance.transcribe(audio);
      const seeks = [...new Set(segments.map((s) => s.seek))];
      assert.ok(seeks.length >= 3, `${seeks}`);
      for (const s of segments) {
        const start = (s.seek * 160) / 16000;
        const next = seeks.find((k) => k > s.seek);
        const end = next === undefined ? audio.length / 16000 : (next * 160) / 16000;
        assert.ok(s.start >= start - 1e-6 && s.end <= end + 1e-6, JSON.stringify(s));
      }
      for (let i = 1; i < segments.length; i++)
        assert.ok(segments[i].start >= segments[i - 1].end - 1e-6, `segment ${i}`);
    });
  });

  for (const [variant, bound] of Object.entries(WER))
    test(`keeps ${variant} within ${bound * 100}% WER of PyTorch`, async (t) => {
      if (!clips) return t.skip('no export/clips/ in the export');
      await withVariant(variant, async (instance) => {
        const heard = [];
        for (const name of clipNames) heard.push((await instance.transcribe(clips[name])).text);
        const wer = corpusWer(
          clipNames.map((n) => expected.clips[n].pytorch.text),
          heard
        );
        assert.ok(wer <= bound, `WER ${wer}`);
      });
    });
});

describe('Parakeet TDT over HTTP', { skip }, () => {
  let server;
  let say;
  let unregister;

  const post = (fields, data = say, name = 's01.flac') => {
    const form = new FormData();
    form.set('file', new Blob([data]), name);
    form.set('model', PARAKEET);
    for (const [k, v] of Object.entries(fields)) form.set(k, v);
    return fetch(`${server.url}/v1/audio/transcriptions`, { method: 'POST', body: form });
  };

  before(async () => {
    say = await readFile(new URL('s01.flac', HERE));
    server = await startServer({
      env: { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_MODEL_PATH: '' },
      port: 0,
    });
    unregister = registerOpenAI();
  });
  after(async () => {
    unregister?.();
    await server?.close();
  });

  test('answers every response format with TDT timestamps', async () => {
    const text = expected.clips.s01.q8.text;
    const json = await (await post({})).json();
    assert.equal(json.text, text);
    assert.equal(json.language, 'english');
    const verbose = await (await post({ response_format: 'verbose_json' })).json();
    assert.equal(verbose.task, 'transcribe');
    assert.equal(verbose.text, text);
    assert.ok(verbose.duration > 2.6 && verbose.duration < 2.7);
    const [segment] = verbose.segments;
    assert.equal(verbose.segments.length, 1);
    assert.equal(segment.text.trim(), text);
    assert.ok(segment.start >= 0 && segment.start < 0.3, `${segment.start}`);
    assert.ok(segment.end > 2 && segment.end <= verbose.duration, `${segment.end}`);
    assert.ok(segment.tokens.length > 5 && !segment.tokens.includes(8192));
    assert.ok(segment.avg_logprob < 0 && segment.avg_logprob > -1);
    assert.equal(await (await post({ response_format: 'text' })).text(), `${text}\n`);
    const srt = await (await post({ response_format: 'srt' })).text();
    assert.match(srt, /^1\n00:00:00,\d{3} --> 00:00:02,\d{3}\nthe quick brown fox/);
    const vtt = await (await post({ response_format: 'vtt' })).text();
    assert.match(vtt, /^WEBVTT\n\n00:00:00\.\d{3} --> 00:00:02\.\d{3}\nthe quick brown fox/);
  });

  test('decodes a clip longer than 30 seconds in windows cut at a pause', async () => {
    const { fixtures } = JSON.parse(
      await readFile(new URL('onnx-community/whisper-tiny.json', SHIPPED), 'utf8')
    );
    const data = await readFile(new URL('audio/lighthouse.wav', SHIPPED));
    const long = await (await post({ response_format: 'verbose_json' }, data, 'l.wav')).json();
    assert.ok(long.duration > 30);
    assert.ok(long.segments.length > 2);
    assert.ok(long.segments.some((s) => s.seek > 0));
    assert.ok(long.segments.at(-1).end > 30 && long.segments.at(-1).end <= long.duration);
    for (let i = 1; i < long.segments.length; i++)
      assert.ok(long.segments[i].start >= long.segments[i - 1].end - 0.01);
    const script = fixtures.find((f) => f.name === 'lighthouse').text;
    assert.ok(wordErrorRate(script, long.text) < 0.1, long.text);
  });

  test('accepts English and refuses other languages', async () => {
    assert.equal((await post({ language: 'en' })).status, 200);
    const french = await post({ language: 'fr' });
    assert.equal(french.status, 400);
    const { error } = await french.json();
    assert.equal(error.param, 'language');
    assert.match(error.message, /only transcribes English/);
  });

  test('transcribes a realtime turn that Silero VAD detects', {
    skip: !vad && 'no VAD',
  }, async () => {
    const pcm = pcm24(await foxWav(), { before: 0.5, after: 1 });
    const format = { encoding: 'pcm16', sampleRate: 24000, channels: 1 };
    async function* chunks() {
      for (let at = 0; at < pcm.length; at += 960)
        yield { data: pcm.subarray(at, at + 960), format, timestamp: at / 48000 };
    }
    const live = realtimeTranscriptionModel({ id: PARAKEET, baseUrl: `${server.url}/v1` });
    const result = await listen(live, chunks(), { apiKey: 'x' }).result();
    assert.match(result.text.toLowerCase(), FOX);
  });
});

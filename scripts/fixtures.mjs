import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toRecord } from '@ai-ecoverse/kev.js';
import { resample, toMono16k } from '../src/audio/pcm.js';
import { toRecord as decisionRecord } from '../src/engines/decisions.js';
import { fixtureRequest } from '../src/engines/decoder/index.js';
import { spectrum } from '../src/engines/kokoro.js';
import { speechProbabilities } from '../src/engines/silero.js';
import { wordErrorRate } from '../src/engines/whisper.js';
import { imageFromUrl } from '../src/images.js';
import { createRuntime, openCatalog } from '../src/index.js';

const ROOT = fileURLToPath(new URL('../fixtures/', import.meta.url));
const AUDIO = join(ROOT, 'audio');

const CLIPS = [
  {
    name: 'fox',
    file: 'fox.wav',
    text: 'The quick brown fox jumps over the lazy dog.',
    detect: 'english',
  },
  {
    name: 'lighthouse',
    file: 'lighthouse.wav',
    voice: 'af_heart',
    detect: 'english',
    text: 'The old lighthouse stood at the edge of the cliff for more than a hundred years. Every night its keeper climbed the narrow stairs, trimmed the wick, and watched the beam sweep across the dark water. Ships passing in the fog relied on that steady light to find their way around the rocks. One winter a storm broke the windows and flooded the lower rooms, but the keeper kept the lamp burning until morning. When the sun rose, the fishermen in the village saw the tower still standing, and they carried bread and coffee up the hill to thank him. Years later the light was replaced by an automatic beacon.',
  },
  {
    name: 'french',
    file: 'french.wav',
    voice: 'ff_siwis',
    detect: 'french',
    text: "Bonjour à tous. Aujourd'hui il fait très beau à Paris, et nous allons nous promener le long de la Seine.",
  },
  {
    name: 'spanish',
    file: 'spanish.wav',
    voice: 'ef_dora',
    detect: 'spanish',
    text: 'Buenos días. Mañana vamos a viajar a Madrid en tren para visitar a nuestros amigos.',
  },
];

const VAD_CLIPS = ['fox', 'french'];

const SENTENCES = [
  {
    name: 'heart',
    voice: 'af_heart',
    text: 'The museum opens at nine, and the tickets cost twelve dollars.',
  },
  {
    name: 'george',
    voice: 'bm_george',
    text: 'Could you bring the umbrella? It looks like rain this afternoon.',
  },
  { name: 'siwis', voice: 'ff_siwis', text: 'Le train pour Lyon part dans dix minutes.' },
];

const PROMPTS = [
  {
    name: 'capital',
    messages: [{ role: 'user', content: 'What is the capital of France? Answer in one sentence.' }],
  },
  {
    name: 'haiku',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Write a haiku about autumn leaves.' },
    ],
  },
  {
    name: 'math',
    messages: [{ role: 'user', content: 'What is 17 times 23? Show the steps briefly.' }],
  },
];

const TOKENS = 24;

const RED_CIRCLE = new URL('../test/integration/fixtures/red-circle.png', import.meta.url);

const VISION = [
  {
    name: 'red-circle',
    request: {
      state: 'Look at the picture.',
      questions: {
        circle: { type: 'noul', instructions: 'Does the picture show a circle?' },
        color: {
          type: 'choice',
          instructions: 'What color is the shape?',
          criteria: { red: null, green: null, blue: null },
        },
      },
    },
  },
];

const env = {
  ...process.env,
  ORTLLAMA_MODEL_PATH: '',
  ORTLLAMA_EP: process.env.ORTLLAMA_EP ?? 'cpu',
};
const catalog = await openCatalog({ env });
const runtime = createRuntime({ env, keepAlive: '0' });

async function open(id, fixtures = []) {
  const resolved = await catalog.resolve(id);
  const { engine } = resolved;
  const overrides = engine.checkOverrides(fixtures);
  const entry = await runtime.acquire({
    ...resolved,
    engine: { ...engine, load: (o) => engine.load({ ...o, overrides }) },
  });
  return { resolved, entry, instance: entry.instance };
}

function wav8k(samples, rate) {
  const s = resample(samples, rate, 8000);
  const out = Buffer.alloc(44 + s.length);
  out.write('RIFF', 0);
  out.writeUInt32LE(36 + s.length, 4);
  out.write('WAVEfmt ', 8);
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(1, 22);
  out.writeUInt32LE(8000, 24);
  out.writeUInt32LE(8000, 28);
  out.writeUInt16LE(1, 32);
  out.writeUInt16LE(8, 34);
  out.write('data', 36);
  out.writeUInt32LE(s.length, 40);
  for (let i = 0; i < s.length; i++)
    out[44 + i] = Math.max(
      0,
      Math.min(255, Math.round(Math.max(-1, Math.min(1, s[i])) * 127) + 128)
    );
  return out;
}

async function audio() {
  const { instance } = await open('speech');
  for (const clip of CLIPS.filter((c) => c.voice)) {
    const parts = [];
    for await (const chunk of instance.speak(clip.text, { voice: clip.voice })) parts.push(chunk);
    const samples = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {
      samples.set(p, at);
      at += p.length;
    }
    await writeFile(join(AUDIO, clip.file), wav8k(samples, instance.sampleRate));
    console.log(clip.file, (samples.length / instance.sampleRate).toFixed(1), 's');
  }
}

const clipAudio = async (clip) =>
  toMono16k(new Uint8Array(await readFile(join(AUDIO, clip.file))), 'audio/wav');

async function whisperFixtures(instance) {
  const out = [];
  for (const clip of CLIPS) {
    const got = await instance.transcribe(await clipAudio(clip));
    const wer = wordErrorRate(clip.text, got.text);
    out.push({
      name: clip.name,
      file: join(AUDIO, clip.file),
      text: clip.text,
      detect: clip.detect,
      wer: +wer.toFixed(3),
      heard: got.text,
    });
  }
  return out;
}

async function sileroFixtures(instance) {
  const out = [];
  for (const name of VAD_CLIPS) {
    const clip = CLIPS.find((c) => c.name === name);
    const probs = await speechProbabilities(instance, await clipAudio(clip));
    out.push({ name, file: join(AUDIO, clip.file), probs: probs.map((p) => +p.toFixed(4)) });
  }
  return out;
}

async function kokoroFixtures(instance) {
  const out = [];
  for (const s of SENTENCES) {
    const phonemes = await instance.phonemes(s.text, s.voice[0]);
    const [ids] = await instance.tokens(s.text, { voice: s.voice });
    const samples = await instance.synthesize(ids, { voice: s.voice });
    out.push({ ...s, phonemes, ids, ...spectrum(samples) });
  }
  return out;
}

async function decoderFixtures(instance) {
  const out = [];
  for (const p of PROMPTS) {
    const steps = await instance.score(fixtureRequest(p), [], TOKENS);
    out.push({
      ...p,
      text: instance.decode(steps.map((s) => s.token)),
      tokens: steps.map((s) => s.token),
      logprobs: steps.map((s) => +s.logprob.toFixed(4)),
      margins: steps.map((s) => +s.margin.toFixed(3)),
    });
  }
  return out;
}

async function systemoneFixtures(instance, resolved, recordOf = toRecord, minOptions = 1) {
  const requests = JSON.parse(
    await readFile(new URL('./systemone-requests.json', import.meta.url), 'utf8')
  );
  const out = [];
  const usable = (request) => {
    const { record } = recordOf(request);
    return { ...record, questions: record.questions.filter((q) => q.options.length >= minOptions) };
  };
  for (const { name, request } of requests) {
    const record = usable(request);
    out.push({
      name,
      record,
      probs: (await instance.probs(record)).map((q) => q.map((p) => +p.toFixed(5))),
    });
  }
  if (resolved.info.inputModalities.includes('image')) {
    const image = `data:image/png;base64,${(await readFile(RED_CIRCLE)).toString('base64')}`;
    for (const { name, request } of VISION) {
      const record = usable(request);
      const probs = await instance.probs(record, await imageFromUrl(image, { param: 'image' }));
      out.push({ name, record, image, probs: probs.map((q) => q.map((p) => +p.toFixed(5))) });
    }
  }
  return out;
}

const BUILDERS = {
  whisper: whisperFixtures,
  silero: sileroFixtures,
  kokoro: kokoroFixtures,
  decoder: decoderFixtures,
  systemone: systemoneFixtures,
  jev: (instance, resolved) => systemoneFixtures(instance, resolved, decisionRecord, 2),
};

async function fixturesFor(id) {
  const { resolved, entry, instance } = await open(id);
  const started = performance.now();
  const fixtures = await BUILDERS[resolved.engine.name](instance, resolved);
  const path = join(ROOT, `${resolved.id}.json`);
  await mkdir(dirname(path), { recursive: true });
  const about = {
    model: resolved.id,
    variant: resolved.variant,
    revision: resolved.revision,
    executionProviders: entry.executionProviders,
    generator: 'scripts/fixtures.mjs',
  };
  for (const f of fixtures) if (f.file) f.file = relative(dirname(path), f.file);
  await writeFile(path, `${JSON.stringify({ ...about, fixtures })}\n`);
  console.log(
    `${resolved.id}:${resolved.variant} → ${path} in ${Math.round(performance.now() - started)} ms`
  );
  await runtime.close();
}

const [command, ...models] = process.argv.slice(2);
if (command === 'audio') await audio();
else for (const id of [command, ...models].filter(Boolean)) await fixturesFor(id);
await runtime.close();
process.exit(0);

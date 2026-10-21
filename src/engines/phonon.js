import { badRequest } from '../http/errors.js';
import { MAX_FIT_TOKENS, normalizeEnglish, plan } from '../phonon/text.js';
import { createTokenizer } from '../phonon/tokenizer.js';
import { spectralDistance, spectrum } from './kokoro.js';

export const SAMPLE_RATE = 24000;
export const PHONON_VARIANTS = { fp32: '', q8: '_quantized', q4: '_q4', fp16: '_fp16' };

const GRAPHS = ['text_conditioner', 'flow_lm_main', 'flow_lm_flow', 'mimi_decoder'];
const DEFAULT_VOICE = 'easton';
const TEMPERATURE = 0.3;
const SEED = 0x2a2a2a2a;
const DECODE_BATCH = 16;
const PIPELINE = ['config.json', 'tokenizer.json'];

const dispose = (...tensors) => {
  for (const t of tensors) t?.dispose?.();
};

const graphFile = (name, suffix) => `onnx/${name}${suffix}.onnx`;
const voiceFile = (voice) => `voices/${voice}.bin`;

async function graphsFor(files, variant) {
  const suffix = PHONON_VARIANTS[variant];
  if (!(await files.exists(graphFile('flow_lm_main', suffix)))) return null;
  const out = {};
  for (const name of GRAPHS) {
    if (await files.exists(graphFile(name, suffix))) out[name] = graphFile(name, suffix);
    else if (await files.exists(graphFile(name, ''))) out[name] = graphFile(name, '');
    else return null;
  }
  return out;
}

export function seededNormal(seed, temperature) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 1) / 4294967297;
  };
  const std = Math.sqrt(temperature);
  return (n) => {
    const out = new Float32Array(n);
    if (std === 0) return out;
    for (let i = 0; i < n; i += 2) {
      const r = Math.sqrt(-2 * Math.log(uniform()));
      const a = 2 * Math.PI * uniform();
      out[i] = r * Math.cos(a) * std;
      if (i + 1 < n) out[i + 1] = r * Math.sin(a) * std;
    }
    return out;
  };
}

export function eosPolicy(framesAfterEos) {
  let countdown = null;
  return (eos) => {
    if (eos && countdown === null) countdown = framesAfterEos;
    if (countdown === null) return false;
    if (countdown === 0) return true;
    countdown--;
    return false;
  };
}

export function concat(parts) {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

export function voiceSlug(voice) {
  return String(voice)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

async function openSessions(ort, files, graphs, options) {
  const sessions = {};
  try {
    for (const name of GRAPHS)
      sessions[name] = await ort.InferenceSession.create(await files.read(graphs[name]), options);
  } catch (error) {
    for (const s of Object.values(sessions)) await s.release?.();
    throw error;
  }
  return sessions;
}

function flowLm(ort, session, { num_layers: layers, num_heads: heads, head_dim: headDim }) {
  return {
    empty: () =>
      Array.from(
        { length: 2 * layers },
        () => new ort.Tensor('float32', new Float32Array(0), [1, heads, 0, headDim])
      ),
    async run(embeds, past) {
      const feeds = { inputs_embeds: embeds };
      for (let i = 0; i < layers; i++) {
        feeds[`past_key_values.${i}.key`] = past[2 * i];
        feeds[`past_key_values.${i}.value`] = past[2 * i + 1];
      }
      const out = await session.run(feeds);
      const present = [];
      for (let i = 0; i < layers; i++)
        present.push(out[`present.${i}.key`], out[`present.${i}.value`]);
      const eos = (await out.eos_logit.getData())[0];
      dispose(out.eos_logit);
      return { hidden: out.hidden, eos, present };
    },
  };
}

function mimiDecoder(ort, session, { mimi_state: shapes, latent_dim: latentDim }) {
  const state = {};
  for (const [n, shape] of shapes)
    state[n] = new ort.Tensor('float32', new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  let offset = new ort.Tensor('int64', BigInt64Array.from([0n]), []);
  const replace = (old, next) => {
    if (old !== next) dispose(old);
    return next;
  };
  const decode = async (latents) => {
    const data = new Float32Array(latents.length * latentDim);
    for (const [i, l] of latents.entries()) data.set(l, i * latentDim);
    const feeds = {
      latent: new ort.Tensor('float32', data, [1, latents.length, latentDim]),
      offset,
    };
    for (const [n] of shapes) feeds[`state.${n}`] = state[n];
    const out = await session.run(feeds);
    dispose(feeds.latent);
    offset = replace(offset, out.next_offset);
    for (const [n] of shapes) state[n] = replace(state[n], out[`next_state.${n}`]);
    const audio = new Float32Array(await out.audio.getData());
    dispose(out.audio);
    return audio;
  };
  decode.close = () => dispose(offset, ...Object.values(state));
  return decode;
}

function createSpeaker({ ort, files, sessions, config }) {
  const meta = config.onnx;
  const lm = flowLm(ort, sessions.flow_lm_main, meta);
  const width = config.flow_lm.d_model;
  const prompts = new Map();

  function prompt(voice) {
    if (!prompts.has(voice)) {
      const pending = (async () => {
        const bytes = await files.read(voiceFile(voice));
        const data = new Float32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
        );
        const embeds = new ort.Tensor('float32', data, [1, data.length / width, width]);
        const empty = lm.empty();
        const out = await lm.run(embeds, empty);
        dispose(embeds, out.hidden, ...empty);
        return out.present;
      })();
      prompts.set(voice, pending);
      pending.catch(() => prompts.delete(voice));
    }
    return prompts.get(voice);
  }

  async function frame(state, draw, f) {
    const out = await lm.run(state.step, state.past);
    dispose(state.step, ...state.past);
    state.past = out.present;
    const z = new ort.Tensor('float32', draw(meta.latent_dim, f), [1, meta.latent_dim]);
    const flow = await sessions.flow_lm_flow.run({ hidden: out.hidden, noise: z });
    dispose(out.hidden, z);
    state.step = flow.next_embeds;
    const latent = new Float32Array(await flow.latent.getData());
    dispose(flow.latent);
    return { latent, eos: out.eos };
  }

  async function generate(chunk, voice, options = {}) {
    const { temperature, seed, noise, signal } = options;
    const batch = options.batch ?? DECODE_BATCH;
    const draw =
      noise ??
      (temperature === undefined && seed === undefined ? chunk.noise : null) ??
      seededNormal(seed ?? SEED, temperature ?? TEMPERATURE);
    const ids = BigInt64Array.from(chunk.tokens, BigInt);
    const input = new ort.Tensor('int64', ids, [1, ids.length]);
    const text = await sessions.text_conditioner.run({ input_ids: input });
    const first = await lm.run(text.inputs_embeds, await prompt(voice));
    dispose(input, text.inputs_embeds, first.hidden);
    const state = { past: first.present, step: text.bos_embeds };
    const decode = mimiDecoder(ort, sessions.mimi_decoder, meta);
    const stop = eosPolicy(chunk.framesAfterEos);
    const frames = [];
    const latents = [];
    const eos = [];
    let pending = 0;
    try {
      for (let f = 0; f < chunk.frameBudget; f++) {
        signal?.throwIfAborted();
        const out = await frame(state, draw, f);
        latents.push(out.latent);
        eos.push(out.eos);
        pending++;
        const done = stop(out.eos > config.eos_threshold);
        if (pending >= batch || done) {
          frames.push(await decode(latents.slice(latents.length - pending)));
          pending = 0;
        }
        if (done) break;
      }
      if (pending) frames.push(await decode(latents.slice(latents.length - pending)));
    } finally {
      dispose(state.step, ...state.past);
      decode.close();
    }
    return { audio: concat(frames), latents, eos };
  }

  async function release() {
    for (const pending of prompts.values()) dispose(...(await pending.catch(() => [])));
    prompts.clear();
  }

  return { generate, release };
}

async function availableVoices(files, meta) {
  const voices = [];
  for (const v of meta.voices ?? []) if (await files.exists(voiceFile(v.name))) voices.push(v.name);
  return voices;
}

export const phonon = {
  name: 'phonon',
  capability: 'speech',
  splits: true,
  providers: { node: ['cpu'], web: ['wasm'] },
  async detect(files) {
    if (!(await files.exists('config.json'))) return null;
    let config;
    try {
      config = await files.json('config.json');
    } catch {
      return null;
    }
    if (!config?.flow_lm || !config?.mimi || !config?.onnx) return null;
    const graphs = {};
    for (const variant of Object.keys(PHONON_VARIANTS)) {
      const found = await graphsFor(files, variant);
      if (found) graphs[variant] = found;
    }
    const variants = Object.keys(graphs);
    if (variants.length === 0) return null;
    return {
      variants,
      defaultVariant: variants[0],
      graphs,
      inputModalities: ['text'],
      outputModalities: ['audio'],
      contextLength: MAX_FIT_TOKENS,
    };
  },
  async missing(files, info, variant) {
    const graphs = info.graphs?.[variant] ?? {};
    const config = (await files.exists('config.json')) ? await files.json('config.json') : {};
    const needed = [
      ...PIPELINE,
      ...GRAPHS.map((g) => graphs[g] ?? graphFile(g, PHONON_VARIANTS[variant])),
      voiceFile(config.onnx?.default_voice ?? DEFAULT_VOICE),
    ];
    const out = [];
    for (const rel of needed) if (!(await files.exists(rel))) out.push(rel);
    return out;
  },
  include: (info, variant) => [
    ...PIPELINE,
    ...Object.values(info.graphs?.[variant] ?? {}),
    'voices/*',
  ],
  async load({ backend, files, variant, info, overrides = {} }) {
    const { ort } = backend;
    const graphs = info?.graphs?.[variant] ?? (await graphsFor(files, variant));
    const [config, tokenizerJson] = await Promise.all([
      files.json('config.json'),
      files.json('tokenizer.json'),
    ]);
    const sessions = await openSessions(ort, files, graphs, {
      executionProviders: backend.executionProviders,
      ...backend.sessionOptions,
      ...overrides,
    });
    const tokenizer = createTokenizer(tokenizerJson);
    const voices = await availableVoices(files, config.onnx);
    const defaultVoice = voices.includes(config.onnx.default_voice)
      ? config.onnx.default_voice
      : voices[0];
    const speaker = createSpeaker({ ort, files, sessions, config });

    function voiceName(voice) {
      const name = voiceSlug(voice ?? defaultVoice);
      if (voices.includes(name)) return name;
      throw badRequest(`voice ${voice} is not available; use one of ${voices.join(', ')}`, 'voice');
    }

    function tokens(text, { voice } = {}) {
      voiceName(voice);
      const chunks = plan(tokenizer, text, { frameRate: config.mimi.frame_rate });
      if (chunks.length === 0) throw badRequest('input has nothing to say', 'input');
      const noise = seededNormal(SEED, TEMPERATURE);
      for (const c of chunks) c.noise = noise;
      return chunks;
    }

    const generate = (chunk, { voice, ...options } = {}) =>
      speaker.generate(chunk, voiceName(voice), options);

    async function synthesize(chunk, { speed = 1, ...options } = {}) {
      if (speed !== 1) throw badRequest('phonon speaks at one speed; use speed 1', 'speed');
      return (await generate(chunk, options)).audio;
    }

    return {
      voices,
      defaultVoice,
      sampleRate: SAMPLE_RATE,
      normalize: normalizeEnglish,
      tokenizer,
      tokens: async (text, options) => tokens(text, options),
      generate,
      synthesize,
      async *speak(text, { signal, ...options } = {}) {
        for (const chunk of tokens(text, options)) {
          signal?.throwIfAborted();
          yield await synthesize(chunk, { ...options, signal });
        }
      },
      async release() {
        await speaker.release();
        for (const s of Object.values(sessions)) await s.release?.();
      },
    };
  },
};

phonon.checkOverrides = () => ({});

const SMOKE =
  'Phonon speaks from the Hugging Face cache. This sentence checks that the voice sounds right.';

async function compareSpeech(instance, fixtures, now) {
  const results = [];
  for (const f of fixtures) {
    const chunks = await instance.tokens(f.text, { voice: f.voice });
    const t = now();
    const parts = [];
    for (const c of chunks)
      parts.push(await instance.synthesize(c, { voice: f.voice, temperature: 0 }));
    const ms = now() - t;
    const audio = concat(parts);
    results.push({
      name: f.name,
      ...spectralDistance(spectrum(audio), f),
      tokens: JSON.stringify(chunks.flatMap((c) => c.tokens)) === JSON.stringify(f.ids),
      rtf: ms / 1000 / (audio.length / SAMPLE_RATE),
    });
  }
  const worst = results.reduce((a, b) => (b.db > a.db ? b : a), results[0]);
  const mean = results.reduce((s, r) => s + r.db, 0) / Math.max(1, results.length);
  const same = results.filter((r) => r.tokens).length;
  return {
    results,
    summary: `fixtures: ${results.length} sentences, mean mel distance ${mean.toFixed(2)} dB, worst ${worst?.db.toFixed(2) ?? '0'} dB (${worst?.name ?? '-'}), tokens match on ${same}`,
    ok: results.every((r) => r.tokens && r.db <= 2 && Math.abs(r.length - 1) <= 0.05),
  };
}

phonon.check = async (instance, { fixtures, now = () => performance.now() } = {}) => {
  const started = now();
  let samples = 0;
  let peak = 0;
  let first = null;
  for await (const audio of instance.speak(SMOKE)) {
    first ??= now() - started;
    samples += audio.length;
    peak = audio.reduce((m, s) => Math.max(m, Math.abs(s)), peak);
  }
  const requestMs = now() - started;
  const audioSeconds = samples / SAMPLE_RATE;
  const smoke = {
    ok: audioSeconds > 2 && peak > 0.01,
    requestMs,
    firstAudioMs: first,
    audioSeconds,
    realTimeFactor: requestMs / 1000 / audioSeconds,
    peak,
  };
  if (!fixtures) return smoke;
  const compared = await compareSpeech(instance, fixtures, now);
  return { ...smoke, ...compared, ok: smoke.ok && compared.ok };
};

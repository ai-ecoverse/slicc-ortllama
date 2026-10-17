import { hubCache } from '../hf/cache.js';
import { badRequest } from '../http/errors.js';
import { loadPhonemizers, phonemizerOrder } from '../kokoro/phonemizers/index.js';
import { fixPhonemes, normalizeEnglish, splitPunctuation } from '../kokoro/text.js';
import { DEFAULT_VOICE, LANGUAGES, VOICES } from '../kokoro/voices.js';
import { createMel } from '../whisper/mel.js';

export const SAMPLE_RATE = 24000;

const MAX_TOKENS = 510;
const STYLE = 256;
const SPACE = ' ';

const VARIANTS = {
  fp32: 'onnx/model.onnx',
  fp16: 'onnx/model_fp16.onnx',
  q8: 'onnx/model_quantized.onnx',
  q8f16: 'onnx/model_q8f16.onnx',
  q4: 'onnx/model_q4.onnx',
  q4f16: 'onnx/model_q4f16.onnx',
  uint8: 'onnx/model_uint8.onnx',
  uint8f16: 'onnx/model_uint8f16.onnx',
};

const voiceFile = (voice) => `voices/${voice}.bin`;

function needed(variant) {
  return ['config.json', 'tokenizer.json', VARIANTS[variant], voiceFile(DEFAULT_VOICE)];
}

export function chunkTokens(ids, space, max = MAX_TOKENS) {
  const chunks = [];
  let rest = ids;
  while (rest.length > max) {
    const cut = rest.lastIndexOf(space, max);
    const at = cut > 0 ? cut : max;
    chunks.push(rest.slice(0, at));
    rest = rest.slice(cut > 0 ? at + 1 : at);
  }
  if (rest.length > 0) chunks.push(rest);
  return chunks;
}

async function availableVoices(files) {
  const out = [];
  for (const voice of VOICES) if (await files.exists(voiceFile(voice))) out.push(voice);
  return out;
}

function environmentPhonemizers(env) {
  return loadPhonemizers({ candidates: phonemizerOrder(env), cache: hubCache(env), env });
}

export const kokoro = {
  name: 'kokoro',
  capability: 'speech',
  async detect(files) {
    if (!(await files.exists('config.json'))) return null;
    try {
      if ((await files.json('config.json')).model_type !== 'style_text_to_speech_2') return null;
    } catch {
      return null;
    }
    return {
      variants: Object.keys(VARIANTS),
      defaultVariant: 'fp32',
      inputModalities: ['text'],
      outputModalities: ['audio'],
      contextLength: MAX_TOKENS,
    };
  },
  async missing(files, _info, variant) {
    const out = [];
    for (const rel of needed(variant)) if (!(await files.exists(rel))) out.push(rel);
    return out;
  },
  include: (_info, variant) => ['config.json', 'tokenizer.json', VARIANTS[variant], 'voices/*'],
  async load({ backend, files, variant, overrides = {}, phonemizers = environmentPhonemizers }) {
    const { ort } = backend;
    const phonemizer = await phonemizers(backend.env ?? process.env);
    const [model, tokenizer, voices] = await Promise.all([
      files.read(VARIANTS[variant]),
      files.json('tokenizer.json'),
      availableVoices(files),
    ]);
    const session = await ort.InferenceSession.create(model, {
      executionProviders: backend.executionProviders,
      ...backend.sessionOptions,
      ...overrides,
    });
    const vocab = tokenizer.model.vocab;
    const styles = new Map();

    async function style(voice) {
      if (!styles.has(voice)) {
        const bytes = await files.read(voiceFile(voice));
        const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        styles.set(voice, new Float32Array(copy));
      }
      return styles.get(voice);
    }

    async function phonemes(text, prefix) {
      const english = prefix === 'a' || prefix === 'b';
      const parts = splitPunctuation(english ? normalizeEnglish(text) : text.trim());
      const spoken = parts.filter((p) => !p.punct && /\S/.test(p.text));
      const lines = await phonemizer.phonemize(
        spoken.map((p) => p.text),
        LANGUAGES[prefix].language
      );
      const said = new Map(spoken.map((p, i) => [p, lines[i]]));
      const joined = parts.map((p) => said.get(p) ?? p.text).join('');
      return fixPhonemes(joined, english, prefix === 'a');
    }

    function tokenize(text) {
      const ids = [];
      for (const ch of text) if (Object.hasOwn(vocab, ch)) ids.push(vocab[ch]);
      return ids;
    }

    async function infer(ids, voiceStyle, speed) {
      const input = new BigInt64Array(ids.length + 2);
      ids.forEach((id, i) => {
        input[i + 1] = BigInt(id);
      });
      const row = Math.min(ids.length, MAX_TOKENS - 1, voiceStyle.length / STYLE - 1);
      const feeds = {
        input_ids: new ort.Tensor('int64', input, [1, input.length]),
        style: new ort.Tensor('float32', voiceStyle.slice(row * STYLE, (row + 1) * STYLE), [
          1,
          STYLE,
        ]),
        speed: new ort.Tensor('float32', new Float32Array([speed]), [1]),
      };
      const outputs = await session.run(feeds);
      const waveform = outputs.waveform;
      const samples = new Float32Array(await waveform.getData());
      for (const tensor of Object.values(outputs)) tensor.dispose?.();
      return samples;
    }

    function voiceStyle(voice) {
      if (!voices.includes(voice))
        throw badRequest(
          `voice ${voice} is not available; use one of ${voices.join(', ')}`,
          'voice'
        );
      return style(voice);
    }

    async function tokens(text, { voice = DEFAULT_VOICE } = {}) {
      await voiceStyle(voice);
      return chunkTokens(tokenize(await phonemes(text, voice[0])), vocab[SPACE]);
    }

    async function synthesize(ids, { voice = DEFAULT_VOICE, speed = 1 } = {}) {
      return infer(ids, await voiceStyle(voice), speed);
    }

    return {
      voices,
      phonemizers: phonemizer.names,
      sampleRate: SAMPLE_RATE,
      phonemes,
      tokens,
      synthesize,
      async *speak(text, { signal, ...options } = {}) {
        for (const ids of await tokens(text, options)) {
          signal?.throwIfAborted();
          yield await synthesize(ids, options);
        }
      },
      release: () => session.release(),
    };
  },
};

kokoro.checkOverrides = () => ({});

const SMOKE =
  'Kokoro speaks from the Hugging Face cache. This sentence checks that the voice sounds right.';

const BANDS = 40;
const RANGE = 2;
const LEVELS = 254;
const DB = 40 * RANGE;

export function spectrum(samples, rate = SAMPLE_RATE) {
  const hop = rate / 50;
  const frames = Math.max(1, Math.ceil(samples.length / hop));
  const mel = createMel({
    feature_size: BANDS,
    sampling_rate: rate,
    hop_length: hop,
    n_fft: hop * 2,
    nb_max_frames: frames,
  });
  const values = mel.compute(samples);
  const floor = values.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY) - RANGE;
  const levels = Uint8Array.from(values, (v) => Math.round(((v - floor) / RANGE) * LEVELS));
  return { frames, bands: BANDS, mel: Buffer.from(levels).toString('base64') };
}

export function spectralDistance(got, ref) {
  const a = Buffer.from(got.mel, 'base64');
  const b = Buffer.from(ref.mel, 'base64');
  const frames = Math.min(got.frames, ref.frames);
  let sum = 0;
  for (let m = 0; m < ref.bands; m++)
    for (let t = 0; t < frames; t++) sum += Math.abs(a[m * got.frames + t] - b[m * ref.frames + t]);
  return {
    db: (sum / (frames * ref.bands) / LEVELS) * DB,
    length: got.frames / ref.frames,
  };
}

async function phonemesOf(instance, f) {
  try {
    return (await instance.phonemes(f.text, f.voice[0])) === f.phonemes;
  } catch {
    return null;
  }
}

async function compareSpeech(instance, fixtures, now) {
  const results = [];
  for (const f of fixtures) {
    const t = now();
    const audio = await instance.synthesize(f.ids, { voice: f.voice, speed: f.speed ?? 1 });
    const ms = now() - t;
    results.push({
      name: f.name,
      ...spectralDistance(spectrum(audio), f),
      phonemes: await phonemesOf(instance, f),
      rtf: ms / 1000 / (audio.length / SAMPLE_RATE),
    });
  }
  const worst = results.reduce((a, b) => (b.db > a.db ? b : a), results[0]);
  const mean = results.reduce((s, r) => s + r.db, 0) / Math.max(1, results.length);
  const same = results.filter((r) => r.phonemes).length;
  return {
    results,
    summary: `fixtures: ${results.length} sentences, mean mel distance ${mean.toFixed(2)} dB, worst ${worst?.db.toFixed(2) ?? '0'} dB (${worst?.name ?? '-'}), phonemes match on ${same}`,
    ok: results.every((r) => r.db <= 2 && Math.abs(r.length - 1) <= 0.05),
  };
}

kokoro.check = async (instance, { fixtures, now = () => performance.now() } = {}) => {
  const started = now();
  let samples = 0;
  let peak = 0;
  for await (const audio of instance.speak(SMOKE)) {
    samples += audio.length;
    peak = audio.reduce((m, s) => Math.max(m, Math.abs(s)), peak);
  }
  const requestMs = now() - started;
  const audioSeconds = samples / SAMPLE_RATE;
  const smoke = {
    ok: audioSeconds > 2 && peak > 0.01,
    requestMs,
    audioSeconds,
    realTimeFactor: requestMs / 1000 / audioSeconds,
    peak,
  };
  if (!fixtures) return smoke;
  const compared = await compareSpeech(instance, fixtures, now);
  return { ...smoke, ...compared, ok: smoke.ok && compared.ok };
};

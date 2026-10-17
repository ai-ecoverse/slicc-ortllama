import { posix } from 'node:path';
import { toMono16k } from '../audio/pcm.js';
import { badRequest } from '../http/errors.js';
import { createWhisper, LanguageError } from '../whisper/model.js';
import { createTokenizer } from '../whisper/tokenizer.js';

export const WHISPER_VARIANTS = {
  fp32: '',
  fp16: '_fp16',
  q8: '_quantized',
  int8: '_int8',
  uint8: '_uint8',
  q4: '_q4',
  q4f16: '_q4f16',
  bnb4: '_bnb4',
};

const CONFIGS = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'tokenizer.json',
];

const encoderFile = (variant) => `onnx/encoder_model${WHISPER_VARIANTS[variant]}.onnx`;
const decoderFile = (variant) => `onnx/decoder_model_merged${WHISPER_VARIANTS[variant]}.onnx`;

async function configOf(files) {
  if (!(await files.exists('config.json'))) return null;
  try {
    const config = await files.json('config.json');
    return config?.model_type === 'whisper' ? config : null;
  } catch {
    return null;
  }
}

async function externalData(files, rel) {
  const out = [];
  for (const name of [`${rel}_data`, `${rel}.data`]) {
    if (await files.exists(name)) out.push(name);
    for (let i = 1; await files.exists(`${name}_${i}`); i++) out.push(`${name}_${i}`);
  }
  return out;
}

async function createSession(backend, files, rel) {
  const data = [];
  for (const name of await externalData(files, rel))
    data.push({ path: posix.basename(name), data: await files.read(name) });
  return backend.ort.InferenceSession.create(await files.read(rel), {
    executionProviders: backend.executionProviders,
    ...backend.sessionOptions,
    ...(data.length > 0 ? { externalData: data } : {}),
  });
}

const SMALL_LAYERS = 6;

export const whisper = {
  name: 'whisper',
  capability: 'transcription',
  providers: (info) =>
    (info?.config?.encoder_layers ?? 0) <= SMALL_LAYERS ? { node: ['cpu'] } : {},
  async detect(files) {
    const config = await configOf(files);
    if (!config) return null;
    const variants = [];
    for (const variant of Object.keys(WHISPER_VARIANTS)) {
      if ((await files.exists(encoderFile(variant))) && (await files.exists(decoderFile(variant))))
        variants.push(variant);
    }
    if (variants.length === 0) return null;
    return {
      variants,
      defaultVariant: variants[0],
      inputModalities: ['audio'],
      outputModalities: ['text'],
      contextLength: config.max_target_positions ?? null,
      config,
    };
  },
  async missing(files, _info, variant) {
    const out = [];
    for (const rel of [...CONFIGS, encoderFile(variant), decoderFile(variant)])
      if (!(await files.exists(rel))) out.push(rel);
    return out;
  },
  include(_info, variant) {
    return ['*.json', `${encoderFile(variant)}*`, `${decoderFile(variant)}*`];
  },
  async load({ backend, files, variant, info, create = createSession }) {
    const [generation, preprocessor, tokenizer] = await Promise.all([
      files.json('generation_config.json'),
      files.json('preprocessor_config.json'),
      files.json('tokenizer.json'),
    ]);
    const encoder = await create(backend, files, encoderFile(variant));
    let decoder;
    try {
      decoder = await create(backend, files, decoderFile(variant));
    } catch (error) {
      await encoder.release?.();
      throw error;
    }
    const model = createWhisper({
      ort: backend.ort,
      encoder,
      decoder,
      tokenizer: createTokenizer(tokenizer),
      config: info.config,
      generation,
      preprocessor,
    });
    return {
      ...model,
      async transcribe(audio, options) {
        try {
          return await model.transcribe(audio, options);
        } catch (error) {
          if (error instanceof LanguageError) throw badRequest(error.message, 'language');
          throw error;
        }
      },
    };
  },
};

const SMOKE_SECONDS = 5;

function smokeAudio() {
  const samples = new Float32Array(SMOKE_SECONDS * 16000);
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.1 * Math.sin((2 * Math.PI * 440 * i) / 16000);
  return samples;
}

export function wordErrorRate(reference, hypothesis) {
  const words = (s) =>
    s
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s']/gu, ' ')
      .split(/\s+/)
      .filter(Boolean);
  const a = words(reference);
  const b = words(hypothesis);
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++)
      next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    row = next;
  }
  return a.length === 0 ? (b.length === 0 ? 0 : 1) : row[b.length] / a.length;
}

whisper.checkOverrides = () => ({});

whisper.check = async (instance, { fixtures, now = () => performance.now() } = {}) => {
  const started = now();
  const answer = await instance.transcribe(smokeAudio());
  const requestMs = now() - started;
  const rtf = requestMs / 1000 / SMOKE_SECONDS;
  if (!fixtures) return { ok: true, requestMs, rtf, answer };
  const results = [];
  for (const f of fixtures) {
    const audio = toMono16k(new Uint8Array(Buffer.from(f.audio, 'base64')), f.type ?? 'audio/wav');
    const t = now();
    const got = await instance.transcribe(audio, { language: f.detect ? null : f.language });
    const ms = now() - t;
    const wer = wordErrorRate(f.text, got.text);
    results.push({
      name: f.name,
      wer,
      text: got.text,
      language: got.language,
      rtf: ms / 1000 / (audio.length / 16000),
      ok: wer <= (f.wer ?? 0.1) + 0.1 && (!f.detect || got.language === f.detect),
    });
  }
  const worst = results.reduce((a, b) => (b.wer > a.wer ? b : a), results[0]);
  const mean = results.reduce((s, r) => s + r.wer, 0) / Math.max(1, results.length);
  const wrong = results.filter((r) => !r.ok).map((r) => r.name);
  return {
    requestMs,
    rtf,
    answer,
    results,
    summary: `fixtures: ${results.length} clips, mean WER ${mean.toFixed(3)}, worst ${worst?.wer.toFixed(3) ?? '0'} (${worst?.name ?? '-'})${wrong.length > 0 ? `, failed ${wrong.join(', ')}` : ''}`,
    ok: wrong.length === 0,
  };
};

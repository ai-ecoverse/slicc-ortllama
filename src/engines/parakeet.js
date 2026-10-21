import { toMono16k } from '../audio/pcm.js';
import { badRequest } from '../http/errors.js';
import { createParakeet } from '../parakeet/model.js';
import { LanguageError } from '../whisper/model.js';
import { missingExternal } from './external.js';
import { createSession, WHISPER_VARIANTS, wordErrorRate } from './whisper.js';

const CONFIGS = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'tokenizer.json',
];

const INTEGER = new Set(['q8', 'int8', 'uint8']);

const encoderFile = (variant) => `onnx/encoder_model${WHISPER_VARIANTS[variant]}.onnx`;
const decoderFile = (variant) => `onnx/decoder_joint_model${WHISPER_VARIANTS[variant]}.onnx`;

async function configOf(files) {
  if (!(await files.exists('config.json'))) return null;
  try {
    const config = await files.json('config.json');
    return config?.model_type === 'parakeet_tdt' ? config : null;
  } catch {
    return null;
  }
}

export function openSession(backend, files, rel) {
  if (backend.track !== 'node') return createSession(backend, files, rel);
  return backend.ort.InferenceSession.create(files.path(rel), {
    executionProviders: backend.executionProviders,
    ...backend.sessionOptions,
  });
}

export const parakeet = {
  name: 'parakeet',
  capability: 'transcription',
  providers: (_info, variant) => (INTEGER.has(variant) ? { node: ['cpu'] } : {}),
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
      defaultVariant: variants.includes('q8') ? 'q8' : variants[0],
      inputModalities: ['audio'],
      outputModalities: ['text'],
      contextLength: null,
      config,
    };
  },
  async missing(files, _info, variant) {
    const graphs = [encoderFile(variant), decoderFile(variant)];
    const out = [];
    for (const rel of [...CONFIGS, ...graphs]) if (!(await files.exists(rel))) out.push(rel);
    out.push(...(await missingExternal(files, graphs)));
    return out;
  },
  include(_info, variant) {
    return ['*.json', 'NOTICE', `${encoderFile(variant)}*`, `${decoderFile(variant)}*`];
  },
  async load({ backend, files, variant, info, create = openSession }) {
    const [generation, preprocessor, tokenizer] = await Promise.all([
      files.json('generation_config.json'),
      files.json('preprocessor_config.json'),
      files.json('tokenizer.json'),
    ]);
    const encoder = await create(backend, files, encoderFile(variant));
    let decoder;
    try {
      decoder = await create(
        { ...backend, executionProviders: backend.fallbackProviders },
        files,
        decoderFile(variant)
      );
    } catch (error) {
      await encoder.release?.();
      throw error;
    }
    const model = createParakeet({
      ort: backend.ort,
      encoder,
      decoder,
      tokenizer,
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

parakeet.checkOverrides = () => ({});

const SMOKE_SECONDS = 5;
const TOLERANCE = 0.05;

parakeet.check = async (instance, { fixtures, now = () => performance.now() } = {}) => {
  const smoke = new Float32Array(SMOKE_SECONDS * 16000);
  for (let i = 0; i < smoke.length; i++) smoke[i] = 0.1 * Math.sin((2 * Math.PI * 440 * i) / 16000);
  const started = now();
  const answer = await instance.transcribe(smoke);
  const requestMs = now() - started;
  const rtf = requestMs / 1000 / SMOKE_SECONDS;
  if (!fixtures) return { ok: true, requestMs, rtf, answer };
  const results = [];
  for (const f of fixtures) {
    const audio = toMono16k(new Uint8Array(Buffer.from(f.audio, 'base64')), f.type ?? 'audio/wav');
    const t = now();
    const got = await instance.transcribe(audio);
    const ms = now() - t;
    const drift = wordErrorRate(f.heard, got.text);
    results.push({
      name: f.name,
      wer: wordErrorRate(f.text, got.text),
      drift,
      text: got.text,
      rtf: ms / 1000 / (audio.length / 16000),
      ok: drift <= TOLERANCE,
    });
  }
  const mean = (key) => results.reduce((s, r) => s + r[key], 0) / Math.max(1, results.length);
  const worst = results.reduce((a, b) => (b.drift > a.drift ? b : a), results[0]);
  const wrong = results.filter((r) => !r.ok).map((r) => r.name);
  return {
    requestMs,
    rtf,
    answer,
    results,
    summary: `fixtures: ${results.length} clips, mean WER ${mean('wer').toFixed(3)}, mean change from the reference ${mean('drift').toFixed(3)}, worst ${worst?.drift.toFixed(3) ?? '0'} (${worst?.name ?? '-'})${wrong.length > 0 ? `, failed ${wrong.join(', ')}` : ''}`,
    ok: wrong.length === 0,
  };
};

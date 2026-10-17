import { toMono16k } from '../audio/pcm.js';

export const SILERO_VARIANTS = { fp32: '', fp16: '_fp16', q8: '_quantized' };

const MAX_SIZE = 32 * 1024 * 1024;
const FRAME = 512;
const CONTEXT = 64;
const RATE = 16000;

const modelFile = (variant) => `onnx/model${SILERO_VARIANTS[variant]}.onnx`;

async function isSilero(files, rel) {
  if (!(await files.exists(rel)) || (await files.size(rel)) > MAX_SIZE) return false;
  const bytes = await files.read(rel);
  const tail = new TextDecoder('latin1').decode(bytes.subarray(Math.max(0, bytes.length - 4096)));
  return tail.includes('stateN') && tail.includes('sr');
}

export function createSileroVad(ort, session) {
  const sr = new ort.Tensor('int64', BigInt64Array.from([BigInt(RATE)]), []);
  let queue = Promise.resolve();
  const run = (feeds) => {
    const result = queue.then(() => session.run(feeds));
    queue = result.catch(() => {});
    return result;
  };
  return {
    frame: FRAME,
    rate: RATE,
    stream() {
      let state = new Float32Array(2 * 128);
      let context = new Float32Array(CONTEXT);
      const input = new Float32Array(CONTEXT + FRAME);
      return {
        async probability(frame) {
          input.set(context);
          input.set(frame, CONTEXT);
          const out = await run({
            input: new ort.Tensor('float32', input.slice(), [1, CONTEXT + FRAME]),
            state: new ort.Tensor('float32', state, [2, 1, 128]),
            sr,
          });
          state = Float32Array.from(out.stateN.data);
          context = frame.slice(FRAME - CONTEXT);
          return out.output.data[0];
        },
      };
    },
    release: () => session.release?.(),
  };
}

export const silero = {
  name: 'silero',
  capability: 'vad',
  providers: { node: ['cpu'], web: ['wasm'] },
  async detect(files) {
    if (await files.exists('config.json')) return null;
    if (!(await isSilero(files, modelFile('fp32')))) return null;
    const variants = [];
    for (const variant of Object.keys(SILERO_VARIANTS))
      if (await files.exists(modelFile(variant))) variants.push(variant);
    return {
      variants,
      defaultVariant: 'fp32',
      inputModalities: ['audio'],
      outputModalities: ['speech'],
      contextLength: null,
    };
  },
  async missing(files, _info, variant) {
    return (await files.exists(modelFile(variant))) ? [] : [modelFile(variant)];
  },
  include(_info, variant) {
    return [modelFile(variant)];
  },
  async load({ backend, files, variant }) {
    const session = await backend.ort.InferenceSession.create(
      await files.read(modelFile(variant)),
      {
        executionProviders: backend.executionProviders,
        ...backend.sessionOptions,
      }
    );
    return createSileroVad(backend.ort, session);
  },
};

silero.checkOverrides = () => ({});

export async function speechProbabilities(instance, samples) {
  const stream = instance.stream();
  const out = [];
  for (let at = 0; at + FRAME <= samples.length; at += FRAME)
    out.push(await stream.probability(samples.subarray(at, at + FRAME)));
  return out;
}

const UNSURE = 0.25;

export function compareSpeech(results, fixtures) {
  let max = 0;
  let sum = 0;
  let frames = 0;
  let flips = 0;
  results.forEach((got, i) => {
    const ref = fixtures[i].probs;
    if (got.length !== ref.length) flips += Math.abs(got.length - ref.length);
    for (let j = 0; j < Math.min(got.length, ref.length); j++) {
      const d = Math.abs(got[j] - ref[j]);
      frames++;
      sum += d;
      max = Math.max(max, d);
      if (got[j] >= 0.5 !== ref[j] >= 0.5 && Math.abs(ref[j] - 0.5) > UNSURE) flips++;
    }
  });
  const mean = frames ? sum / frames : 0;
  return {
    frames,
    mean,
    max,
    flips,
    ok: mean <= 0.01 && max <= 0.1 && flips === 0,
  };
}

silero.check = async (instance, { fixtures, now = () => performance.now() } = {}) => {
  const frame = new Float32Array(FRAME);
  const stream = instance.stream();
  const started = now();
  let probability = 0;
  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < FRAME; j++)
      frame[j] = 0.3 * Math.sin((2 * Math.PI * 220 * (i * FRAME + j)) / RATE);
    probability = await stream.probability(frame);
  }
  const requestMs = now() - started;
  if (!fixtures) return { ok: true, requestMs, answer: { probability } };
  const results = [];
  for (const f of fixtures) {
    const audio = toMono16k(new Uint8Array(Buffer.from(f.audio, 'base64')), f.type ?? 'audio/wav');
    results.push(await speechProbabilities(instance, audio));
  }
  const compared = compareSpeech(results, fixtures);
  return {
    requestMs,
    answer: { probability },
    ...compared,
    summary: `fixtures: ${compared.frames} frames, mean |dp| ${compared.mean.toFixed(4)}, max ${compared.max.toFixed(4)}, ${compared.flips} changed`,
  };
};

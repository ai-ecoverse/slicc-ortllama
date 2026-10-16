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
          const out = await session.run({
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

silero.check = async (instance, { now = () => performance.now() } = {}) => {
  const stream = instance.stream();
  const frame = new Float32Array(FRAME);
  const started = now();
  let probability = 0;
  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < FRAME; j++)
      frame[j] = 0.3 * Math.sin((2 * Math.PI * 220 * (i * FRAME + j)) / RATE);
    probability = await stream.probability(frame);
  }
  return { ok: true, requestMs: now() - started, answer: { probability } };
};

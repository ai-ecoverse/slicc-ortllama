import { providersFrom } from './providers.js';

export function defaultProviders(platform = process.platform, arch = process.arch) {
  return platform === 'linux' && arch === 'arm64' ? ['cpu'] : ['webgpu', 'cpu'];
}

function binaryHalf(tensor) {
  const data = tensor.location === 'cpu' && tensor.type === 'float16' ? tensor.data : null;
  if (!data || data instanceof Uint16Array) return tensor;
  const view = new Uint16Array(data.buffer, data.byteOffset, data.length);
  return Object.defineProperty(Object.create(tensor), 'data', { value: view });
}

export function withHalfInputs(ort) {
  const wrap = (session) =>
    Object.assign(Object.create(session), {
      run(feeds, ...rest) {
        const fixed = Object.fromEntries(Object.entries(feeds).map(([k, t]) => [k, binaryHalf(t)]));
        return session.run(fixed, ...rest);
      },
    });
  return {
    ...ort,
    InferenceSession: {
      create: async (...args) => wrap(await ort.InferenceSession.create(...args)),
    },
  };
}

export async function loadBackend({
  env = process.env,
  importOrt = () => import('onnxruntime-node'),
  platform,
  arch,
} = {}) {
  let ort;
  try {
    ort = await importOrt();
  } catch (error) {
    throw new Error('ortllama on Node needs onnxruntime-node: npm install onnxruntime-node', {
      cause: error,
    });
  }
  ort.env.logLevel = env.ORTLLAMA_ORT_LOG_LEVEL ?? 'error';
  return {
    ort: withHalfInputs(ort),
    track: 'node',
    env,
    executionProviders: providersFrom(env, defaultProviders(platform, arch)),
    fallbackProviders: ['cpu'],
    sessionOptions: { preferredOutputLocation: 'cpu' },
  };
}

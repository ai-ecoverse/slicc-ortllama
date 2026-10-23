import { providersFrom } from './providers.js';
import { stageFiles } from './stage.js';
import { loadBackend as loadWebBackend } from './web.js';

const DEFAULTS = {
  'darwin/arm64': ['webgpu', 'cpu'],
  'linux/x64': ['cuda', 'webgpu', 'cpu'],
  'win32/x64': ['webgpu', 'dml', 'cpu'],
  'win32/arm64': ['webgpu', 'dml', 'cpu'],
};

export function defaultProviders(platform = process.platform, arch = process.arch) {
  return DEFAULTS[`${platform}/${arch}`] ?? ['cpu'];
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

let noticed = false;

async function wasmFallback({ env, error, loadWeb = loadWebBackend, notice }) {
  if (!noticed) {
    noticed = true;
    const reason = String(error.message).split('\n')[0];
    notice(
      `ortllama: onnxruntime-node is not available (${reason}); running on onnxruntime-web WASM, CPU only. npm install onnxruntime-node for native and GPU execution providers`
    );
  }
  const web = await loadWeb({ env, gpu: null });
  return { ...web, availableProviders: ['wasm', 'cpu'] };
}

export async function loadBackend({
  env = process.env,
  hostEnv = process.env,
  importOrt = () => import('onnxruntime-node'),
  loadWeb,
  notice = (message) => process.stderr.write(`${message}\n`),
  platform,
  arch,
} = {}) {
  hostEnv.ORT_DISABLE_TELEMETRY ??= env.ORT_DISABLE_TELEMETRY ?? '1';
  let ort;
  try {
    ort = await importOrt();
  } catch (error) {
    return wasmFallback({ env, error, loadWeb, notice });
  }
  ort.env.logLevel = env.ORTLLAMA_ORT_LOG_LEVEL ?? 'error';
  return {
    ort: withHalfInputs(ort),
    track: 'node',
    env,
    executionProviders: defaultProviders(platform, arch),
    overrideProviders: providersFrom(env),
    fallbackProviders: ['cpu'],
    availableProviders: ort.listSupportedBackends?.().map((b) => b.name) ?? null,
    sessionOptions: { preferredOutputLocation: 'cpu' },
    stage: stageFiles,
  };
}

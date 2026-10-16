import { providersFrom } from './providers.js';

export function defaultProviders(platform = process.platform, arch = process.arch) {
  return platform === 'linux' && arch === 'arm64' ? ['cpu'] : ['webgpu', 'cpu'];
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
    ort,
    track: 'node',
    executionProviders: providersFrom(env, defaultProviders(platform, arch)),
    fallbackProviders: ['cpu'],
    sessionOptions: { preferredOutputLocation: 'cpu' },
  };
}

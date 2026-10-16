import { readFile } from 'node:fs/promises';
import { providersFrom } from './providers.js';

const WASM = 'ort-wasm-simd-threaded.asyncify.wasm';

export async function loadBackend({
  env = process.env,
  importOrt = () => import('onnxruntime-web/webgpu'),
  resolve = (specifier) => import.meta.resolve(specifier),
  readBinary = readFile,
  gpu = globalThis.navigator?.gpu,
} = {}) {
  const ort = await importOrt();
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmBinary = await readBinary(new URL(WASM, resolve('onnxruntime-web/webgpu')));
  ort.env.logLevel = env.ORTLLAMA_ORT_LOG_LEVEL ?? 'error';
  return {
    ort,
    track: 'web',
    executionProviders: providersFrom(env, gpu ? ['webgpu', 'wasm'] : ['wasm']),
    fallbackProviders: ['wasm'],
    sessionOptions: {},
  };
}

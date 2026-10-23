import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HttpError } from '../../http/errors.js';
import { optional, optionalModule } from '../optional.js';
import {
  inThreadWorker,
  rangeBlob,
  releaseDevices,
  trackGpu,
  withWorker,
  workerScope,
} from './shim.js';

export const WLLAMA = '@wllama/wllama';
export const WLLAMA_ENTRY = 'esm/index.js';
export const DAWN = 'webgpu';
export const ALL_LAYERS = 99999;

const GPU = new Set(['webgpu']);
const CPU = new Set(['cpu', 'wasm']);

let dawn = null;

export function wllamaLibrary() {
  return optionalModule(WLLAMA, { entry: WLLAMA_ENTRY });
}

export function hasJspi(wasm = globalThis.WebAssembly) {
  return typeof wasm?.Suspending === 'function' && typeof wasm?.promising === 'function';
}

export function deviceFor(executionProviders) {
  const [first] = executionProviders;
  if (GPU.has(first)) return 'webgpu';
  if (CPU.has(first)) return 'cpu';
  throw new Error(`GGUF models run on webgpu, cpu or wasm, not ${first}`);
}

export async function gpuFor(
  track,
  { navigator = globalThis.navigator, library = () => optional(DAWN) } = {}
) {
  if (track !== 'node') {
    if (!navigator?.gpu) throw new Error('WebGPU is not available here');
    return { gpu: navigator.gpu, globals: {} };
  }
  let module;
  try {
    module = await library();
  } catch (error) {
    throw new Error(
      `WebGPU for GGUF models on Node needs the npm package ${DAWN}: ${error.message}`
    );
  }
  dawn ??= module.create([]);
  return { gpu: dawn, globals: module.globals ?? {} };
}

function quietLogger(errors) {
  const keep = (...args) => {
    errors.push(args.map(String).join(' '));
    if (errors.length > 20) errors.shift();
  };
  return { debug() {}, log() {}, warn: keep, error: keep };
}

export function loadedOk(context) {
  return Number(context?.n_vocab) > 0 && Number(context?.n_layer) > 0;
}

export async function openWllama({
  backend,
  files,
  rels,
  contextLength,
  options = {},
  library = wllamaLibrary,
  gpus = gpuFor,
  wasm = globalThis.WebAssembly,
}) {
  if (!hasJspi(wasm))
    throw new HttpError(
      501,
      'GGUF models need WebAssembly JSPI: Node 25 or later, or Node 24 with --experimental-wasm-jspi',
      { code: 'engine_unavailable' }
    );
  const device = deviceFor(backend.executionProviders);
  const { module, url } = await library();
  const gpu = device === 'webgpu' ? await gpus(backend.track) : { gpu: null, globals: {} };
  const tracked = trackGpu(gpu.gpu);
  const dir = `${dirname(fileURLToPath(url))}/wasm`;
  const wasmPath = `${dir}/wllama.wasm`;
  const scope = workerScope({
    dir,
    navigator: { gpu: tracked.gpu, hardwareConcurrency: 1, userAgent: '' },
    gpuGlobals: gpu.globals,
  });
  const Worker = inThreadWorker(scope);
  class InThreadWllama extends module.Wllama {
    getWorkerResources() {
      const resources = { wasmPath, compat: false };
      Object.defineProperty(resources, 'noWebGPU', { get: () => false, set() {} });
      return resources;
    }
  }
  const errors = [];
  const wllama = new InThreadWllama(
    { default: wasmPath },
    { cacheManager: {}, modelManager: {}, logger: quietLogger(errors) }
  );
  const handles = [];
  const release = async () => {
    await wllama.exit().catch(() => {});
    await releaseDevices(tracked.devices);
    for (const handle of handles.splice(0)) await handle.close().catch(() => {});
  };
  const blobs = [];
  try {
    for (const rel of rels) {
      const handle = await files.open(rel);
      handles.push(handle);
      blobs.push(rangeBlob(handle, (await handle.stat()).size));
    }
    await withWorker(Worker, () =>
      wllama.loadModel(blobs, {
        n_ctx: contextLength,
        n_threads: 1,
        n_gpu_layers: device === 'webgpu' ? ALL_LAYERS : 0,
        jinja: true,
        ...options,
      })
    ).finally(() => {
      for (const blob of blobs) blob.clear();
    });
    const context = wllama.getLoadedContextInfo();
    if (!loadedOk(context))
      throw new HttpError(
        422,
        `wllama ${module.Wllama.getLibllamaVersion()} could not load this GGUF model${errors.length ? `: ${errors.at(-1)}` : ''}`,
        { code: 'model_unsupported', param: 'model' }
      );
    if (device === 'webgpu' && tracked.devices.size === 0)
      throw new Error('no WebGPU adapter was found for GGUF');
    return { wllama, context, device, release };
  } catch (error) {
    await release();
    if (error?.status) throw error;
    const detail = errors.length > 0 ? ` (${errors.at(-1)})` : '';
    throw new Error(`${error?.message ?? error}${detail}`, { cause: error });
  }
}

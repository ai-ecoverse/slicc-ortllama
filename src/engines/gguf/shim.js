import * as fs from 'node:fs';
import * as os from 'node:os';
import * as util from 'node:util';

const later = (fn) => (globalThis.setImmediate ?? setTimeout)(fn);

const NO_GPU = { requestAdapter: async () => null };

export function trackGpu(gpu) {
  const devices = new Set();
  if (!gpu) return { gpu: NO_GPU, devices };
  const adopt = (adapter) => {
    if (!adapter) return adapter;
    const request = adapter.requestDevice;
    adapter.requestDevice = async (descriptor) => {
      const device = await request.call(adapter, descriptor);
      devices.add(device);
      return device;
    };
    return adapter;
  };
  return {
    devices,
    gpu: {
      requestAdapter: async (options) => adopt(await gpu.requestAdapter(options)),
      wgslLanguageFeatures: gpu.wgslLanguageFeatures ?? new Set(),
    },
  };
}

export async function releaseDevices(devices, waitMs = 1000) {
  for (const device of devices) {
    const lost = device.lost?.then(() => {
      device.onuncapturederror = null;
    });
    device.destroy();
    let timer;
    await Promise.race([lost, new Promise((resolve) => (timer = setTimeout(resolve, waitMs)))]);
    clearTimeout(timer);
    device.onuncapturederror = null;
  }
  devices.clear();
}

export async function codeOf(url, { resolveUrl, fetcher = globalThis.fetch } = {}) {
  const resolve = resolveUrl ?? (await import('node:buffer')).resolveObjectURL;
  const blob = resolve?.(url);
  const code = blob ? await blob.text() : await (await fetcher(url)).text();
  URL.revokeObjectURL?.(url);
  return code;
}

const PROCESS_KEYS = ['versions', 'platform', 'arch'];

export function workerScope({ dir, navigator, gpuGlobals = {}, modules = {} }) {
  const shadowProcess = {
    argv: [],
    env: {},
    exitCode: undefined,
    on() {},
    exit() {},
    stdin: { fd: 0 },
    ...Object.fromEntries(PROCESS_KEYS.map((k) => [k, globalThis.process?.[k]])),
  };
  const threads = { isMainThread: true, Worker: undefined, workerData: undefined };
  const found = { worker_threads: threads, fs, util, os, ...modules };
  const require = (name) => {
    const module = found[name.replace(/^node:/, '')];
    if (!module) throw new Error(`the in-thread wllama worker has no module ${name}`);
    return module;
  };
  return {
    self: { name: '', location: { href: `file://${dir}/wllama.js` } },
    require,
    __dirname: dir,
    __filename: `${dir}/wllama.js`,
    global: {},
    process: shadowProcess,
    navigator,
    Atomics: Object.create(Atomics, { waitAsync: { value: undefined } }),
    module: undefined,
    ...gpuGlobals,
  };
}

export function evaluate(code, scope, postMessage) {
  const names = Object.keys(scope);
  const run = new Function(
    ...names,
    'postMessage',
    `let onmessage = null;\n${code}\n;return (event) => onmessage?.(event);`
  );
  return run(...names.map((n) => scope[n]), postMessage);
}

export function inThreadWorker(scope, { read = codeOf, onCreate = () => {} } = {}) {
  return class InThreadWorker {
    constructor(url) {
      this.onmessage = null;
      this.onerror = null;
      this.alive = true;
      this.deliver = null;
      const post = (data) =>
        later(() => {
          if (this.alive) this.onmessage?.({ data });
        });
      this.ready = read(url).then(
        (code) => {
          if (this.alive) this.deliver = evaluate(code, scope, post);
        },
        (error) => this.onerror?.(error)
      );
      onCreate(this);
    }

    postMessage(data) {
      this.ready.then(() =>
        later(() => {
          if (this.alive) this.deliver?.({ data });
        })
      );
    }

    terminate() {
      this.alive = false;
      this.deliver = null;
      this.onmessage = null;
    }
  };
}

export async function withWorker(Worker, fn, target = globalThis) {
  const saved = Object.getOwnPropertyDescriptor(target, 'Worker');
  Object.defineProperty(target, 'Worker', {
    value: Worker,
    configurable: true,
    writable: true,
  });
  try {
    return await fn();
  } finally {
    if (saved) Object.defineProperty(target, 'Worker', saved);
    else delete target.Worker;
  }
}

export function rangeBlob(handle, size, start = 0) {
  return {
    size,
    slice(from = 0, to = size) {
      const a = Math.max(0, Math.min(size, from));
      const b = Math.max(a, Math.min(size, to));
      return rangeBlob(handle, b - a, start + a);
    },
    async arrayBuffer() {
      const out = new Uint8Array(size);
      let at = 0;
      while (at < size) {
        const { bytesRead } = await handle.read(out, at, size - at, start + at);
        if (!bytesRead) break;
        at += bytesRead;
      }
      return at === size ? out.buffer : out.slice(0, at).buffer;
    },
  };
}

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as util from 'node:util';

const later = (fn) => (globalThis.setImmediate ?? setTimeout)(fn);

const NO_GPU = { requestAdapter: async () => null };
const created = new Map();

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
  const blob = created.get(url) ?? resolve?.(url);
  created.delete(url);
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
  const make = URL.createObjectURL;
  const urls = [];
  URL.createObjectURL = (blob) => {
    const url = make.call(URL, blob);
    created.set(url, blob);
    urls.push(url);
    return url;
  };
  try {
    return await fn();
  } finally {
    URL.createObjectURL = make;
    for (const url of urls) created.delete(url);
    if (saved) Object.defineProperty(target, 'Worker', saved);
    else delete target.Worker;
  }
}

export const CHUNK = 16 * 2 ** 20;
export const DIRECT = 4 * 2 ** 20;
export const KEEP = 4;
export const DENSE = 1 / 2;
export const NEAR = 2 ** 16;

async function readAt(handle, out, position) {
  let at = 0;
  while (at < out.length) {
    const { bytesRead } = await handle.read(out, at, out.length - at, position + at);
    if (!bytesRead) break;
    at += bytesRead;
  }
  return at === out.length ? out.buffer : out.slice(0, at).buffer;
}

export function chunkedReader(
  handle,
  size,
  { chunk = CHUNK, direct = DIRECT, keep = KEEP, dense = DENSE, near = NEAR } = {}
) {
  const chunks = new Map();
  const small = new Map();
  let end = -1;
  const get = (index) => {
    let entry = chunks.get(index);
    chunks.delete(index);
    if (!entry) {
      const from = index * chunk;
      entry = readAt(handle, new Uint8Array(Math.min(chunk, size - from)), from);
      entry.catch(() => chunks.get(index) === entry && chunks.delete(index));
    }
    chunks.set(index, entry);
    if (chunks.size > keep) chunks.delete(chunks.keys().next().value);
    return entry;
  };
  const warm = (from, to) => {
    const first = Math.floor(from / chunk);
    const last = Math.floor((to - 1) / chunk);
    const follows = Math.abs(from - end) <= near;
    end = to;
    if (chunks.has(first) && chunks.has(last)) return true;
    const used = (small.get(first) ?? 0) + to - from;
    small.set(first, used);
    return follows || used > chunk * dense;
  };
  return {
    async read(from, to) {
      if (to - from >= direct || !warm(from, to))
        return readAt(handle, new Uint8Array(to - from), from);
      const out = new Uint8Array(to - from);
      let at = from;
      while (at < to) {
        const index = Math.floor(at / chunk);
        const offset = at - index * chunk;
        const part = new Uint8Array(await get(index)).subarray(offset, offset + to - at);
        if (part.length === 0) break;
        out.set(part, at - from);
        at += part.length;
      }
      return at === to ? out.buffer : out.slice(0, at - from).buffer;
    },
    clear() {
      chunks.clear();
      small.clear();
      end = -1;
    },
  };
}

export function rangeBlob(handle, size, options) {
  const reader = chunkedReader(handle, size, options);
  const view = (start, length) => ({
    size: length,
    slice(from = 0, to = length) {
      const a = Math.max(0, Math.min(length, from));
      const b = Math.max(a, Math.min(length, to));
      return view(start + a, b - a);
    },
    arrayBuffer: () => reader.read(start, start + length),
  });
  return Object.assign(view(0, size), { clear: reader.clear });
}

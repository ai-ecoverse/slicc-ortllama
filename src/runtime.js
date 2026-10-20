import { verifyModel } from './hf/verify.js';
import { providerAttempts, providerOrder } from './ort/providers.js';

export const PRIORITY = { expire: -1, realtime: 0, speech: 1, decisions: 2, chat: 3, admin: 4 };

function abortError(signal) {
  return signal?.reason instanceof Error
    ? signal.reason
    : new DOMException('the request was aborted', 'AbortError');
}

const noop = () => {};

export function processMemory(proc = globalThis.process) {
  const usage = proc?.memoryUsage?.();
  if (!usage?.rss) return null;
  const { rss, heapUsed, external, arrayBuffers } = usage;
  return { rss, heapUsed, external, arrayBuffers };
}

async function bytesOf({ files, engine, info, variant }) {
  if (!files?.bytes || !engine.include) return null;
  return files.bytes(engine.include(info, variant)).catch(() => null);
}

export class Runtime {
  constructor({
    loadBackend,
    keepAliveMs = 300_000,
    maxLoaded = 1,
    now = Date.now,
    timers = globalThis,
    warn = (message) => console.warn(message),
    memory = processMemory,
  }) {
    this.loadBackend = loadBackend;
    this.keepAliveMs = keepAliveMs;
    this.maxLoaded = maxLoaded;
    this.now = now;
    this.timers = timers;
    this.warn = warn;
    this.memory = memory;
    this.warned = new Set();
    this.loaded = new Map();
    this.queue = [];
    this.running = false;
    this.seq = 0;
    this.backendPromise = null;
  }

  backend() {
    this.backendPromise ??= this.loadBackend().catch((error) => {
      this.backendPromise = null;
      throw error;
    });
    return this.backendPromise;
  }

  run(priority, task, signal, key = null, ready = null) {
    if (signal?.aborted) return Promise.reject(abortError(signal));
    return new Promise((resolve, reject) => {
      const job = { priority, seq: this.seq++, task, resolve, reject, signal, key, ready };
      ready?.catch(noop).then(() => {
        job.ready = null;
        this.pump();
      });
      job.onAbort = () => {
        this.queue.splice(this.queue.indexOf(job), 1);
        reject(abortError(signal));
      };
      signal?.addEventListener('abort', job.onAbort, { once: true });
      this.queue.push(job);
      this.pump();
    });
  }

  fits(job) {
    const entries = [...this.loaded.values()];
    if (!job.key || this.loaded.has(job.key) || !entries.some((e) => e.busy > 0)) return true;
    return entries.filter((e) => e.busy > 0 || e.pins > 0).length < this.maxLoaded;
  }

  pump() {
    if (this.running) return;
    this.queue.sort((a, b) => a.priority - b.priority || a.seq - b.seq);
    const waiting = this.queue.findIndex((j) => j.ready);
    const ahead = waiting < 0 ? this.queue : this.queue.slice(0, waiting);
    const at = ahead.findIndex((j) => this.fits(j));
    if (at < 0) return;
    const [job] = this.queue.splice(at, 1);
    job.signal?.removeEventListener('abort', job.onAbort);
    this.running = true;
    const turn = this.turn(job.priority);
    Promise.resolve()
      .then(() => job.task(job.signal, turn))
      .then(job.resolve, job.reject)
      .finally(() => turn.end());
  }

  turn(priority) {
    const turn = {
      end: () => {
        this.running = false;
        this.pump();
      },
      yield: () => {
        if (!this.queue.some((j) => j.priority < priority && this.fits(j)))
          return Promise.resolve();
        const release = turn.end;
        return new Promise((resume) => {
          this.queue.push({
            priority,
            seq: -1,
            task: () =>
              new Promise((done) => {
                turn.end = done;
                resume();
              }),
            resolve: noop,
            reject: noop,
          });
          release();
        });
      },
    };
    return turn;
  }

  use(resolved, priority, fn, signal, ready = null) {
    return this.run(
      priority,
      async (_signal, turn) => {
        const entry = await this.acquire(resolved);
        entry.busy++;
        try {
          if (signal?.aborted) throw abortError(signal);
          return await fn(entry.instance, entry, () => turn.yield());
        } finally {
          entry.busy--;
          this.touch(entry);
        }
      },
      signal,
      resolved.key,
      ready
    );
  }

  hold(resolved, signal) {
    return this.run(
      PRIORITY.realtime,
      async () => {
        const entry = await this.acquire(resolved);
        entry.pins++;
        this.timers.clearTimeout(entry.timer);
        entry.timer = null;
        entry.expiresAt = null;
        return entry;
      },
      signal,
      resolved.key
    );
  }

  unhold(entry) {
    entry.pins = Math.max(0, entry.pins - 1);
    if (entry.pins === 0 && this.loaded.get(entry.key) === entry) this.touch(entry);
  }

  async acquire(resolved) {
    const existing = this.loaded.get(resolved.key);
    if (existing) {
      this.timers.clearTimeout(existing.timer);
      existing.generation++;
      return existing;
    }
    const fixed = [...this.loaded.values()].filter((e) => e.busy > 0 || e.pins > 0).length;
    await this.evict(this.maxLoaded - 1 - fixed);
    const backend = await this.backend();
    const started = this.now();
    await verifyModel(resolved, { read: backend.track === 'node' ? 'stream' : 'whole' });
    const { order, source } = providerOrder(backend, resolved);
    const { attempts, skipped } = providerAttempts(order, backend);
    for (const ep of skipped)
      this.warnOnce(
        `ortllama: onnxruntime-${backend.track} has no ${ep} execution provider here; skipping it`
      );
    let executionProviders;
    let instance;
    for (const [i, eps] of attempts.entries()) {
      try {
        instance = await resolved.engine.load({
          backend: { ...backend, executionProviders: eps },
          files: resolved.files,
          variant: resolved.variant,
          info: resolved.info,
        });
        executionProviders = eps;
        break;
      } catch (error) {
        const next = attempts[i + 1];
        if (error.status || !next) throw error;
        this.warnOnce(
          `ortllama: ${resolved.id} cannot load on ${eps.join(',')}, using ${next.join(',')}: ${error.message}`,
          `${eps.join(',')}\n${error.message}`
        );
      }
    }
    const entry = {
      key: resolved.key,
      id: resolved.id,
      variant: resolved.variant,
      engine: resolved.engine.name,
      executionProviders,
      providerSource: source,
      modelBytes: await bytesOf(resolved),
      instance,
      loadedAt: started,
      loadMs: this.now() - started,
      lastUsed: this.now(),
      timer: null,
      expiresAt: null,
      generation: 0,
      pins: 0,
      busy: 0,
    };
    this.loaded.set(resolved.key, entry);
    return entry;
  }

  warnOnce(message, key = message) {
    if (this.warned.has(key)) return;
    this.warned.add(key);
    this.warn(message);
  }

  touch(entry) {
    entry.lastUsed = this.now();
    this.timers.clearTimeout(entry.timer);
    entry.timer = null;
    entry.expiresAt = null;
    const generation = ++entry.generation;
    if (entry.pins > 0 || !Number.isFinite(this.keepAliveMs)) return;
    entry.expiresAt = entry.lastUsed + this.keepAliveMs;
    entry.timer = this.timers.setTimeout(() => {
      this.run(PRIORITY.expire, () => this.expire(entry, generation)).catch(() => {});
    }, this.keepAliveMs);
    entry.timer?.unref?.();
  }

  async expire(entry, generation) {
    if (
      this.loaded.get(entry.key) !== entry ||
      entry.generation !== generation ||
      entry.pins > 0 ||
      entry.busy > 0
    )
      return false;
    return this.unload(entry.key);
  }

  async evict(keep) {
    const byAge = [...this.loaded.values()]
      .filter((entry) => entry.pins === 0 && entry.busy === 0)
      .sort((a, b) => a.lastUsed - b.lastUsed);
    while (byAge.length > Math.max(0, keep)) await this.unload(byAge.shift().key);
  }

  async unload(key) {
    const entry = this.loaded.get(key);
    if (!entry) return false;
    this.loaded.delete(key);
    this.timers.clearTimeout(entry.timer);
    await entry.instance.release?.();
    return true;
  }

  ps() {
    return [...this.loaded.values()].map(
      ({
        id,
        variant,
        engine,
        executionProviders,
        providerSource,
        modelBytes,
        loadMs,
        lastUsed,
        expiresAt,
        pins,
      }) => ({
        id,
        variant,
        engine,
        executionProviders,
        providerSource,
        modelBytes,
        loadMs,
        lastUsed: new Date(lastUsed).toISOString(),
        expiresAt: expiresAt === null ? null : new Date(expiresAt).toISOString(),
        sessions: pins,
      })
    );
  }

  async close() {
    for (const key of [...this.loaded.keys()]) await this.unload(key);
  }
}

export const PRIORITY = { expire: -1, realtime: 0, speech: 1, decisions: 2, chat: 3, admin: 4 };

function abortError(signal) {
  return signal?.reason instanceof Error
    ? signal.reason
    : new DOMException('the request was aborted', 'AbortError');
}

export class Runtime {
  constructor({
    loadBackend,
    keepAliveMs = 300_000,
    maxLoaded = 1,
    now = Date.now,
    timers = globalThis,
  }) {
    this.loadBackend = loadBackend;
    this.keepAliveMs = keepAliveMs;
    this.maxLoaded = maxLoaded;
    this.now = now;
    this.timers = timers;
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

  run(priority, task, signal) {
    if (signal?.aborted) return Promise.reject(abortError(signal));
    return new Promise((resolve, reject) => {
      const job = { priority, seq: this.seq++, task, resolve, reject, signal };
      job.onAbort = () => {
        this.queue.splice(this.queue.indexOf(job), 1);
        reject(abortError(signal));
      };
      signal?.addEventListener('abort', job.onAbort, { once: true });
      this.queue.push(job);
      this.pump();
    });
  }

  pump() {
    if (this.running || this.queue.length === 0) return;
    this.queue.sort((a, b) => a.priority - b.priority || a.seq - b.seq);
    const job = this.queue.shift();
    job.signal?.removeEventListener('abort', job.onAbort);
    this.running = true;
    Promise.resolve()
      .then(() => job.task(job.signal))
      .then(job.resolve, job.reject)
      .finally(() => {
        this.running = false;
        this.pump();
      });
  }

  use(resolved, priority, fn, signal) {
    return this.run(
      priority,
      async () => {
        const entry = await this.acquire(resolved);
        try {
          if (signal?.aborted) throw abortError(signal);
          return await fn(entry.instance, entry);
        } finally {
          this.touch(entry);
        }
      },
      signal
    );
  }

  async acquire(resolved) {
    const existing = this.loaded.get(resolved.key);
    if (existing) {
      this.timers.clearTimeout(existing.timer);
      existing.generation++;
      return existing;
    }
    await this.evict(this.maxLoaded - 1);
    const backend = await this.backend();
    const started = this.now();
    let executionProviders = backend.executionProviders;
    let instance;
    try {
      instance = await resolved.engine.load({
        backend,
        files: resolved.files,
        variant: resolved.variant,
        info: resolved.info,
      });
    } catch (error) {
      const fallback = backend.fallbackProviders;
      if (!fallback || fallback.join() === executionProviders.join()) throw error;
      executionProviders = fallback;
      instance = await resolved.engine.load({
        backend: { ...backend, executionProviders },
        files: resolved.files,
        variant: resolved.variant,
        info: resolved.info,
      });
    }
    const entry = {
      key: resolved.key,
      id: resolved.id,
      variant: resolved.variant,
      engine: resolved.engine.name,
      executionProviders,
      instance,
      loadedAt: started,
      loadMs: this.now() - started,
      lastUsed: this.now(),
      timer: null,
      expiresAt: null,
      generation: 0,
    };
    this.loaded.set(resolved.key, entry);
    return entry;
  }

  touch(entry) {
    entry.lastUsed = this.now();
    this.timers.clearTimeout(entry.timer);
    entry.timer = null;
    entry.expiresAt = null;
    const generation = ++entry.generation;
    if (!Number.isFinite(this.keepAliveMs)) return;
    entry.expiresAt = entry.lastUsed + this.keepAliveMs;
    entry.timer = this.timers.setTimeout(() => {
      this.run(PRIORITY.expire, () => this.expire(entry, generation)).catch(() => {});
    }, this.keepAliveMs);
    entry.timer?.unref?.();
  }

  async expire(entry, generation) {
    if (this.loaded.get(entry.key) !== entry || entry.generation !== generation) return false;
    return this.unload(entry.key);
  }

  async evict(keep) {
    const byAge = [...this.loaded.values()].sort((a, b) => a.lastUsed - b.lastUsed);
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
      ({ id, variant, engine, executionProviders, loadMs, lastUsed, expiresAt }) => ({
        id,
        variant,
        engine,
        executionProviders,
        loadMs,
        lastUsed: new Date(lastUsed).toISOString(),
        expiresAt: expiresAt === null ? null : new Date(expiresAt).toISOString(),
      })
    );
  }

  async close() {
    for (const key of [...this.loaded.keys()]) await this.unload(key);
  }
}

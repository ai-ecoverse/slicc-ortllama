import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { openWllama } from '../../src/engines/gguf/load.js';

const MIB = 2 ** 20;
const byteAt = (position) => position % 251;
const PATTERN = Uint8Array.from({ length: 251 * 4096 }, (_, i) => i % 251);

function fill(buffer, offset, n, position) {
  for (let done = 0; done < n; ) {
    const from = (position + done) % 251;
    const part = PATTERN.subarray(from, from + Math.min(n - done, PATTERN.length - 251));
    buffer.set(part, offset + done);
    done += part.length;
  }
}

function virtualFiles(size) {
  const reads = [];
  const handle = {
    async read(buffer, offset, length, position) {
      const n = Math.max(0, Math.min(length, size - position));
      reads.push({ position, length: n });
      fill(buffer, offset, n, position);
      return { bytesRead: n, buffer };
    },
    async stat() {
      return { size };
    },
    async close() {},
  };
  return { reads, files: { open: async () => handle } };
}

function wllamaLike(access, log) {
  class Wllama {
    static getLibllamaVersion() {
      return 'b1-fake';
    }

    async loadModel([blob]) {
      log.blob = blob;
      log.bad = 0;
      for (const [from, length] of access(blob.size)) {
        const bytes = new Uint8Array(await blob.slice(from, from + length).arrayBuffer());
        log.bytes = (log.bytes ?? 0) + bytes.length;
        for (let i = 0; i < bytes.length; i += 4093) if (bytes[i] !== byteAt(from + i)) log.bad++;
        if (bytes.at(-1) !== byteAt(from + bytes.length - 1)) log.bad++;
      }
    }

    getLoadedContextInfo() {
      return { n_vocab: 3, n_layer: 2 };
    }

    async exit() {}
  }
  const url = pathToFileURL(join(tmpdir(), 'fake-wllama', 'index.js')).href;
  return async () => ({ module: { Wllama }, url });
}

function* tensors(size) {
  yield [0, 131072];
  yield [0, MIB];
  yield [MIB - 1, MIB];
  for (let at = 2 * MIB; at < size; at += MIB - 2304) yield [at, MIB];
  for (let at = 20 * MIB; at < 60 * MIB; at += MIB - 2304) yield [at, MIB];
}

function* far(size) {
  yield [0, 4096];
  for (let at = size - 100 * MIB; at < size; at += MIB - 2304) yield [at, MIB];
}

function* scattered(size) {
  for (let at = 5 * MIB; at < size; at += 70 * MIB) yield [at, MIB];
  for (let at = 3 * MIB; at < size; at += 70 * MIB) yield [at, 2 * MIB];
  for (let at = 9 * MIB; at + 15 * MIB < size; at += 70 * MIB) yield [at, 15 * MIB];
}

async function load(size, access) {
  const { reads, files } = virtualFiles(size);
  const log = {};
  const { release } = await openWllama({
    backend: { executionProviders: ['cpu'], track: 'node' },
    files,
    rels: ['model.gguf'],
    contextLength: 512,
    library: wllamaLike(access, log),
    wasm: { Suspending() {}, promising() {} },
  });
  await release();
  return { reads, log };
}

describe('gguf model files are read in large aligned chunks', () => {
  test('wllama’s 1 MiB tensor reads are coalesced into ≥ 16 MiB chunks', async () => {
    const size = 80 * MIB + 12345;
    const requests = [...tensors(size)].length;
    const { reads, log } = await load(size, tensors);
    assert.equal(log.bad, 0);
    assert.ok(log.bytes > size);
    assert.ok(reads.length < requests / 2, `${reads.length} reads for ${requests}`);
    const chunked = reads.filter((r) => r.length >= 16 * MIB && r.position % (16 * MIB) === 0);
    const bytes = chunked.reduce((a, r) => a + r.length, 0);
    assert.ok(bytes >= size / 2, `${bytes} bytes in chunks`);
  });

  test('scattered reads are not inflated into chunks', async () => {
    const size = 2 * 2 ** 30;
    const requested = [...scattered(size)].reduce((a, [, n]) => a + n, 0);
    const { reads, log } = await load(size, scattered);
    assert.equal(log.bad, 0);
    assert.equal(
      reads.reduce((a, r) => a + r.length, 0),
      requested
    );
  });

  test('a file over 2 GiB is never read whole', async () => {
    const size = 3 * 2 ** 30 + 777;
    const { reads, log } = await load(size, far);
    assert.equal(log.bad, 0);
    assert.ok(reads.length < [...far(size)].length / 2, `${reads.length} reads`);
    assert.ok(Math.max(...reads.map((r) => r.length)) <= 64 * MIB);
    assert.equal(reads.at(-1).position + reads.at(-1).length, size);
  });

  test('the chunks are dropped once the model has loaded', async () => {
    const { reads, log } = await load(20 * MIB, function* () {
      yield [0, MIB];
    });
    const before = reads.length;
    await log.blob.slice(0, 16).arrayBuffer();
    assert.equal(reads.length, before + 1);
  });
});

import assert from 'node:assert/strict';
import {
  lstat,
  mkdir,
  mkdtemp,
  readdir,
  realpath,
  rm,
  stat,
  symlink,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { openModel } from '../../src/engines/decoder/session.js';
import { jev } from '../../src/engines/jev.js';
import { openSession } from '../../src/engines/parakeet.js';
import { systemone } from '../../src/engines/systemone.js';
import { hubFiles } from '../../src/hf/files.js';
import { verifyModel } from '../../src/hf/verify.js';
import { createRuntime, openCatalog } from '../../src/index.js';
import { loadBackend } from '../../src/ort/node.js';
import { loadBackend as loadWebBackend } from '../../src/ort/web.js';
import {
  addGraph,
  hubCache,
  joint,
  KEV,
  PARAKEET,
  REPO,
  repoOf,
  SHA,
  sha256,
  snapshotOf,
  weights,
} from './hub.mjs';

const env = (cache) => ({ HF_HUB_CACHE: cache, ORTLLAMA_MODEL_PATH: '', ORTLLAMA_EP: 'cpu' });
const made = [];
after(() => Promise.all(made.map((dir) => rm(dir, { recursive: true, force: true }))));

async function layoutOf(version, files, options) {
  const out = await hubCache(version, files, options);
  made.push(out.cache);
  if (options?.stray) made.push(dirname(dirname(out.shared[options.stray])));
  return out;
}

async function run(session) {
  const { ort } = await loadBackend();
  const x = new ort.Tensor('float32', new Float32Array([1, 2]), [2]);
  return [...(await session.run({ x })).y.data];
}

function spyBackend() {
  const calls = [];
  const create = async (source, options) => {
    calls.push({ source, options });
    return { inputNames: [], release() {} };
  };
  return { calls, backend: { track: 'node', ort: { InferenceSession: { create } } } };
}

describe('ortllama runs models from the huggingface_hub 2.x shared blob store', () => {
  let cache;
  let shared;
  let catalog;
  before(async () => {
    ({ cache, shared } = await layoutOf(2, PARAKEET));
    catalog = await openCatalog({ env: env(cache) });
  });

  test('lists a repo whose LFS blobs link into <cache>/blobs/<xx>/<hash>', async () => {
    const { models, unrecognized } = await catalog.survey();
    assert.deepEqual(unrecognized, []);
    const found = models.find((m) => m.id === REPO);
    assert.equal(found?.engine, 'parakeet');
    assert.deepEqual(found.variants, ['fp32']);
  });

  test('resolves it with nothing missing and verifies the shared blobs by their etags', async () => {
    const resolved = await catalog.resolve(`${REPO}:fp32`);
    assert.equal(resolved.revision, SHA);
    const blob = await resolved.files.blob('onnx/encoder_model.onnx_data');
    assert.equal(blob.etag, sha256(weights));
    assert.equal(blob.path, await realpath(shared['onnx/encoder_model.onnx_data']));
    assert.equal(await verifyModel(resolved, { memo: new Map() }), 8);
  });

  test('refuses a shared blob whose content does not match its etag', async () => {
    const tampered = await layoutOf(2, PARAKEET);
    await writeFile(tampered.shared['onnx/decoder_joint_model.onnx_data'], 'tampered');
    const resolved = await (await openCatalog({ env: env(tampered.cache) })).resolve(REPO);
    await assert.rejects(verifyModel(resolved, { memo: new Map() }), (error) => {
      assert.equal(error.code, 'model_corrupt');
      assert.ok(error.message.includes(`the hub's etag is ${sha256(joint)}`), error.message);
      return true;
    });
  });

  test('loads it on onnxruntime-node with external data in other shared blob folders', async () => {
    const resolved = await catalog.resolve(REPO);
    const runtime = createRuntime({ env: env(cache), keepAlive: '0' });
    try {
      const { instance } = await runtime.acquire(resolved);
      assert.equal(typeof instance.transcribe, 'function');
    } finally {
      await runtime.close();
    }
    const backend = await loadBackend();
    const session = await openSession(
      { ...backend, executionProviders: ['cpu'] },
      resolved.files,
      'onnx/encoder_model.onnx'
    );
    assert.deepEqual(await run(session), [11, 22]);
  });

  test('hands onnxruntime-node the external data itself, keyed by its name', async () => {
    const { files } = await catalog.resolve(REPO);
    const { calls, backend } = spyBackend();
    await openSession(backend, files, 'onnx/encoder_model.onnx');
    const [{ source, options }] = calls;
    assert.equal(source, files.path('onnx/encoder_model.onnx'));
    assert.deepEqual(
      options.externalData.map(({ path, data }) => [path, Buffer.from(data).equals(weights)]),
      [['encoder_model.onnx_data', true]]
    );
  });

  test('opens a decoder graph with its external data on onnxruntime-node', async () => {
    const { files } = await catalog.resolve(REPO);
    const backend = { ...(await loadBackend()), executionProviders: ['cpu'] };
    const part = {
      decoder: { model: 'onnx/encoder_model.onnx', data: ['onnx/encoder_model.onnx_data'] },
    };
    const model = await openModel({ backend, files, part, shape: { layers: 0 } });
    assert.equal(typeof model, 'object');
    const { calls, backend: spy } = spyBackend();
    await openModel({
      backend: { ...spy, executionProviders: ['cpu'] },
      files,
      part,
      shape: { layers: 0 },
    });
    assert.equal(calls[0].options.externalData[0].path, 'encoder_model.onnx_data');
  });

  test('gives kev.js no file paths, so it reads the weights itself', async () => {
    const { cache: kevCache } = await layoutOf(2, KEV);
    const resolved = await (await openCatalog({ env: env(kevCache) })).resolve(`${REPO}/kev-tiny`);
    let seen;
    const library = async () => ({
      loadKev: async (read, options) => {
        seen = { options, data: await read('r-1/q8f32/model.onnx.data') };
        return { release() {} };
      },
    });
    await systemone.load({ ...resolved, backend: { track: 'node', ort: {} }, library });
    assert.equal(seen.options.filePath, undefined);
    assert.equal(seen.options.readSync, undefined);
    assert.ok(Buffer.from(seen.data).equals(weights));
  });
});

describe('ortllama keeps the shared blob store a tight boundary', () => {
  test('rejects a repo blob that links to a look-alike store outside the cache', async () => {
    const { cache } = await layoutOf(2, PARAKEET, { stray: 'onnx/encoder_model.onnx' });
    const catalog = await openCatalog({ env: env(cache) });
    assert.equal(
      (await catalog.list()).find((m) => m.id === REPO),
      undefined
    );
    await assert.rejects(catalog.resolve(REPO), { status: 404 });
    const files = hubFiles(cache, REPO, snapshotOf(cache));
    assert.equal(await files.exists('onnx/encoder_model.onnx'), false);
    assert.equal(await files.exists('onnx/encoder_model.onnx_data'), true);
  });

  test('rejects files in <cache>/blobs that are not a shard of 64-hex blobs', async () => {
    const { cache, shared } = await layoutOf(2, PARAKEET);
    const repoBlobs = join(repoOf(cache), 'blobs');
    const snapshot = snapshotOf(cache);
    const real = shared['onnx/encoder_model.onnx_data'];
    const odd = {
      'refs.bin': `${real}.refs`,
      'flat.bin': join(cache, 'blobs', 'flat'),
      'shard.bin': join(cache, 'blobs', 'zz', 'f'.repeat(64)),
      'short.bin': join(cache, 'blobs', 'ab', `ab${'c'.repeat(38)}`),
      'nested.bin': join(cache, 'blobs', real.slice(-64, -62), 'x', real.slice(-64)),
      'other.bin': join(cache, 'elsewhere', 'blob'),
    };
    for (const [name, target] of Object.entries(odd)) {
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, 'x');
      const blob = join(repoBlobs, sha256(name));
      await symlink(relative(repoBlobs, target), blob);
      await symlink(relative(snapshot, blob), join(snapshot, name));
    }
    const { files } = await (await openCatalog({ env: env(cache) })).resolve(REPO);
    for (const name of Object.keys(odd)) assert.equal(await files.exists(name), false, name);
    assert.equal(await files.exists('onnx/encoder_model.onnx_data'), true);
  });
});

describe('ortllama keeps the huggingface_hub 1.x layout as it was', () => {
  let catalog;
  before(async () => {
    const { cache } = await layoutOf(1, { ...PARAKEET, ...KEV });
    catalog = await openCatalog({ env: env(cache) });
  });

  test('lists, resolves and verifies it', async () => {
    assert.equal((await catalog.list()).find((m) => m.id === REPO)?.engine, 'parakeet');
    const resolved = await catalog.resolve(REPO);
    assert.equal(await verifyModel(resolved, { memo: new Map() }), 8);
  });

  test('lets onnxruntime-node read the external data beside the graph', async () => {
    const { files } = await catalog.resolve(REPO);
    const { calls, backend } = spyBackend();
    await openSession(backend, files, 'onnx/encoder_model.onnx');
    assert.equal(calls[0].options.externalData, undefined);
    const real = await loadBackend();
    const session = await openSession(
      { ...real, executionProviders: ['cpu'] },
      files,
      'onnx/encoder_model.onnx'
    );
    assert.deepEqual(await run(session), [11, 22]);
  });

  test('gives kev.js file paths for its graphs', async () => {
    const resolved = await catalog.resolve(`${REPO}/kev-tiny`);
    let seen;
    const library = async () => ({
      loadKev: async (_read, options) => {
        seen = options;
        return { release() {} };
      },
    });
    await systemone.load({ ...resolved, backend: { track: 'node', ort: {} }, library });
    assert.equal(
      seen.filePath('r-1/q8f32/model.onnx'),
      resolved.files.path('r-1/q8f32/model.onnx')
    );
  });
});

const BIG = 4 * 1024 * 1024;
const bigWeights = Buffer.from(Float32Array.from({ length: BIG }, (_, i) => i % 7).buffer);
const BIG_PARAKEET = {
  ...PARAKEET,
  'onnx/encoder_model.onnx': addGraph('encoder_model.onnx_data', BIG),
  'onnx/encoder_model.onnx_data': bigWeights,
};
const ENCODER = 'onnx/encoder_model.onnx';
const ENCODER_DATA = 'onnx/encoder_model.onnx_data';

async function stagingBackend() {
  const real = await loadBackend();
  const root = await mkdtemp(join(tmpdir(), 'ortllama-stage-'));
  made.push(root);
  const calls = [];
  const create = async (source, options) => {
    const beside = (name) => join(dirname(source), name);
    const listing = await readdir(dirname(source));
    const inodes = Object.fromEntries(
      await Promise.all(
        listing.map(async (name) => {
          const info = await lstat(beside(name));
          return [name, info.isFile() ? info.ino : null];
        })
      )
    );
    calls.push({ source, options, inodes, arrayBuffers: process.memoryUsage().arrayBuffers });
    return real.ort.InferenceSession.create(source, options);
  };
  return {
    root,
    calls,
    backend: {
      ...real,
      executionProviders: ['cpu'],
      sessionOptions: { ...real.sessionOptions, intraOpNumThreads: 1, interOpNumThreads: 1 },
      ort: { ...real.ort, InferenceSession: { create } },
      stage: (files, rels) => real.stage(files, rels, root),
    },
  };
}

const inode = async (path) => (await stat(path)).ino;

describe('ortllama lets onnxruntime-node read 2.x external data from disk', () => {
  let cache;
  let shared;
  let files;
  before(async () => {
    ({ cache, shared } = await layoutOf(2, BIG_PARAKEET));
    ({ files } = await (await openCatalog({ env: env(cache) })).resolve(REPO));
  });

  test('opens a Parakeet graph beside hard links to its shared blobs, with no weights in JS', async () => {
    const { root, calls, backend } = await stagingBackend();
    globalThis.gc?.();
    const before = process.memoryUsage().arrayBuffers;
    const session = await openSession(backend, files, ENCODER);
    const [{ source, options, inodes, arrayBuffers }] = calls;
    assert.ok(arrayBuffers - before < bigWeights.length / 2, `${arrayBuffers - before} bytes`);
    assert.equal(Object.hasOwn(options, 'externalData'), false);
    assert.ok(source.startsWith(root), source);
    assert.deepEqual(inodes, {
      'encoder_model.onnx': await inode(shared[ENCODER]),
      'encoder_model.onnx_data': await inode(shared[ENCODER_DATA]),
    });
    assert.deepEqual(await readdir(root), []);
    const { ort } = backend;
    const x = new ort.Tensor('float32', new Float32Array(BIG).fill(1), [BIG]);
    const y = (await session.run({ x })).y.data;
    assert.deepEqual([y[0], y[1], y[6], y[BIG - 1]], [1, 2, 7, ((BIG - 1) % 7) + 1]);
    await session.release();
  });

  test('opens decoder graphs the same way', async () => {
    const { root, calls, backend } = await stagingBackend();
    const part = { decoder: { model: ENCODER, data: [ENCODER_DATA] } };
    const model = await openModel({ backend, files, part, shape: { layers: 0 } });
    assert.equal(Object.hasOwn(calls[0].options, 'externalData'), false);
    assert.ok(calls[0].source.startsWith(root), calls[0].source);
    assert.equal(calls[0].inodes['encoder_model.onnx_data'], await inode(shared[ENCODER_DATA]));
    assert.deepEqual(await readdir(root), []);
    await model.release();
  });

  test('gives kev.js paths to its graphs beside their data while it loads', async () => {
    const { cache: kevCache, shared: kevShared } = await layoutOf(2, KEV);
    const resolved = await (await openCatalog({ env: env(kevCache) })).resolve(`${REPO}/kev-tiny`);
    const { root, backend } = await stagingBackend();
    let seen;
    const library = async () => ({
      loadKev: async (_read, options) => {
        const graph = options.filePath('r-1/q8f32/model.onnx');
        seen = {
          graph,
          ino: await inode(join(dirname(graph), 'model.onnx.data')),
          readSync: options.readSync,
        };
        return { release() {} };
      },
    });
    await systemone.load({ ...resolved, backend, library });
    assert.ok(seen.graph.startsWith(root), seen.graph);
    assert.equal(seen.ino, await inode(kevShared['kev-tiny/r-1/q8f32/model.onnx.data']));
    assert.equal(seen.readSync, undefined);
    assert.deepEqual(await readdir(root), []);
  });

  test('removes the kev.js and jev-omni.js folder when the library throws at once', async () => {
    const { cache: kevCache } = await layoutOf(2, KEV);
    const resolved = await (await openCatalog({ env: env(kevCache) })).resolve(`${REPO}/kev-tiny`);
    const { root, backend } = await stagingBackend();
    const sync = (options) => {
      assert.ok(options.filePath('r-1/q8f32/model.onnx').startsWith(root));
      throw new Error('thrown at once');
    };
    const kevLibrary = async () => ({ loadKev: (_read, options) => sync(options) });
    await assert.rejects(
      systemone.load({ ...resolved, backend, library: kevLibrary }),
      /thrown at once/
    );
    assert.deepEqual(await readdir(root), []);
    const jevLibrary = async () => ({ loadJevOmni: (_read, options) => sync(options) });
    await assert.rejects(jev.load({ ...resolved, backend, library: jevLibrary }), /thrown at once/);
    assert.deepEqual(await readdir(root), []);
  });

  test('falls back to handing over the bytes when it cannot link the blobs', async () => {
    const { calls, backend } = await stagingBackend();
    const failing = {
      ...backend,
      stage: async () => {
        throw Object.assign(new Error('cross-device link'), { code: 'EXDEV' });
      },
    };
    const session = await openSession(failing, files, ENCODER);
    assert.equal(calls[0].source, files.path(ENCODER));
    assert.equal(calls[0].options.externalData[0].path, 'encoder_model.onnx_data');
    await session.release();
  });

  test('links nothing that the cache boundary refuses and leaves no folder behind', async () => {
    const { cache: strayCache } = await layoutOf(2, PARAKEET, { stray: ENCODER_DATA });
    const strayFiles = hubFiles(strayCache, REPO, snapshotOf(strayCache));
    const { root, backend } = await stagingBackend();
    await assert.rejects(backend.stage(strayFiles, [ENCODER, ENCODER_DATA]), { code: 'ENOENT' });
    assert.deepEqual(await readdir(root), []);
    await assert.rejects(openSession(backend, strayFiles, ENCODER));
    assert.deepEqual(await readdir(root), []);
  });

  test('leaves onnxruntime-web reading the bytes itself', async () => {
    const web = await loadWebBackend({ env: {}, gpu: undefined });
    const calls = [];
    const create = async (source, options) => {
      calls.push({ bytes: source instanceof Uint8Array, external: options.externalData.length });
      return web.ort.InferenceSession.create(source, options);
    };
    const backend = { ...web, ort: { ...web.ort, InferenceSession: { create } } };
    const tiny = await (
      await openCatalog({ env: env((await layoutOf(2, PARAKEET)).cache) })
    ).resolve(REPO);
    const session = await openSession(backend, tiny.files, ENCODER);
    assert.deepEqual(calls, [{ bytes: true, external: 1 }]);
    const x = new web.ort.Tensor('float32', new Float32Array([1, 2]), [2]);
    assert.deepEqual([...(await session.run({ x })).y.data], [11, 22]);
    await session.release();
  });
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { openModel } from '../../src/engines/decoder/session.js';
import { openSession } from '../../src/engines/parakeet.js';
import { systemone } from '../../src/engines/systemone.js';
import { hubFiles } from '../../src/hf/files.js';
import { verifyModel } from '../../src/hf/verify.js';
import { createRuntime, openCatalog } from '../../src/index.js';
import { loadBackend } from '../../src/ort/node.js';

const SHA = 'b'.repeat(40);
const REPO = 'test/bark';
const WEIGHTS = [10, 20];

const varint = (n) => {
  const out = [];
  for (; n > 127; n >>>= 7) out.push((n & 127) | 128);
  return [...out, n];
};
const tag = (field, wire) => varint((field << 3) | wire);
const int = (field, n) => [...tag(field, 0), ...varint(n)];
const msg = (field, body) => [...tag(field, 2), ...varint(body.length), ...body];
const str = (field, s) => msg(field, [...Buffer.from(s)]);
const entry = (key, value) => msg(13, [...str(1, key), ...str(2, value)]);
const vector = (name) => [
  ...str(1, name),
  ...msg(2, msg(1, [...int(1, 1), ...msg(2, msg(1, int(1, 2)))])),
];

function addGraph(location) {
  const w = [
    ...int(1, 2),
    ...int(2, 1),
    ...str(8, 'w'),
    ...entry('location', location),
    ...entry('offset', '0'),
    ...entry('length', '8'),
    ...int(14, 1),
  ];
  const node = msg(1, [...str(1, 'x'), ...str(1, 'w'), ...str(2, 'y'), ...str(4, 'Add')]);
  const graph = [
    ...node,
    ...str(2, 'g'),
    ...msg(5, w),
    ...msg(11, vector('x')),
    ...msg(12, vector('y')),
  ];
  return Buffer.from([...int(1, 8), ...msg(7, graph), ...msg(8, [...str(1, ''), ...int(2, 13)])]);
}

const weights = Buffer.from(new Float32Array(WEIGHTS).buffer);
const joint = Buffer.from(new Float32Array([30, 40]).buffer);
const sha256 = (content) => createHash('sha256').update(content).digest('hex');
const gitSha = (content) =>
  createHash('sha1').update(`blob ${content.length}\0`).update(content).digest('hex');

const PARAKEET = {
  'config.json': JSON.stringify({ model_type: 'parakeet_tdt' }),
  'generation_config.json': JSON.stringify({ eos_token_id: 1 }),
  'preprocessor_config.json': '{}',
  'tokenizer.json': JSON.stringify({ model: { vocab: { a: 0 } } }),
  'onnx/encoder_model.onnx': addGraph('encoder_model.onnx_data'),
  'onnx/encoder_model.onnx_data': weights,
  'onnx/decoder_joint_model.onnx': addGraph('decoder_joint_model.onnx_data'),
  'onnx/decoder_joint_model.onnx_data': joint,
};

const KEV = {
  'kev-tiny/manifest.json': JSON.stringify({
    name: 'kev-tiny',
    special: { state: 1 },
    files: { tokenizer: 'r-1/tokenizer.json' },
    variants: { q8f32: { model: 'r-1/q8f32/model.onnx', data: ['r-1/q8f32/model.onnx.data'] } },
  }),
  'kev-tiny/r-1/tokenizer.json': '{}',
  'kev-tiny/r-1/q8f32/model.onnx': addGraph('model.onnx.data'),
  'kev-tiny/r-1/q8f32/model.onnx.data': weights,
};

const isLfs = (rel) => /\.onnx|_data|\.data/.test(rel);

const repoOf = (cache) => join(cache, `models--${REPO.replace('/', '--')}`);
const snapshotOf = (cache) => join(repoOf(cache), 'snapshots', SHA);

async function hubCache(layout, files, { stray = null } = {}) {
  const cache = await mkdtemp(join(tmpdir(), `ortllama-hub${layout}-`));
  const repo = repoOf(cache);
  const snapshot = snapshotOf(cache);
  await mkdir(join(repo, 'refs'), { recursive: true });
  await writeFile(join(repo, 'refs', 'main'), SHA);
  const shared = {};
  for (const [rel, text] of Object.entries(files)) {
    const content = Buffer.from(text);
    const lfs = isLfs(rel);
    const blob = join(repo, 'blobs', lfs ? sha256(content) : gitSha(content));
    await mkdir(dirname(blob), { recursive: true });
    if (layout === 2 && lfs) {
      const xet = sha256(Buffer.concat([Buffer.from(rel), content]));
      const store = rel === stray ? join(cache, '..', `${xet}-elsewhere`) : join(cache, 'blobs');
      shared[rel] = join(store, xet.slice(0, 2), xet);
      await mkdir(dirname(shared[rel]), { recursive: true });
      await writeFile(shared[rel], content);
      await symlink(relative(dirname(blob), shared[rel]), blob);
    } else await writeFile(blob, content);
    const link = join(snapshot, rel);
    await mkdir(dirname(link), { recursive: true });
    await symlink(relative(dirname(link), blob), link);
  }
  return { cache, shared };
}

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

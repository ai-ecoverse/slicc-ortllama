import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, open, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { modelFiles } from '../../src/hf/files.js';
import { blobDigest, verifyModel, WHOLE_MAX } from '../../src/hf/verify.js';
import { startServer } from '../../src/index.js';
import { fakeCache, fakeHome, SHA } from './cache.mjs';

const FILE = 'r-1/q8f32/model.onnx.data';

const field = (n, bytes) => Buffer.concat([Buffer.from([(n << 3) | 2, bytes.length]), bytes]);

function graphNaming(location) {
  const entry = Buffer.concat([field(1, Buffer.from('location')), field(2, Buffer.from(location))]);
  const tensor = Buffer.concat([field(13, entry), Buffer.from([14 << 3, 1])]);
  return field(7, field(5, tensor));
}

async function hubRepo(entries) {
  const root = await mkdtemp(join(tmpdir(), 'ortllama-verify-'));
  const blobs = join(root, 'blobs');
  const snapshot = join(root, 'snapshots', SHA);
  await mkdir(join(snapshot, 'onnx'), { recursive: true });
  await mkdir(blobs);
  const paths = {};
  for (const [rel, { content, etag = sha256(content) }] of Object.entries(entries)) {
    paths[rel] = join(blobs, etag);
    await writeFile(paths[rel], content);
    await symlink(paths[rel], join(snapshot, rel));
  }
  return { files: modelFiles(snapshot, { allow: [snapshot, blobs] }), paths };
}

const sha256 = (content) => createHash('sha256').update(content).digest('hex');

describe('ortllama refuses a cached blob that does not match its etag', () => {
  let server;
  let etag;
  let actual;
  const decide = (model) =>
    fetch(`${server.url}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        state: 'x',
        questions: { a: { type: 'noul', instructions: '?' } },
      }),
    });

  before(async () => {
    const cache = await fakeCache();
    const link = join(cache, 'models--test--kev', 'snapshots', SHA, 'kev-tiny', FILE);
    const blob = await realpath(link);
    etag = basename(blob);
    const tampered = 'data q8f32, tampered';
    actual = createHash('sha256').update(tampered).digest('hex');
    await writeFile(blob, tampered);
    const home = await fakeHome();
    server = await startServer({ env: { HF_HUB_CACHE: cache, ORTLLAMA_HOME: home }, port: 0 });
  });
  after(() => server.close());

  test('names the file and both hashes before loading the variant', async () => {
    const response = await decide('test/kev/kev-tiny:q8f32');
    assert.equal(response.status, 500);
    const { error } = await response.json();
    assert.equal(error.code, 'model_corrupt');
    assert.equal(error.param, 'model');
    assert.ok(error.message.includes(`corrupt file ${FILE}:`), error.message);
    assert.ok(error.message.includes(`its sha256 is ${actual}`), error.message);
    assert.ok(error.message.includes(`the hub's etag is ${etag}`), error.message);
    assert.ok(
      error.message.endsWith(
        `run: hf download test/kev --include 'kev-tiny/${FILE}' --force-download`
      ),
      error.message
    );
  });

  test('still refuses it on the next request', async () => {
    const { error } = await (await decide('test/kev/kev-tiny:q8f32')).json();
    assert.equal(error.code, 'model_corrupt');
  });

  test('lets an intact variant through to its engine', async () => {
    const { error } = await (await decide('test/kev/kev-tiny:fp32')).json();
    assert.notEqual(error?.code, 'model_corrupt');
  });
});

describe('ortllama verifies the external data an ONNX graph names', () => {
  test('refuses a tampered weights file the include patterns do not cover', async () => {
    const etag = sha256('weights');
    const { files } = await hubRepo({
      'onnx/model.onnx': { content: graphNaming('weights.bin') },
      'onnx/weights.bin': { content: 'tampered', etag },
    });
    const resolved = {
      id: 'o/r',
      variant: 'v',
      files,
      info: {},
      engine: { include: () => ['onnx/model.onnx*'] },
    };
    await assert.rejects(verifyModel(resolved, { memo: new Map() }), (error) => {
      assert.equal(error.code, 'model_corrupt');
      assert.ok(error.message.includes('corrupt file onnx/weights.bin:'), error.message);
      assert.ok(error.message.includes(`its sha256 is ${sha256('tampered')}`), error.message);
      assert.ok(error.message.includes(`the hub's etag is ${etag}`), error.message);
      return true;
    });
  });

  test('checks an intact one, and skips a named file that is not cached', async () => {
    const { files } = await hubRepo({
      'onnx/model.onnx': { content: graphNaming('weights.bin') },
      'onnx/weights.bin': { content: 'weights' },
      'onnx/other.onnx': { content: graphNaming('absent.bin') },
    });
    const resolved = { files, info: {}, engine: { include: () => ['onnx/*'] } };
    assert.equal(await verifyModel(resolved, { memo: new Map() }), 3);
  });
});

test('hashes a blob too large for one buffer in ranged reads when it would read whole', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'ortllama-verify-big-'));
  const path = join(dir, 'big.gguf');
  const size = WHOLE_MAX + 2 ** 23;
  const handle = await open(path, 'w');
  await handle.truncate(size);
  await handle.write(Buffer.from('end'), 0, 3, size - 3);
  await handle.close();
  const hash = createHash('sha256');
  const zeros = Buffer.alloc(2 ** 24);
  for (let left = size - 3; left > 0; left -= zeros.length)
    hash.update(zeros.subarray(0, Math.min(left, zeros.length)));
  const expected = hash.update('end').digest('hex');
  try {
    const found = await blobDigest({ path, etag: expected, size }, { read: 'whole' });
    assert.deepEqual(found, { algorithm: 'sha256', digest: expected });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { realpath, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { startServer } from '../../src/index.js';
import { fakeCache, fakeHome, SHA } from './cache.mjs';

const FILE = 'r-1/q8f32/model.onnx.data';

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

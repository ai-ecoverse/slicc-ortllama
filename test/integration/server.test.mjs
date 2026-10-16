import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { request } from 'node:http';
import { join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { promisify } from 'node:util';
import { startServer } from '../../src/index.js';
import { fakeCache, fakeHome } from './cache.mjs';

const run = promisify(execFile);
const bin = new URL('../../src/bin.js', import.meta.url).pathname;

describe('ortllama server on a Hugging Face cache', () => {
  let server;
  let cache;
  let home;
  const post = (path, body, headers = {}) =>
    fetch(`${server.url}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    });

  before(async () => {
    cache = await fakeCache();
    home = await fakeHome({ decisions: 'test/kev/kev-tiny', broken: 'nobody/none' });
    server = await startServer({ env: { HF_HUB_CACHE: cache, ORTLLAMA_HOME: home }, port: 0 });
  });
  after(() => server.close());

  test('answers its root without a key', async () => {
    const response = await fetch(`${server.url}/`);
    assert.equal(response.status, 200);
    assert.match(await response.text(), /^ortllama .* is running$/);
  });

  test('lists cached models through blob symlinks, with aliases', async () => {
    const { object, data } = await (await fetch(`${server.url}/v1/models`)).json();
    assert.equal(object, 'list');
    const ids = data.map((m) => m.id);
    assert.deepEqual(ids.slice(0, 2), ['test/kev/kev-part', 'test/kev/kev-tiny']);
    const tiny = data.find((m) => m.id === 'test/kev/kev-tiny');
    assert.deepEqual(tiny.architecture.output_modalities, ['decisions']);
    assert.equal(tiny.context_length, 4096);
    assert.deepEqual(tiny.ortllama.variants, ['q8f32', 'fp32']);
    const alias = data.find((m) => m.id === 'decisions');
    assert.equal(alias.ortllama.alias_of, 'test/kev/kev-tiny');
    assert.equal(
      data.some((m) => m.id === 'broken'),
      false
    );
  });

  test('names the hf download command for an incomplete variant', async () => {
    const response = await post('/v1/systemone', {
      model: 'test/kev/kev-part',
      state: 'x',
      questions: { a: { type: 'noul', instructions: '?' } },
    });
    assert.equal(response.status, 404);
    const { error } = await response.json();
    assert.equal(error.code, 'model_incomplete');
    assert.match(
      error.message,
      /hf download test\/kev --include 'kev-part\/manifest.json' .*'kev-part\/r-1\/q8f32\/\*'/
    );
  });

  test('names the hf download command for a model that is not cached', async () => {
    const missing = await (
      await post('/v1/systemone', { model: 'nobody/none@v2', state: 'x', questions: {} })
    ).json();
    assert.equal(missing.error.code, 'model_not_found');
    assert.match(missing.error.message, /hf download nobody\/none --revision v2$/);
    const sub = await (
      await post('/v1/systemone', { model: 'test/kev/kev-gone', state: 'x', questions: {} })
    ).json();
    assert.match(sub.error.message, /hf download test\/kev --include 'kev-gone\/\*'$/);
  });

  test('rejects unknown variants and non-decision models', async () => {
    const variant = await (
      await post('/v1/systemone', { model: 'test/kev/kev-tiny:int4', state: 'x', questions: {} })
    ).json();
    assert.equal(variant.error.code, 'variant_not_found');
    assert.match(variant.error.message, /q8f32, fp32/);
    const bad = await post('/v1/systemone', { model: '', state: 'x', questions: {} });
    assert.equal(bad.status, 400);
  });

  test('answers malformed requests with the OpenAI error envelope', async () => {
    const json = await post('/v1/systemone', '{not json');
    assert.equal(json.status, 400);
    assert.equal((await json.json()).error.type, 'invalid_request_error');
    const array = await post('/v1/systemone', []);
    assert.equal(array.status, 400);
    const route = await fetch(`${server.url}/v1/nothing`);
    assert.equal(route.status, 404);
    assert.equal((await route.json()).error.code, 'unknown_route');
  });

  test('refuses path tricks in model ids and foreign Host headers', async () => {
    const traversal = await post('/v1/systemone', {
      model: 'nobody/none@../../../../tmp/pointer',
      state: 'x',
      questions: {},
    });
    assert.equal(traversal.status, 400);
    const injected = await post('/v1/systemone', {
      model: 'test/kev@main;touch /tmp/pwned',
      state: 'x',
      questions: {},
    });
    assert.equal(injected.status, 400);
    const status = await new Promise((resolve, reject) => {
      const r = request(
        `${server.url}/v1/models`,
        { headers: { host: 'rebind.example:11435' } },
        (res) => {
          res.resume();
          resolve(res.statusCode);
        }
      );
      r.on('error', reject);
      r.end();
    });
    assert.equal(status, 403);
  });

  test('answers CORS preflights for kernel loopback origins only', async () => {
    const kernel = await fetch(`${server.url}/v1/models`, {
      method: 'OPTIONS',
      headers: { origin: 'http://11435.kernel.localhost' },
    });
    assert.equal(kernel.status, 204);
    assert.equal(
      kernel.headers.get('access-control-allow-origin'),
      'http://11435.kernel.localhost'
    );
    const other = await fetch(`${server.url}/v1/models`, {
      method: 'OPTIONS',
      headers: { origin: 'https://example.com' },
    });
    assert.equal(other.headers.get('access-control-allow-origin'), null);
  });

  test('requires the API key when one is set', async () => {
    const locked = await startServer({
      env: { HF_HUB_CACHE: cache, ORTLLAMA_HOME: home, ORTLLAMA_API_KEY: 'sekrit' },
      port: 0,
    });
    try {
      const denied = await fetch(`${locked.url}/v1/models`);
      assert.equal(denied.status, 401);
      assert.equal((await denied.json()).error.type, 'authentication_error');
      const allowed = await fetch(`${locked.url}/v1/models`, {
        headers: { authorization: 'Bearer sekrit' },
      });
      assert.equal(allowed.status, 200);
      assert.equal((await fetch(`${locked.url}/`)).status, 200);
    } finally {
      await locked.close();
    }
  });

  test('the CLI lists models and manages aliases', async () => {
    const env = { ...process.env, HF_HUB_CACHE: cache, ORTLLAMA_HOME: home };
    const ls = JSON.parse((await run(process.execPath, [bin, 'ls', '--json'], { env })).stdout);
    assert.deepEqual(
      ls.models.map((m) => m.id),
      ['test/kev/kev-part', 'test/kev/kev-tiny']
    );
    await run(process.execPath, [bin, 'alias', 'speech', 'test/kev/kev-tiny'], { env });
    const config = JSON.parse(await readFile(join(home, 'config.json'), 'utf8'));
    assert.equal(config.aliases.speech, 'test/kev/kev-tiny');
    const { stdout } = await run(process.execPath, [bin, '--version'], { env });
    assert.match(stdout, /^\d+\.\d+\.\d+/);
  });
});

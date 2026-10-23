import assert from 'node:assert/strict';
import { execFile, spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { promisify } from 'node:util';
import { openCatalog } from '../../src/index.js';
import { loadBackend as loadNodeBackend } from '../../src/ort/node.js';
import { loadBackend as loadWebBackend } from '../../src/ort/web.js';
import { foxWav } from './audio.mjs';
import { hubCache, PARAKEET, REPO } from './hub.mjs';

const run = promisify(execFile);
const HIDE = new URL('./hide-ort-node.mjs', import.meta.url).pathname;
const BIN = new URL('../../src/bin.js', import.meta.url).pathname;
const INDEX = new URL('../../src/index.js', import.meta.url).href;
const PARAKEET_ENGINE = new URL('../../src/engines/parakeet.js', import.meta.url).href;
const NOTICE =
  /onnxruntime-node is not available \(Cannot find package 'onnxruntime-node'\); running on onnxruntime-web WASM, CPU only/g;
const WHISPER = 'onnx-community/whisper-tiny';

const made = [];
after(() => Promise.all(made.map((dir) => rm(dir, { recursive: true, force: true }))));

async function scratch(prefix) {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  made.push(dir);
  return dir;
}

async function hidden(script, env) {
  try {
    return await run(process.execPath, ['--import', HIDE, '--input-type=module', '-e', script], {
      env,
    });
  } catch (error) {
    assert.fail(`${error.message}\n${error.stderr}`);
  }
}

const LOAD = `
import { createRuntime, openCatalog } from ${JSON.stringify(INDEX)};
import { openSession } from ${JSON.stringify(PARAKEET_ENGINE)};
const resolved = await (await openCatalog()).resolve(${JSON.stringify(REPO)});
const first = createRuntime({ keepAlive: '0' });
await first.acquire(resolved);
const [ps] = first.ps();
await first.close();
const second = createRuntime({ keepAlive: '0' });
const backend = await second.backend();
const session = await openSession(backend, resolved.files, 'onnx/encoder_model.onnx');
const x = new backend.ort.Tensor('float32', new Float32Array([1, 2]), [2]);
const y = [...(await session.run({ x })).y.data];
await session.release();
console.log(JSON.stringify({ track: backend.track, available: backend.availableProviders, ps, y }));
`;

describe('ortllama on Node without onnxruntime-node', () => {
  let env;
  before(async () => {
    const { cache } = await hubCache(2, PARAKEET);
    made.push(cache);
    env = {
      ...process.env,
      HF_HUB_CACHE: cache,
      ORTLLAMA_MODEL_PATH: '',
      ORTLLAMA_HOME: await scratch('ortllama-home-'),
      ORTLLAMA_EP: '',
    };
  });

  test('runs a hub 2.x model with external data on onnxruntime-web WASM, noticing once', async () => {
    const { stdout, stderr } = await hidden(LOAD, env);
    const report = JSON.parse(stdout);
    assert.equal(stderr.match(NOTICE)?.length, 1, stderr);
    assert.equal(report.track, 'web');
    assert.deepEqual(report.available, ['wasm', 'cpu']);
    assert.deepEqual(
      [report.ps.id, report.ps.backend, report.ps.executionProviders, report.ps.providerSource],
      [REPO, 'onnxruntime-web', ['wasm'], 'default']
    );
    assert.deepEqual(report.y, [11, 22]);
  });

  test('skips GPU providers asked for in ORTLLAMA_EP and loads on wasm', async () => {
    const { stdout, stderr } = await hidden(LOAD, { ...env, ORTLLAMA_EP: 'cuda,webgpu,cpu' });
    const { ps } = JSON.parse(stdout);
    assert.deepEqual([ps.executionProviders, ps.providerSource], [['cpu', 'wasm'], 'env']);
    assert.match(stderr, /onnxruntime-web has no cuda execution provider here; skipping it/);
    assert.match(stderr, /onnxruntime-web has no webgpu execution provider here; skipping it/);
  });

  test('keeps onnxruntime-node when it is there', async () => {
    const backend = await loadNodeBackend({ env: {}, hostEnv: {} });
    assert.equal(backend.track, 'node');
    assert.ok(backend.availableProviders.includes('cpu'));
  });
});

describe('the backend loaders as package exports', () => {
  test('ort/node and ort/web are the loaders ortllama uses', async () => {
    const node = await import('@ai-ecoverse/slicc-ortllama/ort/node');
    const web = await import('@ai-ecoverse/slicc-ortllama/ort/web');
    assert.equal(node.loadBackend, loadNodeBackend);
    assert.equal(web.loadBackend, loadWebBackend);
    const backend = await web.loadBackend({ env: {}, gpu: null });
    assert.deepEqual([backend.track, backend.executionProviders], ['web', ['wasm']]);
  });
});

async function whisperCached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(
      WHISPER
    );
    return true;
  } catch {
    return false;
  }
}

const skip =
  !(await whisperCached()) && !process.env.ORTLLAMA_REQUIRE_MODEL && `${WHISPER} is not cached`;

describe('ortllama serve without onnxruntime-node', { skip }, () => {
  let child;
  let url;
  let env;
  let stderr = '';

  before(async () => {
    env = {
      ...process.env,
      ORTLLAMA_EP: '',
      ORTLLAMA_API_KEY: '',
      ORTLLAMA_MODEL_PATH: '',
      ORTLLAMA_HOME: await scratch('ortllama-home-'),
    };
    child = spawn(process.execPath, ['--import', HIDE, BIN, 'serve', '--port', '0'], { env });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    url = await new Promise((resolve, reject) => {
      let out = '';
      child.stdout.on('data', (chunk) => {
        out += chunk;
        const match = /listening on (\S+)/.exec(out);
        if (match) resolve(match[1]);
      });
      child.once('exit', (code) => reject(new Error(`serve exited with ${code}: ${stderr}`)));
    });
  });
  after(() => child?.kill('SIGTERM'));

  test('transcribes with Whisper on wasm and reports onnxruntime-web in ps', async () => {
    for (let i = 0; i < 2; i++) {
      const form = new FormData();
      form.append('model', WHISPER);
      form.append('file', new Blob([await foxWav()], { type: 'audio/wav' }), 'fox.wav');
      const response = await fetch(`${url}/v1/audio/transcriptions`, {
        method: 'POST',
        body: form,
      });
      assert.equal(response.status, 200, await response.clone().text());
      assert.match((await response.json()).text, /fox/i);
    }
    const { stdout } = await run(process.execPath, [BIN, 'ps', '--json'], { env });
    const [model] = JSON.parse(stdout).models;
    assert.deepEqual(
      [model.id, model.backend, model.executionProviders],
      [WHISPER, 'onnxruntime-web', ['wasm']]
    );
    const table = await run(process.execPath, [BIN, 'ps'], { env });
    assert.match(table.stdout, /BACKEND\s+PROVIDERS/);
    assert.match(table.stdout, /onnxruntime-web\s+wasm/);
    assert.equal(stderr.match(NOTICE)?.length, 1, stderr);
  });
});

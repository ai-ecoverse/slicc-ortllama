import assert from 'node:assert/strict';
import { execFile, spawn } from 'node:child_process';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { promisify } from 'node:util';
import { createRuntime, openCatalog } from '../../src/index.js';
import { foxWav } from './audio.mjs';

const WHISPER = 'onnx-community/whisper-tiny';
const SILERO = 'onnx-community/silero-vad';
const BIN = new URL('../../src/bin.js', import.meta.url).pathname;

async function cached() {
  try {
    const catalog = await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } });
    await catalog.resolve(WHISPER);
    await catalog.resolve(SILERO);
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${WHISPER} or ${SILERO} is not cached`;

async function homeWith(hints) {
  const home = await mkdtemp(join(tmpdir(), 'ortllama-ep-'));
  await writeFile(join(home, 'config.json'), JSON.stringify({ aliases: {}, hints }));
  return home;
}

async function loaded(env, id) {
  const warnings = [];
  const runtime = createRuntime({ env, keepAlive: '0' });
  runtime.warn = (m) => warnings.push(m);
  const catalog = await openCatalog({ env: { ...env, ORTLLAMA_MODEL_PATH: '' } });
  const resolved = await catalog.resolve(id);
  const entry = await runtime.acquire(resolved);
  const [ps] = runtime.ps();
  await runtime.close();
  return { resolved, entry, ps, warnings };
}

describe('execution providers on real models', { skip }, () => {
  const env = { ...process.env, ORTLLAMA_EP: '' };

  test('Whisper and Silero VAD default to their engine preference, cpu', async () => {
    for (const id of [WHISPER, SILERO]) {
      const { ps } = await loaded({ ...env, ORTLLAMA_HOME: await homeWith({}) }, id);
      assert.deepEqual([ps.executionProviders, ps.providerSource], [['cpu'], 'engine']);
      assert.ok(ps.modelBytes > 1_000_000);
    }
  });

  test('a hint moves Whisper to WebGPU and keys it apart', async () => {
    const home = await homeWith({ [WHISPER]: { executionProviders: ['webgpu', 'cpu'] } });
    const { resolved, ps } = await loaded({ ...env, ORTLLAMA_HOME: home }, WHISPER);
    assert.match(resolved.key, /\+ep=webgpu,cpu$/);
    assert.equal(ps.providerSource, 'hint');
    if (process.platform === 'darwin' && process.arch === 'arm64')
      assert.deepEqual(ps.executionProviders, ['webgpu', 'cpu']);
    else assert.equal(ps.executionProviders.at(-1), 'cpu');
  });

  test('a provider this onnxruntime lacks is skipped, and ORTLLAMA_EP beats hints', async () => {
    const home = await homeWith({ [SILERO]: { executionProviders: ['nope', 'cpu'] } });
    const hinted = await loaded({ ...env, ORTLLAMA_HOME: home }, SILERO);
    assert.deepEqual(hinted.ps.executionProviders, ['cpu']);
    assert.match(hinted.warnings.join('\n'), /has no nope execution provider/);
    const pinned = await loaded({ ...env, ORTLLAMA_HOME: home, ORTLLAMA_EP: 'cpu' }, SILERO);
    assert.deepEqual([pinned.ps.executionProviders, pinned.ps.providerSource], [['cpu'], 'env']);
  });
});

describe('ortllama ps --json on a running server', { skip }, () => {
  let child;
  let url;
  let env;

  before(async () => {
    env = {
      ...process.env,
      ORTLLAMA_EP: '',
      ORTLLAMA_API_KEY: '',
      ORTLLAMA_HOME: await homeWith({}),
    };
    child = spawn(process.execPath, [BIN, 'serve', '--port', '0'], { env });
    url = await new Promise((resolve, reject) => {
      let out = '';
      child.stdout.on('data', (chunk) => {
        out += chunk;
        const match = /listening on (\S+)/.exec(out);
        if (match) resolve(match[1]);
      });
      child.once('exit', (code) => reject(new Error(`serve exited with ${code}`)));
    });
  });
  after(() => child?.kill('SIGTERM'));

  test('shows the providers, size and process memory of a loaded model', async () => {
    const form = new FormData();
    form.append('model', WHISPER);
    form.append('file', new Blob([await foxWav()], { type: 'audio/wav' }), 'fox.wav');
    const response = await fetch(`${url}/v1/audio/transcriptions`, { method: 'POST', body: form });
    assert.equal(response.status, 200);
    const { stdout } = await promisify(execFile)(process.execPath, [BIN, 'ps', '--json'], { env });
    const report = JSON.parse(stdout);
    const [model] = report.models;
    assert.deepEqual(
      [model.id, model.executionProviders, model.providerSource],
      [WHISPER, ['cpu'], 'engine']
    );
    assert.ok(model.modelBytes > 1_000_000);
    for (const field of ['rss', 'heapUsed', 'external', 'arrayBuffers'])
      assert.ok(report.memory[field] > 0, field);
  });
});

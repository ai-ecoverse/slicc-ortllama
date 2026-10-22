import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { createCatalog } from '../../src/catalog.js';
import { streamChat } from '../../src/engines/gguf/chat.js';
import { createRuntime, startServer } from '../../src/index.js';
import { chatMeta, decisionMeta, ggufBytes, ggufCache, projectorMeta } from './gguf.mjs';

const WORKER = `
onmessage = async (e) => {
  if (e.data.verb !== 'init') return;
  global.Worker = 'leaked';
  process.exitCode = 3;
  const adapter = await navigator.gpu.requestAdapter();
  const device = adapter ? await adapter.requestDevice() : null;
  postMessage({
    verb: 'ready',
    gpu: Boolean(device),
    dir: __dirname,
    waitAsync: typeof Atomics.waitAsync,
    os: typeof require('node:os').cpus,
    error: typeof GPUValidationError,
  });
};`;

async function fakeCompletion(log, params) {
  log.chat.push(params);
  const chunks = log.chunks;
  const run = async (onData) => {
    for (const chunk of chunks) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, log.delay ?? 0);
        params.abortSignal?.addEventListener('abort', () => resolve(clearTimeout(timer)), {
          once: true,
        });
      });
      if (params.abortSignal?.aborted)
        throw Object.assign(new Error('Operation aborted'), { name: 'AbortError' });
      if (chunk instanceof Error) throw chunk;
      onData(chunk);
    }
  };
  if (params.onData) return run(params.onData);
  const values = [];
  let resolve;
  let reject;
  const next = () => values.push(new Promise((a, b) => ([resolve, reject] = [a, b])));
  next();
  const callback = (value, done, error) => {
    if (error) return reject(error);
    resolve([value, done]);
    next();
  };
  run((chunk) => callback(chunk)).then(
    () => callback(undefined, true),
    (error) => callback(undefined, false, error)
  );
  return (async function* () {
    for (let i = 0, done = false; !done; i++) {
      let value;
      [value, done] = await values[i];
      if (value !== undefined) yield value;
    }
  })();
}

function fakeWllama(log) {
  class Wllama {
    constructor(paths, config) {
      log.paths = paths;
      log.config = config;
    }

    static getLibllamaVersion() {
      return 'b1-fake';
    }

    async loadModel(blobs, params) {
      log.params = params;
      const resources = this.getWorkerResources();
      resources.noWebGPU = true;
      log.resources = { ...resources, noWebGPU: resources.noWebGPU };
      log.heads = [];
      for (const blob of blobs)
        log.heads.push(Buffer.from(await blob.slice(0, 4).arrayBuffer()).toString());
      const url = URL.createObjectURL(new Blob([WORKER], { type: 'text/javascript' }));
      this.worker = new Worker(url, { type: 'module' });
      log.ready = await new Promise((resolve) => {
        this.worker.onmessage = (e) => resolve(e.data);
        this.worker.postMessage({ verb: 'init' });
      });
      this.context = { n_vocab: log.vocab ?? 3, n_layer: 2, n_ctx: params.n_ctx };
    }

    getLoadedContextInfo() {
      return this.context;
    }

    async createChatCompletion(params) {
      return fakeCompletion(log, params);
    }

    async createSystemOne(params) {
      log.systemone.push(params);
      return log.answers;
    }

    async exit() {
      log.exited = true;
      this.worker?.terminate();
    }
  }
  const url = pathToFileURL(join(tmpdir(), 'fake-wllama', 'index.js')).href;
  return async () => ({ module: { Wllama }, url });
}

function fakeGpu(log) {
  const device = {
    destroyed: false,
    onuncapturederror: () => {},
    destroy() {
      device.destroyed = true;
      device.lose();
    },
  };
  device.lost = new Promise((resolve) => (device.lose = resolve));
  log.device = device;
  const gpu = { requestAdapter: async () => ({ requestDevice: async () => device }) };
  return async (track) => {
    log.track = track;
    if (log.noGpu) throw new Error('no GPU in this test');
    return { gpu, globals: { GPUValidationError: class {} } };
  };
}

const chunk = (delta, extra = {}) => ({
  choices: [{ index: 0, delta, finish_reason: null }],
  ...extra,
});

const CHUNKS = [
  chunk({ role: 'assistant', content: '' }),
  chunk({ reasoning_content: 'Looking.' }),
  chunk({ content: 'A red ' }),
  chunk({ content: 'circle.' }),
  chunk({
    tool_calls: [{ index: 0, id: 'call_1', function: { name: 'shape', arguments: '{"a":' } }],
  }),
  chunk({ tool_calls: [{ index: 0, function: { arguments: '1}' } }] }),
  {
    choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
    timings: {
      cache_n: 2,
      prompt_n: 10,
      prompt_ms: 5,
      prompt_per_second: 2000,
      predicted_n: 6,
      predicted_ms: 3,
      predicted_per_second: 2000,
    },
  },
  { choices: [], usage: { prompt_tokens: 12, completion_tokens: 6, total_tokens: 18 } },
];

const RED = new URL('./fixtures/red-circle.png', import.meta.url);

describe('gguf engine plumbing on a fake wllama', () => {
  const log = { chat: [], systemone: [], chunks: CHUNKS };
  let server;
  let runtime;
  const exitCode = process.exitCode;
  const hadWorker = Object.hasOwn(globalThis, 'Worker');

  before(async () => {
    const cache = await ggufCache({
      'test/chat-GGUF': {
        'chat-Q4_K_M.gguf': ggufBytes(chatMeta()),
        'chat-Q8_0.gguf': ggufBytes(chatMeta()),
        'mmproj-chat-F16.gguf': ggufBytes(projectorMeta()),
        'README.md': 'not a model',
      },
      'test/decide-GGUF': { 'decide-Q8_0.gguf': ggufBytes(decisionMeta()) },
      'test/split-GGUF': { 'big-Q4_0-00001-of-00002.gguf': ggufBytes(chatMeta()) },
      'test/xn-GGUF': { 'model.q8.gguf': ggufBytes({ 'xn.tensor': 'x' }) },
    });
    const base = createCatalog({ cache, home: cache, config: { aliases: {} } });
    const library = fakeWllama(log);
    const gpus = fakeGpu(log);
    const catalog = {
      ...base,
      async resolve(id) {
        const resolved = await base.resolve(id);
        const { engine } = resolved;
        return {
          ...resolved,
          engine: { ...engine, load: (o) => engine.load({ ...o, library, gpus }) },
        };
      },
    };
    runtime = createRuntime({
      env: {},
      backend: async () => ({
        track: 'node',
        executionProviders: ['webgpu', 'cpu'],
        fallbackProviders: ['cpu'],
      }),
    });
    server = await startServer({ env: {}, port: 0, catalog, runtime });
  });
  after(() => server?.close());

  const post = (path, body) =>
    fetch(`${server.url}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });

  test('lists GGUF repos by quant, with the projector and decision models', async () => {
    const { data } = await (await fetch(`${server.url}/v1/models`)).json();
    const chat = data.find((m) => m.id === 'test/chat-GGUF');
    assert.equal(chat.ortllama.engine, 'gguf');
    assert.deepEqual(chat.ortllama.variants, ['Q4_K_M', 'Q8_0']);
    assert.equal(chat.ortllama.default_variant, 'Q4_K_M');
    assert.equal(chat.ortllama.mmproj, 'mmproj-chat-F16.gguf');
    assert.deepEqual(chat.architecture.input_modalities, ['text', 'image']);
    assert.equal(chat.context_length, 4096);
    const decide = data.find((m) => m.id === 'test/decide-GGUF');
    assert.equal(decide.ortllama.engine, 'gguf-decisions');
    assert.deepEqual(decide.architecture.output_modalities, ['decisions']);
    assert.ok(!data.some((m) => m.id === 'test/xn-GGUF'));
  });

  test('streams a chat with an image through the in-thread worker', async () => {
    const image = `data:image/png;base64,${(await readFile(RED)).toString('base64')}`;
    const response = await post('/v1/chat/completions', {
      model: 'test/chat-GGUF',
      stream: true,
      stream_options: { include_usage: true },
      temperature: 0,
      messages: [
        { role: 'developer', content: 'Be brief.' },
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: image } },
            { type: 'text', text: 'What is it?' },
          ],
        },
      ],
    });
    assert.equal(response.status, 200);
    const events = (await response.text())
      .split('\n\n')
      .filter((l) => l.startsWith('data: {'))
      .map((l) => JSON.parse(l.slice(6)));
    const deltas = events.flatMap((e) => e.choices.map((c) => c.delta));
    assert.equal(deltas.map((d) => d.content ?? '').join(''), 'A red circle.');
    assert.equal(deltas.find((d) => d.reasoning_content).reasoning_content, 'Looking.');
    const calls = deltas.flatMap((d) => d.tool_calls ?? []);
    assert.equal(calls[0].function.name, 'shape');
    assert.equal(calls.map((c) => c.function.arguments).join(''), '{"a":1}');
    assert.equal(
      events.find((e) => e.choices[0]?.finish_reason).choices[0].finish_reason,
      'tool_calls'
    );
    assert.deepEqual(events.at(-1).usage.prompt_tokens_details, { cached_tokens: 2 });
    const [sent] = log.chat;
    assert.equal(sent.messages[0].role, 'system');
    const [picture, text] = sent.messages[1].content;
    assert.equal(Buffer.from(picture.data).subarray(1, 4).toString(), 'PNG');
    assert.deepEqual(text, { type: 'text', text: 'What is it?' });
    assert.equal(sent.temperature, 0);
    assert.deepEqual(log.heads, ['GGUF', 'GGUF']);
    assert.equal(log.params.n_ctx, 4096);
    assert.equal(log.params.n_gpu_layers > 0, true);
    assert.equal(log.resources.noWebGPU, false);
    assert.deepEqual(log.ready, {
      verb: 'ready',
      gpu: true,
      dir: join(tmpdir(), 'fake-wllama', 'wasm'),
      waitAsync: 'undefined',
      os: 'function',
      error: 'function',
    });
    assert.equal(process.exitCode, exitCode);
    assert.equal(Object.hasOwn(globalThis, 'Worker'), hadWorker);
  });

  test('ps reports the GGUF model with its projector bytes', async () => {
    const { models } = await (await fetch(`${server.url}/ortllama/ps`)).json();
    assert.equal(models[0].engine, 'gguf');
    assert.deepEqual(models[0].executionProviders, ['webgpu', 'cpu']);
    assert.equal(
      models[0].modelBytes,
      ggufBytes(chatMeta()).length + ggufBytes(projectorMeta()).length
    );
  });

  test('answers System One on a decision GGUF and refuses it on a chat GGUF', async () => {
    log.answers = {
      answers: { billing: { type: 'noul', noul: 0.938833 } },
      usage: { input_tokens: 9, output_tokens: 0 },
    };
    const request = {
      state: 'Charged twice.',
      questions: { billing: { type: 'noul', instructions: 'Billing?' } },
    };
    const ok = await post('/v1/systemone', { model: 'test/decide-GGUF', ...request });
    const body = await ok.json();
    assert.equal(ok.status, 200, JSON.stringify(body));
    assert.deepEqual(body.answers, { billing: { type: 'noul', noul: 0.9388 } });
    assert.ok(log.device.destroyed, 'the chat model was unloaded and its device destroyed');
    assert.equal(log.device.onuncapturederror, null);
    const refused = await post('/v1/systemone', { model: 'test/chat-GGUF', ...request });
    assert.equal(refused.status, 400);
  });

  test('falls back to the CPU when WebGPU cannot start', async () => {
    log.noGpu = true;
    const response = await post('/v1/chat/completions', {
      model: 'test/chat-GGUF:Q8_0',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    assert.equal(response.status, 200);
    const { models } = await (await fetch(`${server.url}/ortllama/ps`)).json();
    assert.deepEqual(models[0].executionProviders, ['cpu']);
    assert.equal(log.params.n_gpu_layers, 0);
    log.noGpu = false;
  });

  test('answers 507 when the wasm heap runs out', async () => {
    log.chunks = [new Error('decode() failed: std::bad_alloc')];
    const response = await post('/v1/chat/completions', {
      model: 'test/chat-GGUF:Q8_0',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    const { error } = await response.json();
    assert.equal(response.status, 507);
    assert.equal(error.code, 'out_of_memory');
    assert.match(error.message, /4 GB heap/);
    log.chunks = CHUNKS;
  });

  test('names the missing split parts of a variant', async () => {
    const response = await post('/v1/chat/completions', {
      model: 'test/split-GGUF',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    const { error } = await response.json();
    assert.equal(response.status, 404);
    assert.equal(error.code, 'model_incomplete');
    assert.match(error.message, /big-Q4_0-00002-of-00002\.gguf/);
  });

  test('refuses a model wllama could not load', async () => {
    log.vocab = 0;
    const response = await post('/v1/chat/completions', {
      model: 'test/chat-GGUF:Q4_K_M',
      messages: [{ role: 'user', content: 'Hi' }],
    });
    assert.equal(response.status, 422);
    assert.equal((await response.json()).error.code, 'model_unsupported');
    log.vocab = 3;
    await runtime.close();
    assert.ok(log.exited);
  });
});

test('a stream aborted while it waits for its client fails cleanly, without an unhandled rejection', async () => {
  const log = {
    chat: [],
    delay: 50,
    chunks: Array.from({ length: 40 }, (_, i) => chunk({ content: `w${i} ` })),
  };
  const abort = new AbortController();
  const unhandled = [];
  const listener = (error) => unhandled.push(error);
  process.on('unhandledRejection', listener);
  let events = 0;
  try {
    await assert.rejects(
      streamChat(
        { createChatCompletion: (params) => fakeCompletion(log, params) },
        { messages: [{ role: 'user', content: 'Go' }], tools: [], stop: [] },
        {
          signal: abort.signal,
          emit() {
            if (++events === 4) abort.abort(new Error('the client left'));
          },
          yieldTurn: () => new Promise((resolve) => setTimeout(resolve, 250)),
        }
      ),
      /the client left/
    );
    await new Promise((resolve) => setTimeout(resolve, 100));
  } finally {
    process.off('unhandledRejection', listener);
  }
  assert.deepEqual(unhandled, []);
});

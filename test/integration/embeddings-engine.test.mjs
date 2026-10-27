import assert from 'node:assert/strict';
import { rm } from 'node:fs/promises';
import { after, before, describe, test } from 'node:test';
import { decoder } from '../../src/engines/decoder/index.js';
import { embedding } from '../../src/engines/embedding.js';
import { hubFiles } from '../../src/hf/files.js';
import { createRuntime, openCatalog, startServer } from '../../src/index.js';
import { fakeHome } from './cache.mjs';
import { addGraph, hubCache, REPO, snapshotOf, weights } from './hub.mjs';

const DIM = 768;

const CHARS = Array.from({ length: 94 }, (_, i) => [String.fromCharCode(33 + i), -2]);

const special = (id, content) => ({
  id,
  content,
  single_word: false,
  lstrip: false,
  rstrip: false,
  normalized: false,
  special: true,
});

const TOKENIZER = {
  added_tokens: [special(0, '<pad>'), special(1, '<eos>'), special(2, '<bos>')],
  normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '\u2581' },
  pre_tokenizer: null,
  decoder: null,
  post_processor: {
    type: 'TemplateProcessing',
    single: [
      { SpecialToken: { id: '<bos>', type_id: 0 } },
      { Sequence: { id: 'A', type_id: 0 } },
      { SpecialToken: { id: '<eos>', type_id: 0 } },
    ],
    pair: [],
    special_tokens: {
      '<bos>': { id: '<bos>', ids: [2], tokens: ['<bos>'] },
      '<eos>': { id: '<eos>', ids: [1], tokens: ['<eos>'] },
    },
  },
  model: {
    type: 'Unigram',
    unk_id: 3,
    vocab: [['<pad>', 0], ['<eos>', 0], ['<bos>', 0], ['<unk>', 0], ['\u2581', -1], ...CHARS],
  },
};

const CONFIG = {
  architectures: ['EmbeddingGemma2Model'],
  model_type: 'embedding_gemma2',
  text_config: { pad_token_id: 0 },
  'transformers.js_config': {
    use_external_data_format: {
      'model_quantized.onnx': 1,
      'model_q4.onnx': 1,
      'vision_encoder.onnx': 1,
      'audio_encoder.onnx': 1,
    },
  },
};

const TEXT = {
  'config.json': JSON.stringify(CONFIG),
  'tokenizer.json': JSON.stringify(TOKENIZER),
  'tokenizer_config.json': JSON.stringify({ bos_token: '<bos>', eos_token: '<eos>' }),
  'onnx/model_quantized.onnx': addGraph('model_quantized.onnx_data'),
  'onnx/model_quantized.onnx_data': weights,
};

const MEDIA = {
  'preprocessor_config.json': '{}',
  'processor_config.json': '{}',
  'chat_template.jinja': '{{ messages[0].content }}',
  'onnx/model_q4.onnx': addGraph('model_q4.onnx_data'),
  'onnx/vision_encoder.onnx': addGraph('vision_encoder.onnx_data'),
  'onnx/vision_encoder.onnx_data': weights,
  'onnx/audio_encoder.onnx': addGraph('audio_encoder.onnx_data'),
  'onnx/audio_encoder.onnx_data': weights,
};

function fakeOrt() {
  const runs = [];
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }

    async getData() {
      return this.data;
    }
  }
  const session = {
    inputNames: ['input_ids', 'attention_mask', 'image_features', 'audio_features'],
    outputNames: ['last_hidden_state', 'sentence_embedding'],
    inputMetadata: [
      { name: 'input_ids', type: 'int64', shape: ['batch_size', 'sequence_length'] },
      { name: 'attention_mask', type: 'int64', shape: ['batch_size', 'sequence_length'] },
      { name: 'image_features', type: 'float32', shape: ['num_image_tokens', 512] },
      { name: 'audio_features', type: 'float32', shape: ['num_audio_tokens', 512] },
    ],
    async run(feeds, fetches) {
      const [batch, width] = feeds.input_ids.dims;
      runs.push({
        batch,
        width,
        fetches,
        media: [feeds.image_features.dims, feeds.audio_features.dims],
      });
      const out = new Float32Array(batch * DIM);
      for (let b = 0; b < batch; b++) {
        for (let i = 0; i < width; i++) {
          if (feeds.attention_mask.data[b * width + i] === 0n) continue;
          const id = Number(feeds.input_ids.data[b * width + i]);
          for (let d = 0; d < DIM; d++) out[b * DIM + d] += Math.cos((id + 1) * (d + 1) * 0.01);
        }
        let norm = 0;
        for (let d = 0; d < DIM; d++) norm += out[b * DIM + d] ** 2;
        for (let d = 0; d < DIM; d++) out[b * DIM + d] /= Math.sqrt(norm);
      }
      return { sentence_embedding: new Tensor('float32', out, [batch, DIM]) };
    },
    release: async () => {},
  };
  const created = [];
  return {
    runs,
    created,
    ort: {
      Tensor,
      InferenceSession: {
        create: async (model) => {
          created.push(model);
          return session;
        },
      },
    },
  };
}

const made = [];
after(() => Promise.all(made.map((dir) => rm(dir, { recursive: true, force: true }))));

async function cacheOf(files) {
  const { cache } = await hubCache(1, files);
  made.push(cache);
  return cache;
}

const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

describe('the embedding engine finds EmbeddingGemma 2 exports', () => {
  test('a text-only download lists and the decoder leaves it alone', async () => {
    const cache = await cacheOf(TEXT);
    const files = hubFiles(cache, REPO, snapshotOf(cache));
    assert.equal(await decoder.detect(files), null);
    const catalog = await openCatalog({ cache, home: await fakeHome(), modelPath: [] });
    const { models, unrecognized } = await catalog.survey();
    assert.deepEqual(unrecognized, []);
    assert.equal(models.length, 1);
    assert.equal(models[0].engine, 'embedding');
    assert.deepEqual(models[0].variants, ['q8']);
    assert.deepEqual(models[0].outputModalities, ['embeddings']);
    assert.equal(models[0].contextLength, 8192);
    assert.deepEqual(models[0].details, { dimensions: [768, 512, 256, 128] });
  });

  test('a full download lists only the text graphs and fetches only them', async () => {
    const cache = await cacheOf({ ...TEXT, ...MEDIA });
    const files = hubFiles(cache, REPO, snapshotOf(cache));
    assert.equal(await decoder.detect(files), null);
    const info = await embedding.detect(files);
    assert.deepEqual(info.variants, ['q8', 'q4']);
    assert.equal(info.defaultVariant, 'q8');
    assert.deepEqual(embedding.include(info, 'q8'), ['*.json', 'onnx/model_quantized.onnx*']);
    const catalog = await openCatalog({ cache, home: await fakeHome(), modelPath: [] });
    const { unrecognized } = await catalog.survey();
    assert.deepEqual(unrecognized, []);
    await assert.rejects(catalog.resolve(`${REPO}:q4`), (error) => {
      assert.equal(error.code, 'model_incomplete');
      assert.match(error.message, /onnx\/model_q4\.onnx_data/);
      assert.match(
        error.message,
        new RegExp(`run: hf download ${REPO} --include '\\*\\.json' 'onnx/model_q4\\.onnx\\*'$`)
      );
      return true;
    });
  });
});

describe('the embedding engine on onnxruntime-web', () => {
  test('reads the external data into the session, and answers 422 for a missing kernel', async () => {
    const cache = await cacheOf({ ...TEXT, 'onnx/model.onnx': addGraph('model.onnx_data') });
    const files = hubFiles(cache, REPO, snapshotOf(cache));
    const info = await embedding.detect(files);
    const fake = fakeOrt();
    const options = [];
    const backend = (create) => ({
      track: 'web',
      executionProviders: ['wasm'],
      sessionOptions: {},
      ort: { ...fake.ort, InferenceSession: { create } },
    });
    const loaded = await embedding.load({
      backend: backend(async (model, o) => {
        options.push(o);
        return fake.ort.InferenceSession.create(model);
      }),
      files,
      variant: 'q8',
      info,
    });
    assert.deepEqual(
      options[0].externalData.map((d) => [d.path, d.data.length]),
      [['model_quantized.onnx_data', weights.length]]
    );
    assert.equal((await loaded.embed(['hi'])).vectors[0].length, DIM);
    const refused = embedding.load({
      backend: backend(async () => {
        throw new Error(
          "Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: Could not find an implementation for GatherBlockQuantized(1) node with name '/model/embed_tokens/Gather_Quant'"
        );
      }),
      files,
      variant: 'q8',
      info,
    });
    await assert.rejects(refused, (error) => {
      assert.equal(error.status, 422);
      assert.equal(error.code, 'model_unsupported');
      assert.equal(
        error.message,
        'variant q8 does not run on wasm (onnxruntime-web), which has no GatherBlockQuantized kernel; use variant fp32 here, or a backend with WebGPU'
      );
      return true;
    });
  });
});

describe('POST /v1/embeddings on a fake ORT', () => {
  let server;
  let fake;
  let post;

  before(async () => {
    const cache = await cacheOf({ ...TEXT, ...MEDIA });
    fake = fakeOrt();
    const home = await fakeHome({ embed: REPO });
    const env = { HF_HUB_CACHE: cache, ORTLLAMA_HOME: home, ORTLLAMA_MODEL_PATH: '' };
    const runtime = createRuntime({
      env,
      keepAlive: '5m',
      backend: async () => ({
        ort: fake.ort,
        track: 'node',
        executionProviders: ['cpu'],
        sessionOptions: {},
      }),
    });
    server = await startServer({ env, port: 0, runtime });
    post = async (path, body) => {
      const response = await fetch(`${server.url}${path}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      return { status: response.status, body: await response.json() };
    };
  });
  after(() => server?.close());

  test('lists the model with embeddings output and its dimensions', async () => {
    const response = await fetch(`${server.url}/v1/models`);
    const { data } = await response.json();
    const entry = data.find((m) => m.id === REPO);
    assert.deepEqual(entry.architecture.output_modalities, ['embeddings']);
    assert.deepEqual(entry.ortllama.dimensions, [768, 512, 256, 128]);
    assert.equal(entry.ortllama.engine, 'embedding');
    const alias = data.find((m) => m.id === 'embed');
    assert.equal(alias.ortllama.alias_of, REPO);
  });

  test('answers in OpenAI shape, batched, with usage', async () => {
    const input = ['a', 'a longer one', 'the longest input of the three'];
    const { status, body } = await post('/v1/embeddings', { input, user: 'someone' });
    assert.equal(status, 200, JSON.stringify(body));
    assert.equal(body.object, 'list');
    assert.equal(body.model, 'embed');
    assert.deepEqual(
      body.data.map((d) => [d.object, d.index, d.embedding.length]),
      [
        ['embedding', 0, DIM],
        ['embedding', 1, DIM],
        ['embedding', 2, DIM],
      ]
    );
    const tokens = input.reduce((n, s) => n + s.length + 2, 0);
    assert.deepEqual(body.usage, { prompt_tokens: tokens, total_tokens: tokens });
    const run = fake.runs.at(-1);
    assert.equal(run.batch, 3);
    assert.deepEqual(run.fetches, ['sentence_embedding']);
    assert.deepEqual(run.media, [
      [0, 512],
      [0, 512],
    ]);
    for (const [i, text] of input.entries()) {
      const single = await post('/v1/embeddings', { input: text });
      assert.ok(cosine(single.body.data[0].embedding, body.data[i].embedding) > 0.99999);
    }
    const [ps] = (await (await fetch(`${server.url}/ortllama/ps`)).json()).models;
    assert.equal(ps.engine, 'embedding');
    assert.equal(ps.id, REPO);
  });

  test('builds task prefixes, and leaves prefixed input alone', async () => {
    const text = 'Which planet is known as the Red Planet?';
    const raw = await post('/v1/embeddings', {
      input: [`task: search result | query: ${text}`, `title: Mars | text: ${text}`],
    });
    const built = await post('/v1/embeddings', { input: text, task: 'search result' });
    assert.ok(cosine(raw.body.data[0].embedding, built.body.data[0].embedding) > 0.99999);
    const titled = await post('/v1/embeddings', {
      input: [text, `task: search result | query: ${text}`],
      input_type: 'document',
      title: 'Mars',
    });
    assert.ok(cosine(raw.body.data[1].embedding, titled.body.data[0].embedding) > 0.99999);
    assert.ok(cosine(raw.body.data[0].embedding, titled.body.data[1].embedding) > 0.99999);
    const plain = await post('/v1/embeddings', { input: text });
    assert.equal(plain.body.usage.prompt_tokens, text.length + 2);
    assert.ok(cosine(plain.body.data[0].embedding, built.body.data[0].embedding) < 0.99999);
  });

  test('truncates to a Matryoshka size and normalizes again, in float or base64', async () => {
    const full = await post('/v1/embeddings', { input: 'hello there' });
    const short = await post('/v1/embeddings', { input: 'hello there', dimensions: 256 });
    const head = full.body.data[0].embedding.slice(0, 256);
    const expected = head.map((x) => x / norm(head));
    const got = short.body.data[0].embedding;
    assert.equal(got.length, 256);
    assert.ok(Math.abs(norm(got) - 1) < 1e-6);
    assert.ok(got.every((x, i) => Math.abs(x - expected[i]) < 1e-6));
    const coded = await post('/v1/embeddings', {
      input: 'hello there',
      dimensions: 256,
      encoding_format: 'base64',
    });
    const bytes = Buffer.from(coded.body.data[0].embedding, 'base64');
    assert.equal(bytes.length, 256 * 4);
    const floats = Array.from({ length: 256 }, (_, i) => bytes.readFloatLE(i * 4));
    assert.deepEqual(floats, Array.from(new Float32Array(got)));
  });

  test('answers bad requests in the OpenAI envelope', async () => {
    const cases = [
      [{ input: [] }, 'input', /non-empty array/],
      [{ input: [1, 2, 3] }, 'input', /token IDs is not supported/],
      [{ input: [[1, 2]] }, 'input', /token IDs/],
      [{ input: ['ok', 7] }, 'input', /input\[1\] must be a string/],
      [{ input: ['ok', ''] }, 'input', /input\[1\] is empty/],
      [{ input: Array(2049).fill('x') }, 'input', /at most 2048/],
      [{ input: 'x', dimensions: 300 }, 'dimensions', /768, 512, 256, 128/],
      [{ input: 'x', dimensions: '256' }, 'dimensions', /768, 512, 256, 128/],
      [{ input: 'x', encoding_format: 'int8' }, 'encoding_format', /float or base64/],
      [{ input: 'x', task: 'search' }, 'task', /search result, question answering/],
      [{ input: 'x', input_type: 'passage' }, 'input_type', /query or document/],
      [{ input: 'x', title: 'T' }, 'title', /only to input_type document/],
      [{ input: 'x', input_type: 'document', task: 'clustering', title: 'T' }, 'title', /only/],
      [{ input: ['x', 'y'], input_type: 'document', title: ['T'] }, 'title', /for each input/],
    ];
    for (const [body, param, message] of cases) {
      const { status, body: error } = await post('/v1/embeddings', body);
      assert.equal(status, 400, JSON.stringify(body).slice(0, 80));
      assert.equal(error.error.type, 'invalid_request_error');
      assert.equal(error.error.param, param);
      assert.match(error.error.message, message);
    }
    const notJson = await post('/v1/embeddings', [1]);
    assert.equal(notJson.status, 400);
    assert.match(notJson.body.error.message, /JSON object/);
  });

  test('refuses inputs over the context window, never truncating them', async () => {
    const { status, body } = await post('/v1/embeddings', { input: ['short', 'x'.repeat(8191)] });
    assert.equal(status, 400);
    assert.equal(body.error.code, 'context_length_exceeded');
    assert.equal(body.error.param, 'input');
    assert.match(body.error.message, /^input\[1\] has 8193 tokens, which exceeds .* 8192 tokens/);
  });

  test('names the route a model answers on the wrong one', async () => {
    const chat = await post('/v1/chat/completions', {
      model: 'embed',
      messages: [{ role: 'user', content: 'hi' }],
    });
    assert.equal(chat.status, 400);
    assert.match(
      chat.body.error.message,
      new RegExp(`^model ${REPO} does not answer chat completions; it answers POST /v1/embeddings$`)
    );
    const speech = await post('/v1/audio/speech', { model: 'embed', input: 'hi' });
    assert.equal(speech.status, 400);
    assert.match(
      speech.body.error.message,
      /does not synthesize speech; it answers POST \/v1\/embeddings$/
    );
  });

  test('loads the text graph alone', async () => {
    assert.ok(fake.created.length > 0);
    for (const model of fake.created) assert.match(String(model), /model_quantized\.onnx$/);
  });
});

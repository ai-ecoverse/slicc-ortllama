import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { after, before, describe, test } from 'node:test';
import { promisify } from 'node:util';
import { embedding } from '../../src/engines/embedding.js';
import { createRuntime, openCatalog, startServer } from '../../src/index.js';

const run = promisify(execFile);
const MODEL = 'onnx-community/embeddinggemma-2-ONNX';
const HIDE = new URL('./hide-ort-node.mjs', import.meta.url).pathname;
const INDEX = new URL('../../src/index.js', import.meta.url).href;
const FIXTURES = new URL(`../../fixtures/${MODEL}.json`, import.meta.url);
const env = { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_MODEL_PATH: '' };

async function cached(id) {
  try {
    return await (await openCatalog({ env })).resolve(id);
  } catch {
    return null;
  }
}

const resolved = await cached(MODEL);
const skip =
  !resolved && !process.env.ORTLLAMA_REQUIRE_MODEL && `${MODEL} is not in the Hugging Face cache`;

const QUERY = 'Which planet is known as the Red Planet?';
const PLANETS = [
  "Venus is often called Earth's twin because of its similar size and proximity.",
  'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
  'Jupiter, the largest planet in our solar system, has a prominent red spot.',
  'Saturn, famous for its rings, is sometimes mistaken for the Red Planet.',
];

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

describe('embeddings on EmbeddingGemma 2', { skip }, () => {
  let server;
  const post = async (body) => {
    const response = await fetch(`${server.url}/v1/embeddings`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: MODEL, ...body }),
    });
    return { status: response.status, body: await response.json() };
  };

  before(async () => {
    server = await startServer({ env, port: 0 });
  });
  after(() => server?.close());

  test('ranks Mars first for the Red Planet, at 768 and 128 dimensions', async () => {
    for (const dimensions of [768, 128]) {
      const query = await post({ input: QUERY, task: 'search result', dimensions });
      const docs = await post({ input: PLANETS, input_type: 'document', dimensions });
      assert.equal(query.status, 200, JSON.stringify(query.body));
      const q = query.body.data[0].embedding;
      assert.equal(q.length, dimensions);
      const scores = docs.body.data.map((d) => dot(q, d.embedding));
      assert.equal(scores.indexOf(Math.max(...scores)), 1, scores.join(', '));
    }
  });

  test('embeds hand-built prefixes as the extension fields build them', async () => {
    const raw = await post({
      input: [`task: code retrieval | query: ${QUERY}`, `title: planets.md | text: ${PLANETS[1]}`],
    });
    const query = await post({ input: QUERY, task: 'code retrieval', input_type: 'query' });
    const doc = await post({ input: PLANETS[1], input_type: 'document', title: 'planets.md' });
    assert.ok(dot(raw.body.data[0].embedding, query.body.data[0].embedding) > 0.9999);
    assert.ok(dot(raw.body.data[1].embedding, doc.body.data[0].embedding) > 0.9999);
    assert.equal(
      raw.body.usage.prompt_tokens,
      query.body.usage.prompt_tokens + doc.body.usage.prompt_tokens
    );
  });

  test('base64 holds the same float32 vector', async () => {
    const floats = await post({ input: QUERY, dimensions: 256 });
    const coded = await post({ input: QUERY, dimensions: 256, encoding_format: 'base64' });
    const bytes = Buffer.from(coded.body.data[0].embedding, 'base64');
    const decoded = Array.from({ length: 256 }, (_, i) => bytes.readFloatLE(i * 4));
    assert.deepEqual(decoded, Array.from(new Float32Array(floats.body.data[0].embedding)));
  });

  test('refuses an input over 8192 tokens', async () => {
    const { status, body } = await post({ input: 'lighthouse keeper '.repeat(5000) });
    assert.equal(status, 400);
    assert.equal(body.error.code, 'context_length_exceeded');
  });

  test('matches the sentence-transformers fixtures', async () => {
    const { fixtures } = JSON.parse(await readFile(FIXTURES, 'utf8'));
    const runtime = createRuntime({ env, keepAlive: '0' });
    const entry = await runtime.acquire(resolved);
    const report = await embedding.check(entry.instance, {
      fixtures,
      variant: resolved.variant,
    });
    await runtime.close();
    assert.ok(report.ok, report.summary);
    assert.equal(report.moved.length, 0);
  });
});

const LOAD = `
import { createRuntime, openCatalog } from ${JSON.stringify(INDEX)};
const runtime = createRuntime({ keepAlive: '0' });
const resolved = await (await openCatalog()).resolve(process.argv[1]);
try {
  const entry = await runtime.acquire(resolved);
  const { vectors } = await entry.instance.embed(['hello']);
  console.log(JSON.stringify({ backend: entry.backend, length: vectors[0].length }));
} catch (error) {
  console.log(JSON.stringify({ status: error.status, code: error.code, message: error.message }));
}
await runtime.close();
`;

async function hidden(id) {
  const { stdout } = await run(
    process.execPath,
    ['--import', HIDE, '--input-type=module', '-e', LOAD, id],
    { env: { ...env, ORTLLAMA_EP: '' } }
  );
  return JSON.parse(stdout);
}

const quantized = resolved?.info.variants.find((v) => v.startsWith('q'));
const fp32 = await cached(`${MODEL}:fp32`);

describe('EmbeddingGemma 2 without onnxruntime-node', { skip }, () => {
  test('a quantized variant answers 422 on WASM', {
    skip: !quantized && 'no quantized variant is cached',
  }, async () => {
    const report = await hidden(`${MODEL}:${quantized}`);
    assert.equal(report.status, 422, JSON.stringify(report));
    assert.equal(report.code, 'model_unsupported');
    assert.match(report.message, /no GatherBlockQuantized kernel/);
  });

  test('fp32 runs on WASM', { skip: !fp32 && 'fp32 is not cached' }, async () => {
    assert.deepEqual(await hidden(`${MODEL}:fp32`), { backend: 'onnxruntime-web', length: 768 });
  });
});

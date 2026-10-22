import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { main } from '../../src/cli.js';
import { openCatalog, startServer } from '../../src/index.js';

const REPO = 'bartowski/SmolLM2-135M-Instruct-GGUF';
const REVISION = '09816acd5d99df7be770d85ea30822623dab342c';
const FILE = 'SmolLM2-135M-Instruct-Q4_0.gguf';
const ETAG = 'bcc3af2849ad6095af57e9b5cd43775256efdc66e306acb529172f92d0c04b03';
const MODEL = `${REPO}@${REVISION}`;
const JSPI = '--experimental-wasm-jspi';
const VISION = 'mistralai/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M';
const DECISIONS = 'ggml-org/Laya-GGUF:Q8_0';
const RED = new URL('./fixtures/red-circle.png', import.meta.url);

const jspi = typeof WebAssembly.Suspending === 'function';

async function cached(env, model = MODEL) {
  try {
    await (await openCatalog({ env: { ...env, ORTLLAMA_MODEL_PATH: '' } })).resolve(model);
    return true;
  } catch {
    return false;
  }
}

async function download() {
  const root = await mkdtemp(join(tmpdir(), 'ortllama-smollm-'));
  const dir = join(root, `models--${REPO.replace('/', '--')}`);
  const response = await fetch(`https://huggingface.co/${REPO}/resolve/${REVISION}/${FILE}`);
  assert.equal(response.status, 200, `downloading ${FILE}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  assert.equal(createHash('sha256').update(bytes).digest('hex'), ETAG);
  const blob = join(dir, 'blobs', ETAG);
  const link = join(dir, 'snapshots', REVISION, FILE);
  await mkdir(dirname(blob), { recursive: true });
  await mkdir(dirname(link), { recursive: true });
  await writeFile(blob, bytes);
  await mkdir(join(dir, 'refs'), { recursive: true });
  await writeFile(join(dir, 'refs', 'main'), REVISION);
  await symlink(relative(dirname(link), blob), link);
  return root;
}

if (!jspi && !process.execArgv.includes(JSPI)) {
  test(`GGUF tests on wllama run under ${JSPI}`, () => {
    const run = spawnSync(
      process.execPath,
      [JSPI, '--test', '--test-timeout=600000', fileURLToPath(import.meta.url)],
      { stdio: 'inherit', env: process.env }
    );
    assert.equal(run.status, 0);
  });
} else {
  let env = { ...process.env, ORTLLAMA_EP: process.env.ORTLLAMA_EP ?? 'cpu', ORTLLAMA_API_KEY: '' };
  let fetched = null;
  if (!(await cached(env)) && process.env.CI) {
    fetched = await download();
    env = { ...env, HF_HUB_CACHE: fetched };
  }
  const have = await cached(env);
  const skip =
    !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${MODEL} is not in the Hugging Face cache`;

  describe('chat on a GGUF model through wllama', { skip }, () => {
    let server;

    before(async () => {
      server = await startServer({ env, port: 0 });
    });
    after(async () => {
      await server?.close();
      if (fetched) await rm(fetched, { recursive: true, force: true });
    });

    const chat = (body) =>
      fetch(`${server.url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: MODEL, temperature: 0, max_tokens: 16, ...body }),
      });

    test('lists the cached quant', async () => {
      const { data } = await (await fetch(`${server.url}/v1/models`)).json();
      const m = data.find((x) => x.id === REPO);
      assert.equal(m?.ortllama.engine, 'gguf');
      assert.deepEqual(m.ortllama.variants, ['Q4_0']);
      assert.equal(m.ortllama.architecture, 'llama');
    });

    test('answers a chat completion', async () => {
      const response = await chat({
        messages: [{ role: 'user', content: 'What is the capital of France?' }],
      });
      const body = await response.json();
      assert.equal(response.status, 200, JSON.stringify(body));
      assert.match(body.choices[0].message.content, /Paris/);
      assert.ok(body.usage.prompt_tokens > 5);
      assert.equal(body.usage.completion_tokens, body.timings.predicted_n);
    });

    test('streams the tokens one chunk at a time', async () => {
      const response = await chat({
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: 12,
        messages: [{ role: 'user', content: 'Count from one to ten.' }],
      });
      const events = (await response.text())
        .split('\n\n')
        .filter((l) => l.startsWith('data: {'))
        .map((l) => JSON.parse(l.slice(6)));
      const pieces = events.flatMap((e) => e.choices.map((c) => c.delta.content)).filter(Boolean);
      assert.ok(pieces.length >= 6, `${pieces.length} content chunks`);
      const { usage } = events.at(-1);
      assert.ok(usage.completion_tokens >= pieces.length && usage.completion_tokens <= 12);
    });

    test('reports the loaded model', async () => {
      const { models } = await (await fetch(`${server.url}/ortllama/ps`)).json();
      assert.equal(models[0].engine, 'gguf');
      assert.equal(models[0].variant, 'Q4_0');
    });

    test('passes its check fixtures', async () => {
      let out = '';
      const status = await main(['check', MODEL, '--json'], {
        env,
        out: { write: (s) => (out += s) },
        err: { write: () => {} },
      });
      const report = JSON.parse(out);
      assert.equal(status, 0, report.summary);
      assert.ok(report.tokens > 0);
    });
  });

  const skipFor = async (model) =>
    !(await cached(env, model)) &&
    !process.env.ORTLLAMA_REQUIRE_MODEL &&
    `${model} is not in the Hugging Face cache`;

  const skipVision = await skipFor(VISION);
  const skipDecisions = await skipFor(DECISIONS);

  describe('vision and decisions on GGUF models', { skip: skipVision && skipDecisions }, () => {
    let server;
    before(async () => {
      server = await startServer({
        env: { ...env, ORTLLAMA_EP: process.env.ORTLLAMA_EP ?? '' },
        port: 0,
      });
    });
    after(() => server?.close());

    const post = (path, body) =>
      fetch(`${server.url}${path}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });

    test('Ministral 3 sees the red circle through its mmproj', { skip: skipVision }, async () => {
      const url = `data:image/png;base64,${(await readFile(RED)).toString('base64')}`;
      const response = await post('/v1/chat/completions', {
        model: VISION,
        temperature: 0,
        max_tokens: 32,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image_url', image_url: { url } },
              { type: 'text', text: 'What shape and color is in this picture? One sentence.' },
            ],
          },
        ],
      });
      const body = await response.json();
      assert.equal(response.status, 200, JSON.stringify(body));
      assert.match(body.choices[0].message.content, /red/i);
      assert.match(body.choices[0].message.content, /circle/i);
    });

    test('Laya answers System One', { skip: skipDecisions }, async () => {
      const response = await post('/v1/systemone', {
        model: DECISIONS,
        state: 'I was charged twice. Please fix this ASAP.',
        questions: {
          billing: { type: 'noul', instructions: 'Is this ticket about billing?' },
          urgency: {
            type: 'score',
            instructions: 'How urgent is this ticket?',
            criteria: ['can wait', 'this week', 'today'],
          },
        },
      });
      const body = await response.json();
      assert.equal(response.status, 200, JSON.stringify(body));
      assert.ok(body.answers.billing.noul > 0.8);
      assert.ok(body.answers.urgency.score > 1.5);
      assert.ok(body.usage.input_tokens > 0);
    });
  });
}

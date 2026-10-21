import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, describe, test } from 'node:test';
import { openCatalog, startServer } from '../../src/index.js';

const MODEL = 'ai-ecoverse/kev-ministral/kev-ministral-3b';
const BASE = `${MODEL}:base`;
const png = readFileSync(new URL('./fixtures/red-circle.png', import.meta.url)).toString('base64');

async function cached() {
  try {
    const catalog = await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } });
    return (await catalog.resolve(BASE)).info.inputModalities.includes('image');
  } catch {
    return false;
  }
}

const skip =
  !(await cached()) &&
  !process.env.ORTLLAMA_REQUIRE_MODEL &&
  `${MODEL} is not in the Hugging Face cache`;

const QUESTION = {
  state: 'My card was charged twice for one order.',
  questions: { billing: { type: 'noul', instructions: 'Is this a billing issue?' } },
};

let server;
const post = (path, body) =>
  fetch(`${server.url}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
const json = async (path, body) => {
  const response = await post(path, body);
  return { status: response.status, body: await response.json() };
};

async function streamed(body) {
  const response = await post('/v1/chat/completions', { ...body, stream: true });
  assert.equal(response.status, 200);
  const chunks = [];
  const decoder = new TextDecoder();
  let buffer = '';
  for await (const part of response.body) {
    buffer += decoder.decode(part, { stream: true });
    for (let i = buffer.indexOf('\n\n'); i >= 0; i = buffer.indexOf('\n\n')) {
      const data = buffer.slice(6, i);
      buffer = buffer.slice(i + 2);
      if (data !== '[DONE]') chunks.push({ at: performance.now(), chunk: JSON.parse(data) });
    }
  }
  return chunks;
}

describe('kev-ministral base weights on /v1/chat/completions', { skip }, () => {
  before(async () => {
    server = await startServer({
      env: { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_KEEP_ALIVE: '-1' },
      port: 0,
    });
  });
  after(() => server?.close());

  test('is listed as a text model next to the System One one', async () => {
    const { data } = await (await fetch(`${server.url}/v1/models`)).json();
    const base = data.find((m) => m.id === BASE);
    assert.ok(base, JSON.stringify(data.map((m) => m.id)));
    assert.deepEqual(base.architecture.input_modalities, ['text', 'image']);
    assert.deepEqual(base.architecture.output_modalities, ['text']);
    const decisions = data.find((m) => m.id === MODEL);
    assert.deepEqual(decisions.architecture.output_modalities, ['decisions']);
  });

  test('answers a question and stops at the next turn', async () => {
    const { status, body } = await json('/v1/chat/completions', {
      model: BASE,
      messages: [{ role: 'user', content: 'What is the capital of Australia?' }],
      max_tokens: 48,
    });
    assert.equal(status, 200, JSON.stringify(body));
    const { content } = body.choices[0].message;
    assert.match(content, /Canberra/);
    assert.doesNotMatch(content, /User:/);
    assert.equal(body.choices[0].finish_reason, 'stop');
    assert.ok(body.usage.completion_tokens < 48);
  });

  test('describes an image', async () => {
    const { status, body } = await json('/v1/chat/completions', {
      model: BASE,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: `data:image/png;base64,${png}` } },
            { type: 'text', text: 'What shape and color is in this image?' },
          ],
        },
      ],
      max_tokens: 40,
    });
    assert.equal(status, 200, JSON.stringify(body));
    assert.match(body.choices[0].message.content, /red/i);
    assert.match(body.choices[0].message.content, /circle/i);
    assert.ok(body.usage.prompt_tokens > 54, JSON.stringify(body.usage));
  });

  test('streams token by token and answers System One from the same load meanwhile', async () => {
    const stream = streamed({
      model: BASE,
      messages: [{ role: 'user', content: 'Count from one to thirty in words.' }],
      max_tokens: 40,
      stream_options: { include_usage: true },
    });
    const decision = json('/v1/systemone', { model: MODEL, ...QUESTION });
    const chunks = await stream;
    const { status, body } = await decision;
    assert.equal(status, 200, JSON.stringify(body));
    assert.ok(body.answers.billing.noul > 0.5, JSON.stringify(body.answers));
    const content = chunks.filter((c) => c.chunk.choices[0]?.delta.content);
    assert.ok(content.length > 10, `${content.length} content chunks`);
    const span = content.at(-1).at - content[0].at;
    assert.ok(span > 200, `the content chunks arrived within ${Math.round(span)} ms`);
    assert.equal(chunks.at(-1).chunk.usage.completion_tokens, 40);
    const { models } = await (await fetch(`${server.url}/ortllama/ps`)).json();
    assert.equal(models.filter((m) => m.id === MODEL).length, 1, JSON.stringify(models));
  });

  test('refuses tools and a second image', async () => {
    const tools = await json('/v1/chat/completions', {
      model: BASE,
      messages: [{ role: 'user', content: 'Weather?' }],
      tools: [{ type: 'function', function: { name: 'weather' } }],
    });
    assert.equal(tools.status, 400, JSON.stringify(tools.body));
    const url = `data:image/png;base64,${png}`;
    const two = await json('/v1/chat/completions', {
      model: BASE,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url } },
            { type: 'image_url', image_url: { url } },
          ],
        },
      ],
    });
    assert.equal(two.status, 400, JSON.stringify(two.body));
  });

  test('keeps decisions and generation on their own ids', async () => {
    const chat = await json('/v1/chat/completions', {
      model: MODEL,
      messages: [{ role: 'user', content: 'Hi' }],
    });
    assert.equal(chat.status, 400);
    assert.match(chat.body.error.message, /:base does/);
    const decide = await json('/v1/systemone', { model: BASE, ...QUESTION });
    assert.equal(decide.status, 400);
  });
});

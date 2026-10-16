import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, describe, test } from 'node:test';
import { stream } from '@earendil-works/pi-ai/api/openai-completions';
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript';
import { openCatalog, startServer } from '../../src/index.js';

const CHAT = process.env.ORTLLAMA_CHAT_MODEL ?? 'mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16';
const KEV = 'ai-ecoverse/kev.js/kev-0.8b-vision';
const png = readFileSync(new URL('./fixtures/red-circle.png', import.meta.url)).toString('base64');

async function takesImages(id) {
  try {
    const catalog = await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } });
    return (await catalog.resolve(id)).info.inputModalities.includes('image');
  } catch {
    return false;
  }
}

const require = process.env.ORTLLAMA_REQUIRE_MODEL;
const skipChat =
  !(await takesImages(CHAT)) &&
  !require &&
  `${CHAT} with its vision encoder is not in the Hugging Face cache`;
const skipKev = !(await takesImages(KEV)) && !require && `${KEV} is not in the Hugging Face cache`;

let server;
before(async () => {
  server = await startServer({
    env: {
      ...process.env,
      ORTLLAMA_API_KEY: '',
      ORTLLAMA_MAX_LOADED: '2',
      ORTLLAMA_KEEP_ALIVE: '-1',
    },
    port: 0,
  });
});
after(() => server?.close());

describe('image input in chat through pi-ai', { skip: skipChat }, () => {
  const model = () => ({
    id: CHAT,
    name: 'Ministral 3 3B',
    api: 'openai-completions',
    provider: 'ortllama',
    baseUrl: `${server.url}/v1`,
    reasoning: false,
    input: ['text', 'image'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 32768,
    maxTokens: 64,
  });
  const run = (context) =>
    stream(model(), normalizeContext(context), { apiKey: 'ortllama', temperature: 0 }).result();
  const text = (message) =>
    message.content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('');

  test('describes the shape and color, then reuses the image from the prefix cache', async () => {
    const context = {
      systemPrompt: 'Answer in one short sentence.',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What shape is in this image, and what color is it?' },
            { type: 'image', data: png, mimeType: 'image/png' },
          ],
          timestamp: Date.now(),
        },
      ],
    };
    const first = await run(context);
    assert.equal(first.stopReason, 'stop', first.errorMessage);
    assert.match(text(first), /red/i);
    assert.match(text(first), /circle/i);
    context.messages.push(first, {
      role: 'user',
      content: 'What color is the background?',
      timestamp: Date.now(),
    });
    const second = await run(context);
    assert.equal(second.stopReason, 'stop', second.errorMessage);
    assert.match(text(second), /white/i);
    assert.ok(second.usage.cacheRead >= first.usage.input, JSON.stringify(second.usage));
  });

  test('refuses image URLs it would have to fetch', async () => {
    const response = await fetch(`${server.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: CHAT,
        messages: [
          {
            role: 'user',
            content: [{ type: 'image_url', image_url: { url: 'https://example.com/a.png' } }],
          },
        ],
      }),
    });
    assert.equal(response.status, 400);
    assert.match((await response.json()).error.message, /does not fetch image URLs/);
  });
});

describe('image input on System One with a kev.js vision bundle', { skip: skipKev }, () => {
  test('answers about the image', async () => {
    const response = await fetch(`${server.url}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: KEV,
        image: `data:image/png;base64,${png}`,
        state: 'Look at the attached image.',
        questions: {
          color: {
            type: 'choice',
            instructions: 'What is the main color of the image?',
            criteria: { red: 'Red', green: 'Green', blue: 'Blue' },
          },
        },
      }),
    });
    const body = await response.json();
    assert.equal(response.status, 200, JSON.stringify(body));
    assert.equal(body.answers.color.choice, 'red');
  });
});

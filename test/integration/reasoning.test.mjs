import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { stream } from '@earendil-works/pi-ai/api/openai-completions';
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript';
import { openCatalog, startServer } from '../../src/index.js';

const MODEL = process.env.ORTLLAMA_REASONING_MODEL ?? 'chat-reasoning';

async function cached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(MODEL);
    return true;
  } catch {
    return false;
  }
}

const skip =
  !(await cached()) &&
  !process.env.ORTLLAMA_REQUIRE_MODEL &&
  `${MODEL} is not in the Hugging Face cache`;

const weather = {
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  parameters: {
    type: 'object',
    properties: { city: { type: 'string', description: 'City name' } },
    required: ['city'],
  },
};

const textOf = (message) =>
  message.content
    .filter((b) => b.type === 'text')
    .map((b) => b.text)
    .join('');

describe('chat completions on Ministral 3 Reasoning through pi-ai', { skip }, () => {
  let server;
  let model;
  const run = async (context, options = {}) => {
    const events = [];
    const s = stream(model, normalizeContext(context), {
      apiKey: 'ortllama',
      temperature: 0,
      maxTokens: 2048,
      reasoningEffort: 'medium',
      ...options,
    });
    for await (const event of s) events.push(event.type);
    return { message: await s.result(), events };
  };

  before(async () => {
    server = await startServer({
      env: { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_KEEP_ALIVE: '-1' },
      port: 0,
    });
    model = {
      id: MODEL,
      name: 'Ministral 3 3B Reasoning',
      api: 'openai-completions',
      provider: 'ortllama',
      baseUrl: `${server.url}/v1`,
      reasoning: true,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 32768,
      maxTokens: 2048,
    };
  });
  after(() => server?.close());

  test('streams its reasoning before the answer', async () => {
    const { message, events } = await run({
      messages: [
        {
          role: 'user',
          content: 'What is 17 multiplied by 23? Give just the number.',
          timestamp: Date.now(),
        },
      ],
    });
    assert.equal(message.stopReason, 'stop', message.errorMessage);
    const thinking = message.content.find((b) => b.type === 'thinking');
    assert.ok(thinking?.thinking.length > 20, JSON.stringify(message.content));
    assert.ok(events.filter((e) => e === 'thinking_delta').length > 5, events.join(' '));
    assert.ok(events.indexOf('thinking_start') < events.indexOf('text_start'), events.join(' '));
    assert.match(textOf(message), /391/);
    assert.ok(message.usage.output > 0);
  });

  test('thinks, calls a tool, and answers from its result', async () => {
    const context = {
      tools: [weather],
      messages: [
        {
          role: 'user',
          content: 'Use the get_weather tool to look up the current weather in Paris.',
          timestamp: Date.now(),
        },
      ],
    };
    const first = await run(context);
    assert.equal(first.message.stopReason, 'toolUse', first.message.errorMessage);
    const call = first.message.content.find((b) => b.type === 'toolCall');
    assert.equal(call.name, 'get_weather');
    assert.match(call.arguments.city, /Paris/);
    context.messages.push(first.message, {
      role: 'toolResult',
      toolCallId: call.id,
      toolName: call.name,
      content: [{ type: 'text', text: '{"temperature_c": 21, "sky": "sunny"}' }],
      isError: false,
      timestamp: Date.now(),
    });
    const second = await run(context);
    assert.equal(second.message.stopReason, 'stop', second.message.errorMessage);
    assert.match(textOf(second.message), /21|sunny/i);
    assert.ok(
      second.message.usage.cacheRead > first.message.usage.input / 2,
      JSON.stringify(second.message.usage)
    );
  });

  test('answers without reasoning when reasoning_effort is none', async () => {
    const response = await fetch(`${server.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        reasoning_effort: 'none',
        temperature: 0,
        max_tokens: 64,
        messages: [{ role: 'user', content: 'What is the capital of France? One word.' }],
      }),
    });
    assert.equal(response.status, 200);
    const { choices } = await response.json();
    assert.equal(choices[0].message.reasoning_content, undefined);
    assert.match(choices[0].message.content, /Paris/);
  });
});

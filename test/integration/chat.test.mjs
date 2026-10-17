import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { stream } from '@earendil-works/pi-ai/api/openai-completions';
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript';
import { openCatalog, startServer } from '../../src/index.js';

const MODEL =
  process.env.ORTLLAMA_CHAT_MODEL ?? 'mistralai/Ministral-3-3B-Instruct-2512-ONNX:q4f16';

async function cached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(MODEL);
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${MODEL} is not in the Hugging Face cache`;

const weather = {
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  parameters: {
    type: 'object',
    properties: { city: { type: 'string', description: 'City name' } },
    required: ['city'],
  },
};

describe('chat completions on Ministral 3 through pi-ai', { skip }, () => {
  let server;
  let model;
  const run = (context, options = {}) =>
    stream(model, normalizeContext(context), {
      apiKey: 'ortllama',
      temperature: 0,
      ...options,
    }).result();

  before(async () => {
    server = await startServer({
      env: { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_KEEP_ALIVE: '-1' },
      port: 0,
    });
    model = {
      id: MODEL,
      name: 'Ministral 3 3B',
      api: 'openai-completions',
      provider: 'ortllama',
      baseUrl: `${server.url}/v1`,
      reasoning: false,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 32768,
      maxTokens: 256,
    };
  });
  after(() => server?.close());

  test('streams plain text with usage', async () => {
    const message = await run(
      {
        systemPrompt: 'You are terse.',
        messages: [
          {
            role: 'user',
            content: 'What is the capital of France? One word.',
            timestamp: Date.now(),
          },
        ],
      },
      { maxTokens: 16 }
    );
    assert.equal(message.stopReason, 'stop', message.errorMessage);
    const text = message.content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('');
    assert.match(text, /Paris/);
    assert.ok(message.usage.input > 0);
    assert.ok(message.usage.output > 0);
  });

  test('round-trips a tool call and reuses the prefix cache', async () => {
    const context = {
      systemPrompt: 'Use tools when they help. Answer in one sentence.',
      tools: [weather],
      messages: [
        { role: 'user', content: 'What is the weather in Paris right now?', timestamp: Date.now() },
      ],
    };
    const first = await run(context);
    assert.equal(first.stopReason, 'toolUse', first.errorMessage);
    const call = first.content.find((b) => b.type === 'toolCall');
    assert.equal(call.name, 'get_weather');
    assert.match(call.arguments.city, /Paris/);
    context.messages.push(first, {
      role: 'toolResult',
      toolCallId: call.id,
      toolName: call.name,
      content: [{ type: 'text', text: '{"temperature_c": 21, "sky": "sunny"}' }],
      isError: false,
      timestamp: Date.now(),
    });
    const second = await run(context);
    assert.equal(second.stopReason, 'stop', second.errorMessage);
    const text = second.content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('');
    assert.match(text, /21|sunny/i);
    assert.ok(second.usage.cacheRead > first.usage.input / 2, JSON.stringify(second.usage));
  });

  test('refuses a prompt over the context window so pi can compact', async () => {
    const response = await fetch(`${server.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: 'user', content: 'word '.repeat(40000) }],
      }),
    });
    assert.equal(response.status, 400);
    const { error } = await response.json();
    assert.equal(error.code, 'context_length_exceeded');
    assert.match(error.message, /exceeds the context window/);
  });

  test('stops a request without a limit at ORTLLAMA_MAX_TOKENS', async () => {
    const capped = await startServer({
      env: { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_MAX_TOKENS: '12' },
      port: 0,
      runtime: server.runtime,
    });
    try {
      const response = await fetch(`${capped.url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: MODEL,
          messages: [{ role: 'user', content: 'Count from 1 to 300, separated by commas.' }],
        }),
      });
      const body = await response.json();
      assert.equal(body.choices[0].finish_reason, 'length', JSON.stringify(body));
      assert.equal(body.usage.completion_tokens, 12);
      assert.equal(body.generation_settings.max_tokens, 12);
      assert.equal(body.generation_settings.temperature, 0.1);
    } finally {
      await new Promise((resolve) => capped.server.close(resolve));
    }
  });
});

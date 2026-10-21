import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { chatRoute } from '../../src/chat.js';
import { createRuntime, PRIORITY, startServer } from '../../src/index.js';

const TOKENS = ['One', ' two', ' three', ' four', ' five'];
const SPIN_MS = 150;

function spin(ms) {
  const until = performance.now() + ms;
  while (performance.now() < until);
}

function blockingEngine(timeline) {
  const instance = {
    async chat(_request, { emit, yieldTurn }) {
      emit({ type: 'start', promptTokens: 3 });
      await yieldTurn();
      for (const [i, text] of TOKENS.entries()) {
        timeline.push({ at: performance.now(), what: 'start' });
        spin(SPIN_MS);
        emit({ type: 'content', text });
        if (i < TOKENS.length - 1) await yieldTurn();
      }
      return { finishReason: 'length', usage: {}, timings: {}, settings: {} };
    },
  };
  return {
    key: 'test/blocking-chat',
    id: 'test/blocking-chat',
    variant: 'fp32',
    info: { inputModalities: ['text'] },
    engine: { name: 'blocking', capability: 'chat', load: async () => instance },
  };
}

describe('chat streams token by token from a synchronous engine', () => {
  const timeline = [];
  let server;

  before(async () => {
    const resolved = blockingEngine(timeline);
    const catalog = { aliases: {}, list: async () => [], resolve: async () => resolved };
    const runtime = createRuntime({
      env: {},
      backend: async () => ({ track: 'node', executionProviders: ['cpu'] }),
    });
    server = await startServer({ env: {}, port: 0, catalog, runtime });
  });
  after(() => server?.close());

  test('each token arrives before the next one is generated', async () => {
    timeline.length = 0;
    const response = await fetch(`${server.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ stream: true, messages: [{ role: 'user', content: 'Count.' }] }),
    });
    assert.equal(response.status, 200);
    const arrivals = [];
    const decoder = new TextDecoder();
    let buffer = '';
    for await (const chunk of response.body) {
      buffer += decoder.decode(chunk, { stream: true });
      for (let i = buffer.indexOf('\n\n'); i >= 0; i = buffer.indexOf('\n\n')) {
        const data = buffer.slice(6, i);
        buffer = buffer.slice(i + 2);
        if (data === '[DONE]') continue;
        const content = JSON.parse(data).choices[0]?.delta.content;
        if (content) arrivals.push({ at: performance.now(), content });
      }
    }
    const starts = timeline.map((e) => e.at);
    assert.deepEqual(
      arrivals.map((a) => a.content),
      TOKENS
    );
    const report = arrivals
      .map((a, i) => `token ${i + 1} arrived at ${Math.round(a.at - starts[0])} ms`)
      .join(', ');
    const generated = starts.map((s) => `${Math.round(s - starts[0])}`).join(', ');
    const message = `${report}; tokens started at ${generated} ms`;
    for (let i = 0; i < TOKENS.length - 1; i++) assert.ok(arrivals[i].at < starts[i + 1], message);
  });
});

test('a more urgent job runs while a stream waits for its client to read', async () => {
  const timeline = [];
  const resolved = blockingEngine(timeline);
  const catalog = { aliases: {}, list: async () => [], resolve: async () => resolved };
  const runtime = createRuntime({
    env: {},
    backend: async () => ({ track: 'node', executionProviders: ['cpu'] }),
  });
  const route = chatRoute({ catalog, runtime });
  const { events } = await route.handle({
    body: { stream: true, messages: [{ role: 'user', content: 'Count.' }] },
  });
  const asked = performance.now();
  await new Promise((resolve) => setTimeout(resolve, 50));
  const ran = await runtime.run(PRIORITY.realtime, async () => performance.now());
  let chunks = 0;
  for await (const _ of events) chunks++;
  await runtime.close();
  assert.equal(chunks, TOKENS.length + 2);
  assert.ok(
    ran < timeline[0].at,
    `the urgent job ran ${Math.round(ran - asked)} ms after the stream started, the first token at ${Math.round(timeline[0].at - asked)} ms`
  );
});

import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { PIXTRAL, pixtralSize } from '@ai-ecoverse/kev.js';
import { systemone } from '../../src/engines/systemone.js';
import { createRuntime, startServer } from '../../src/index.js';

const WORDS = 6;
const SPIN_MS = 150;
const TURN = ['\n', '\n', 'User', ':', ' Next', ' question'];

function spin(ms) {
  const until = performance.now() + ms;
  while (performance.now() < until);
}

function fakeKev(timeline) {
  return {
    PIXTRAL,
    pixtralSize,
    ValidationError: class extends Error {},
    ContextOverflow: class extends Error {},
    async loadKevMinistral(_read, { ort }) {
      const session = await ort.InferenceSession.create('decoder.onnx');
      let queue = Promise.resolve();
      const serial = (fn) => {
        const run = queue.then(fn, fn);
        queue = run.catch(() => undefined);
        return run;
      };
      return {
        acceptsImages: false,
        tokenizer: { token_to_id: (name) => ({ '<s>': 1, '[IMG]': 10 })[name] },
        userTokens: (text) => [...text].map((c) => c.charCodeAt(0)),
        systemOne: () =>
          serial(async () => {
            await session.run({});
            return { answers: { ok: { type: 'noul', noul: 1 } }, usage: {} };
          }),
        generate: (ids, { maxTokens, onToken }) =>
          serial(async () => {
            const turn = String.fromCharCode(...ids.slice(1)).includes('Turn.');
            let text = '';
            for (let i = 0; i < maxTokens; i++) {
              await session.run({});
              if (turn) text += TURN[i] ?? ' more';
              else {
                timeline.push(performance.now());
                spin(SPIN_MS);
                text += ` word${i}`;
              }
              onToken(text, i + 1);
            }
            return { text, ids: [], inputTokens: ids.length, finish: 'length' };
          }),
        release: async () => {},
      };
    },
  };
}

function resolvedFor(kevjs, mode) {
  const info = {
    ministral: true,
    variants: ['q8f32'],
    inputModalities: ['text'],
    contextLength: 4096,
    modes: { base: { variant: 'q8f32', capability: 'chat', info: {} } },
  };
  return {
    key: 'test/kev-ministral:q8f32',
    id: 'test/kev-ministral',
    variant: mode ?? 'q8f32',
    mode: mode ?? null,
    capability: mode ? 'chat' : 'systemone',
    info,
    files: {},
    engine: { ...systemone, load: (o) => systemone.load({ ...o, library: async () => kevjs }) },
  };
}

describe('kev-ministral base weights stream from a synchronous decoder', () => {
  const timeline = [];
  let server;

  before(async () => {
    const kevjs = fakeKev(timeline);
    const catalog = {
      aliases: {},
      list: async () => [],
      resolve: async (id) => resolvedFor(kevjs, id.endsWith(':base') ? 'base' : null),
    };
    const runtime = createRuntime({
      env: {},
      backend: async () => ({
        track: 'node',
        executionProviders: ['cpu'],
        ort: { InferenceSession: { create: async () => ({ run: async () => ({}) }) } },
      }),
    });
    server = await startServer({ env: {}, port: 0, catalog, runtime });
  });
  after(() => server?.close());

  test('each token arrives before the next one is generated, and decisions wait their turn', async () => {
    const response = await fetch(`${server.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'test/kev-ministral:base',
        stream: true,
        max_tokens: WORDS,
        messages: [{ role: 'user', content: 'Count.' }],
      }),
    });
    assert.equal(response.status, 200);
    const decision = fetch(`${server.url}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'test/kev-ministral',
        state: 'x',
        questions: { ok: { type: 'noul', instructions: 'ok?' } },
      }),
    }).then(async (r) => ({ status: r.status, at: performance.now() }));
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
    assert.equal(arrivals.map((a) => a.content).join(''), 'word0 word1 word2 word3 word4 word5');
    const report = arrivals.map((a) => Math.round(a.at - timeline[0])).join(', ');
    for (let i = 0; i < WORDS - 1; i++)
      assert.ok(arrivals[i].at < timeline[i + 1], `tokens arrived at ${report} ms`);
    const { status, at } = await decision;
    assert.equal(status, 200);
    assert.ok(at > timeline.at(-1));
  });

  test('a reply that opens with the next turn stops there', async () => {
    const response = await fetch(`${server.url}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'test/kev-ministral:base',
        max_tokens: 20,
        messages: [{ role: 'user', content: 'Turn.' }],
      }),
    });
    const body = await response.json();
    assert.equal(response.status, 200, JSON.stringify(body));
    assert.equal(body.choices[0].message.content, '');
    assert.equal(body.choices[0].finish_reason, 'stop');
    assert.equal(body.usage.completion_tokens, 4);
  });

  test('System One on the base id names the decision model', async () => {
    const response = await fetch(`${server.url}/v1/systemone`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'test/kev-ministral:base',
        state: 'x',
        questions: { ok: { type: 'noul', instructions: 'ok?' } },
      }),
    });
    const body = await response.json();
    assert.equal(response.status, 400);
    assert.match(
      body.error.message,
      /test\/kev-ministral:base does not answer System One requests; test\/kev-ministral does$/
    );
  });
});

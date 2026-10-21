import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { decodeFile } from '../../src/audio/decode.js';
import { createRuntime, startServer } from '../../src/index.js';

const SENTENCES = ['One fish swims.', 'Two fish dive.', 'Red fish rest.', 'Blue fish sleep.'];
const SPIN_MS = 200;

function spin(ms) {
  const until = performance.now() + ms;
  while (performance.now() < until);
}

function blockingEngine(timeline) {
  const instance = {
    sampleRate: 24000,
    async tokens(text) {
      return [[text.length]];
    },
    async synthesize() {
      timeline.push({ at: performance.now(), what: 'start' });
      spin(SPIN_MS);
      timeline.push({ at: performance.now(), what: 'end' });
      return new Float32Array(2400).fill(0.1);
    },
  };
  return {
    key: 'test/blocking-tts',
    id: 'test/blocking-tts',
    variant: 'fp32',
    info: {},
    engine: { name: 'blocking', capability: 'speech', load: async () => instance },
  };
}

describe('speech streams sentence by sentence from a synchronous engine', () => {
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

  for (const format of ['wav', 'pcm', 'aac']) {
    test(`${format}: the first sentence arrives before the last one is synthesized`, async () => {
      timeline.length = 0;
      const response = await fetch(`${server.url}/v1/audio/speech`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ input: SENTENCES.join(' '), response_format: format }),
      });
      assert.equal(response.status, 200);
      const arrivals = [];
      const chunks = [];
      for await (const chunk of response.body) {
        arrivals.push(performance.now());
        chunks.push(chunk);
      }
      const bytes = new Uint8Array(Buffer.concat(chunks));
      const starts = timeline.filter((e) => e.what === 'start').map((e) => e.at);
      const ends = timeline.filter((e) => e.what === 'end').map((e) => e.at);
      assert.equal(starts.length, SENTENCES.length);
      if (format === 'aac') {
        assert.equal(response.headers.get('content-type'), 'audio/aac');
        const decoded = await decodeFile(bytes, { rate: 24000 });
        assert.equal(decoded.length, (Math.ceil((SENTENCES.length * 2400) / 1024) + 1) * 1024);
        assert.ok(Math.abs(decoded[1024 + 4800] - 0.1) < 0.01, `${decoded[1024 + 4800]}`);
      } else assert.equal(bytes.length, SENTENCES.length * 4800 + (format === 'wav' ? 44 : 0));
      const report = `first chunk ${Math.round(arrivals[0] - starts[0])} ms after the first synthesis started; last synthesis started at ${Math.round(starts.at(-1) - starts[0])} ms and ended at ${Math.round(ends.at(-1) - starts[0])} ms`;
      assert.ok(arrivals[0] < starts.at(-1), report);
      if (format !== 'aac') assert.ok(arrivals[0] < starts[1], report);
    });
  }
});

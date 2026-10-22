import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { phonon } from '../../src/engines/phonon.js';
import { createRuntime, startServer } from '../../src/index.js';
import { normalizeEnglish } from '../../src/phonon/text.js';

const BYTES = Array.from({ length: 256 }, (_, b) => [
  `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`,
  -5,
]);
const TOKENIZER = {
  model: {
    type: 'Unigram',
    unk_id: 0,
    byte_fallback: true,
    vocab: [['<unk>', 0], ['\u2581', -1], ...BYTES],
  },
};

const CONFIG = {
  flow_lm: { d_model: 4 },
  mimi: { frame_rate: 12.5 },
  eos_threshold: 4,
  onnx: {
    num_layers: 1,
    num_heads: 1,
    head_dim: 2,
    latent_dim: 2,
    mimi_state: [['conv', [1, 2]]],
    voices: [{ name: 'easton' }],
    default_voice: 'easton',
  },
};

function memoryFiles() {
  const encoder = new TextEncoder();
  const map = new Map([
    ['config.json', JSON.stringify(CONFIG)],
    ['tokenizer.json', JSON.stringify(TOKENIZER)],
    ...['text_conditioner', 'flow_lm_main', 'flow_lm_flow', 'mimi_decoder'].map((g) => [
      `onnx/${g}.onnx`,
      g,
    ]),
    ['voices/easton.bin', new Uint8Array(new Float32Array(8).buffer)],
  ]);
  return {
    exists: async (rel) => map.has(rel),
    json: async (rel) => JSON.parse(map.get(rel)),
    read: async (rel) =>
      typeof map.get(rel) === 'string' ? encoder.encode(map.get(rel)) : map.get(rel),
  };
}

function spin(ms) {
  const until = performance.now() + ms;
  while (performance.now() < until);
}

function fakeOrt({ eosAt = 3, spinMs = 0, paced = false } = {}) {
  const outputs = [];
  const deltas = [];
  const flows = [];
  const noises = [];
  const runs = {};
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims, disposed: false });
    }

    async getData() {
      if (this.disposed) throw new Error('read after dispose');
      return this.data;
    }

    dispose() {
      this.disposed = true;
    }
  }
  const made = (type, data, dims) => {
    const t = new Tensor(type, data, dims);
    outputs.push(t);
    return t;
  };
  let steps = 0;
  const graphs = {
    text_conditioner: ({ input_ids: ids }) => ({
      inputs_embeds: made('float32', new Float32Array(ids.dims[1] * 4), [1, ids.dims[1], 4]),
      bos_embeds: made('float32', new Float32Array(4), [1, 1, 4]),
    }),
    flow_lm_main: (feeds) => {
      const t = feeds.inputs_embeds.dims[1];
      const p = feeds['past_key_values.0.key'].dims[2];
      steps += t === 1 ? 1 : 0;
      const eos = t === 1 && steps % eosAt === 0 ? 9 : -9;
      if (t === 1) spin(spinMs);
      return {
        hidden: made('float32', new Float32Array(4), [1, 4]),
        eos_logit: made('float32', Float32Array.of(eos), [1, 1]),
        'present.0.key': made('float32', new Float32Array((p + t) * 2), [1, 1, p + t, 2]),
        'present.0.value': made('float32', new Float32Array((p + t) * 2), [1, 1, p + t, 2]),
      };
    },
    flow_lm_flow: ({ noise }) => {
      flows.push(performance.now());
      noises.push([...noise.data]);
      return {
        latent: made('float32', Float32Array.from(noise.data), [1, 1, 2]),
        next_embeds: made('float32', new Float32Array(4), [1, 1, 4]),
      };
    },
    mimi_decoder: (feeds) => ({
      audio: made('float32', new Float32Array(feeds.latent.dims[1] * 2400).fill(0.1), [1, 1, 2400]),
      next_offset: made('int64', BigInt64Array.of(feeds.offset.data[0] + 1n), []),
      'next_state.conv': made('float32', new Float32Array(2), [1, 2]),
    }),
  };
  const ort = {
    Tensor,
    InferenceSession: {
      async create(bytes) {
        const name = new TextDecoder().decode(bytes);
        const conditioned = paced && ['text_conditioner', 'flow_lm_flow'].includes(name);
        return {
          inputNames: conditioned ? ['duration_delta'] : [],
          async run(feeds) {
            for (const t of Object.values(feeds))
              if (t.disposed) throw new Error(`${name} was fed a disposed tensor`);
            if (conditioned && !feeds.duration_delta)
              throw new Error(`input 'duration_delta' is missing in 'feeds'`);
            if (feeds.duration_delta) deltas.push([name, feeds.duration_delta.data[0]]);
            runs[name] = (runs[name] ?? 0) + 1;
            return graphs[name](feeds);
          },
          release: async () => {},
        };
      },
    },
  };
  return { ort, outputs, noises, runs, deltas, flows };
}

async function load(options) {
  const fake = fakeOrt(options);
  const files = memoryFiles();
  const instance = await phonon.load({
    backend: { ort: fake.ort, executionProviders: ['cpu'] },
    files,
    variant: 'fp32',
    info: await phonon.detect(files),
  });
  return { ...fake, instance, files };
}

const TWO =
  'The first sentence of this request is long enough. The second one follows it right away.';

test('each chunk of a request draws new noise, and a new request starts over', async () => {
  const { instance, noises } = await load();
  const chunks = await instance.tokens(TWO);
  assert.equal(chunks.length, 2);
  await instance.synthesize(chunks[0]);
  const first = noises.length;
  await instance.synthesize(chunks[1]);
  assert.notDeepEqual(noises[first], noises[0]);
  const again = await instance.tokens(TWO);
  const before = noises.length;
  await instance.synthesize(again[0]);
  assert.deepEqual(noises[before], noises[0]);
});

test('generation frees every tensor but the cached voice prompt, which release frees', async () => {
  const { instance, outputs } = await load();
  for await (const _ of instance.speak(TWO));
  const live = outputs.filter((t) => !t.disposed);
  assert.equal(live.length, 2, `${live.length} tensors left`);
  assert.deepEqual(
    live.map((t) => t.dims[2]),
    [2, 2]
  );
  await instance.release();
  assert.equal(outputs.filter((t) => !t.disposed).length, 0);
});

async function serve(options) {
  const fake = fakeOrt(options);
  const files = memoryFiles();
  const resolved = {
    key: 'test/phonon',
    id: 'test/phonon',
    variant: 'fp32',
    files,
    info: await phonon.detect(files),
    engine: phonon,
  };
  const catalog = { aliases: {}, list: async () => [], resolve: async () => resolved };
  const runtime = createRuntime({
    env: {},
    backend: async () => ({ ort: fake.ort, track: 'node', executionProviders: ['cpu'] }),
  });
  const server = await startServer({ env: {}, port: 0, catalog, runtime });
  const post = (body) =>
    fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ response_format: 'pcm', ...body }),
    });
  return { fake, server, post };
}

describe('speech hands Phonon the whole input', () => {
  let served;

  before(async () => {
    served = await serve();
  });
  after(() => served?.server.close());

  test('abbreviations the sentence splitter does not know stay in their sentence', async () => {
    const input = 'Ft. Worth is large. Vol. 2 is out.';
    assert.equal(normalizeEnglish(input), 'Fort Worth is large. Volume 2 is out.');
    const response = await served.post({ input });
    assert.equal(response.status, 200);
    assert.ok((await response.arrayBuffer()).byteLength > 0);
    assert.equal(served.fake.runs.text_conditioner, 1);
  });
});

const LONG = 'This one sentence takes many frames to say.';

test('a chunk streams its first Mimi frame alone, then groups that double up to 16', async () => {
  const { instance, runs } = await load({ eosAt: 1000 });
  const [chunk] = await instance.tokens(LONG);
  const sizes = [];
  for await (const audio of instance.stream(chunk)) sizes.push(audio.length / 2400);
  const decoded = runs.mimi_decoder;
  assert.deepEqual(sizes.slice(0, 6), [1, 2, 4, 8, 16, 16]);
  assert.equal(
    sizes.reduce((a, b) => a + b, 0),
    chunk.frameBudget
  );
  assert.equal(decoded, sizes.length);
  const whole = await instance.synthesize(chunk);
  assert.equal(whole.length, chunk.frameBudget * 2400);
});

describe('speech streams Mimi frames from inside a chunk', () => {
  let served;

  before(async () => {
    served = await serve({ eosAt: 1000, spinMs: 5 });
    await (await served.post({ input: 'Hi.' })).arrayBuffer();
  });
  after(() => served?.server.close());

  for (const format of ['pcm', 'wav']) {
    test(`${format}: audio arrives frame by frame while the chunk is generated`, async () => {
      served.fake.flows.length = 0;
      const started = performance.now();
      const response = await served.post({ input: LONG, response_format: format });
      assert.equal(response.status, 200);
      const arrivals = [];
      let bytes = 0;
      for await (const chunk of response.body) {
        arrivals.push({ at: performance.now(), bytes: chunk.length });
        bytes += chunk.length;
      }
      const flows = served.fake.flows;
      const last = flows.at(-1);
      const early = arrivals.filter((a) => a.at < last).reduce((n, a) => n + a.bytes, 0);
      const report = `first bytes at ${Math.round(arrivals[0].at - started)} ms, frame 10 at ${Math.round(flows[9] - started)} ms, last frame at ${Math.round(last - started)} ms, ${early} of ${bytes} bytes before it`;
      assert.ok(arrivals[0].at < flows[9], report);
      assert.ok(early > bytes / 2, report);
      assert.equal(bytes, flows.length * 4800 + (format === 'wav' ? 44 : 0));
    });
  }
});

describe('speed maps onto the duration conditioner when the export takes it', () => {
  let paced;
  let fixed;

  before(async () => {
    paced = await serve({ paced: true });
    fixed = await serve();
  });
  after(() => Promise.all([paced?.server.close(), fixed?.server.close()]));

  test('both conditioned graphs get duration_delta, 0 at speed 1 and clamped at the ends', async () => {
    const expected = {
      1: 0,
      1.25: -2 * Math.log(1.25),
      0.9: -2 * Math.log(0.9),
      4: -0.6,
      0.25: 0.5,
    };
    for (const [speed, delta] of Object.entries(expected)) {
      paced.fake.deltas.length = 0;
      const response = await paced.post({ input: 'Hi there.', speed: Number(speed) });
      assert.equal(response.status, 200, `speed ${speed}`);
      await response.arrayBuffer();
      const graphs = new Set(paced.fake.deltas.map(([g]) => g));
      assert.deepEqual([...graphs].sort(), ['flow_lm_flow', 'text_conditioner']);
      for (const [, d] of paced.fake.deltas)
        assert.ok(Math.abs(d - delta) < 1e-6, `${speed}: ${d}`);
    }
  });

  test('answers 400 only outside 0.25 to 4, or for any other speed on an export without it', async () => {
    const outside = await paced.post({ input: 'Hi.', speed: 4.5 });
    assert.equal(outside.status, 400);
    assert.equal((await outside.json()).error.param, 'speed');
    const unpaced = await fixed.post({ input: 'Hi.', speed: 1.25 });
    assert.equal(unpaced.status, 400);
    assert.equal((await unpaced.json()).error.param, 'speed');
    const normal = await fixed.post({ input: 'Hi.' });
    assert.equal(normal.status, 200);
    await normal.arrayBuffer();
    assert.equal(fixed.fake.deltas.length, 0);
  });
});

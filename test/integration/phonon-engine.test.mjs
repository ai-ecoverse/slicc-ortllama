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

function fakeOrt() {
  const outputs = [];
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
      const eos = t === 1 && steps % 3 === 0 ? 9 : -9;
      return {
        hidden: made('float32', new Float32Array(4), [1, 4]),
        eos_logit: made('float32', Float32Array.of(eos), [1, 1]),
        'present.0.key': made('float32', new Float32Array((p + t) * 2), [1, 1, p + t, 2]),
        'present.0.value': made('float32', new Float32Array((p + t) * 2), [1, 1, p + t, 2]),
      };
    },
    flow_lm_flow: ({ noise }) => {
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
        return {
          async run(feeds) {
            for (const t of Object.values(feeds))
              if (t.disposed) throw new Error(`${name} was fed a disposed tensor`);
            runs[name] = (runs[name] ?? 0) + 1;
            return graphs[name](feeds);
          },
          release: async () => {},
        };
      },
    },
  };
  return { ort, outputs, noises, runs };
}

async function load() {
  const fake = fakeOrt();
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

describe('speech hands Phonon the whole input', () => {
  let server;
  let fake;

  before(async () => {
    fake = fakeOrt();
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
    server = await startServer({ env: {}, port: 0, catalog, runtime });
  });
  after(() => server?.close());

  test('abbreviations the sentence splitter does not know stay in their sentence', async () => {
    const input = 'Ft. Worth is large. Vol. 2 is out.';
    assert.equal(normalizeEnglish(input), 'Fort Worth is large. Volume 2 is out.');
    const response = await fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ input, response_format: 'pcm' }),
    });
    assert.equal(response.status, 200);
    assert.ok((await response.arrayBuffer()).byteLength > 0);
    assert.equal(fake.runs.text_conditioner, 1);
  });
});

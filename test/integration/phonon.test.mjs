import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { decodeFile } from '../../src/audio/decode.js';
import { openCatalog, startServer } from '../../src/index.js';
import { normalizeEnglish, plan } from '../../src/phonon/text.js';
import { createTokenizer } from '../../src/phonon/tokenizer.js';

const PHONON = 'ai-ecoverse/phonon-31m-en-ONNX';
const TEXT =
  'The old lighthouse stood at the edge of the cliff for more than a hundred years. Every night its keeper climbed the narrow stairs, trimmed the wick, and watched the beam sweep across the dark water. Ships passing in the fog relied on that steady light to find their way around the rocks.';

const PTTS = [
  ['I paid $45.99 for the headphones.', 'I paid 45 dollars and 99 cents for the headphones.'],
  [
    'Our revenue grew by 12.5% in 2024, reaching $3 million.',
    'Our revenue grew by 12.5% in 2024, reaching 3 million dollars.',
  ],
  [
    'Send the report to jane.doe@example.com before Friday.',
    'Send the report to jane dot doe at example dot com before Friday.',
  ],
  [
    'You can find the documentation at https://docs.example.org/guide.',
    'You can find the documentation at H-T-T-P-S colon slash slash docs dot example dot org slash guide.',
  ],
  ['There are 1,234 people.', 'There are 1 thousand 234 people.'],
  ['The temperature dropped to -5 degrees.', 'The temperature dropped to minus 5 degrees.'],
  ['Wow, that is sooooo cool!', 'Wow, that is so cool!'],
  [
    'Call 555-123-4567 or +44 20 7946 0958.',
    'Call 555 123 4 5 6 7 or plus four four, two zero, seven nine four six, zero nine five eight.',
  ],
  ['Lt Col Jones arrived on Jan. 5th.', 'Lieutenant Colonel Jones arrived on January 5th.'],
  ['Meet me on 2024-05-12 at 10:30.', 'Meet me on 2024-05-12 at 10:30.'],
  ['Visit www.kyutai.fr now.', 'Visit W-W-W dot kyutai dot F-R now.'],
  ['The ratio is 3×4 ≈ 12 ± 1.', 'The ratio is 3 times 4 approximately 12 plus or minus 1.'],
  ['Temperatures: 20°C to 25°F.', 'Temperatures, 20 degrees Celsius to 25 degrees Fahrenheit.'],
  ['Rated ★★★☆☆ by users.', 'Rated 3 out of 5 stars by users.'],
  ['“Hello”; please email user@host (now)… 🚀', '"Hello", please email user at host, now.'],
  ['and/or, mg/kg, km/h, c/o.', 'and or, mg/kg, km/h, c/o.'],
  ['007 is a code.', 'zero zero seven is a code.'],
  ['From 1445-1450 and 20-25 people.', 'From 1445 to 1450 and 20 to 25 people.'],
];

test('English normalization reads text the way ptts --lang en does', () => {
  for (const [input, spoken] of PTTS) assert.equal(normalizeEnglish(input), spoken, input);
});

test('the tokenizer falls back to bytes for pieces it lacks, as the Rust tokenizer does', () => {
  const tokenizer = createTokenizer({
    model: {
      type: 'Unigram',
      unk_id: 0,
      byte_fallback: true,
      vocab: [
        ['<unk>', 0],
        ['\u2581Meet', -1],
        ['\u2581at', -1],
        ['\u2581', -2],
        ['3', -2],
        ['0', -2],
        ['.', -2],
        ['<0x3A>', 0],
      ],
    },
  });
  const ids = tokenizer.encode('Meet at 3:30.');
  assert.deepEqual(ids, [1, 2, 3, 4, 7, 4, 5, 6]);
  assert.equal(tokenizer.decode(ids), 'Meet at 3:30.');
  assert.deepEqual(
    plan(tokenizer, 'meet at 3:30').map((c) => [c.text, c.framesAfterEos, c.frameBudget]),
    [['Meet at 3:30.', 3, 59]]
  );
});

async function cached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(PHONON);
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${PHONON} is not in the Hugging Face cache`;

function pcmStats(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let sum = 0;
  for (let i = 0; i + 1 < bytes.length; i += 2) sum += (view.getInt16(i, true) / 32768) ** 2;
  const samples = Math.floor(bytes.length / 2);
  return { seconds: samples / 24000, rms: Math.sqrt(sum / Math.max(1, samples)) };
}

function plausible(seconds, text) {
  const expected = text.length / 15;
  assert.ok(
    seconds > expected / 2 && seconds < expected * 2,
    `${seconds.toFixed(2)} s for ${text.length} characters`
  );
}

describe('Phonon speech', { skip }, () => {
  let server;

  before(async () => {
    server = await startServer({ env: { ...process.env, ORTLLAMA_API_KEY: '' }, port: 0 });
  });
  after(async () => {
    await server?.close();
  });

  const post = (body) =>
    fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: PHONON, response_format: 'pcm', ...body }),
    });

  test('streams 24 kHz speech frame by frame in the default voice', async () => {
    await (await post({ input: 'Hi.' })).arrayBuffer();
    const started = performance.now();
    const response = await post({ input: TEXT });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'audio/pcm');
    let first;
    const chunks = [];
    for await (const chunk of response.body) {
      first ??= performance.now() - started;
      chunks.push(chunk);
    }
    const total = performance.now() - started;
    assert.ok(first < total * 0.25, `first audio after ${first} of ${total} ms`);
    const { seconds, rms } = pcmStats(new Uint8Array(Buffer.concat(chunks)));
    plausible(seconds, TEXT);
    assert.ok(rms > 0.02, `rms ${rms}`);
  });

  test('speaks in a named voice, through the codecs', async () => {
    const input = 'Good evening, and welcome to the news at 6:30.';
    const response = await post({ input, voice: 'Freya', response_format: 'mp3' });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'audio/mpeg');
    const samples = await decodeFile(new Uint8Array(await response.arrayBuffer()), {
      rate: 24000,
    });
    plausible(samples.length / 24000, input);
  });

  test('answers 400 for an unknown voice and for a speed out of range', async () => {
    const voice = await post({ input: 'Hi.', voice: 'af_heart' });
    assert.equal(voice.status, 400);
    assert.equal((await voice.json()).error.param, 'voice');
    const speed = await post({ input: 'Hi.', speed: 4.5 });
    assert.equal(speed.status, 400);
    assert.equal((await speed.json()).error.param, 'speed');
  });

  test('speaks faster and slower when the export takes duration_delta, else answers 400', async () => {
    const resolved = await (
      await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })
    ).resolve(PHONON);
    const config = await resolved.files.json('config.json');
    const input = 'The committee will meet again on Thursday afternoon to review the budget.';
    const seconds = async (speed) => {
      const response = await post({ input, speed });
      if (response.status !== 200) return response;
      return pcmStats(new Uint8Array(await response.arrayBuffer())).seconds;
    };
    if (config.onnx.inputs?.duration_delta === undefined) {
      const response = await seconds(1.25);
      assert.equal(response.status, 400);
      assert.equal((await response.json()).error.param, 'speed');
      return;
    }
    const normal = await seconds(1);
    const fast = await seconds(1.25);
    const slow = await seconds(0.85);
    assert.ok(fast < normal * 0.92 && slow > normal * 1.05, `${fast} ${normal} ${slow} s`);
    const fastest = await seconds(4);
    const slowest = await seconds(0.25);
    assert.ok(fastest <= fast && slowest >= slow * 0.97, `${fastest} ${slowest} s`);
    assert.ok(fastest > normal * 0.7 && slowest < normal * 1.35, `${fastest} ${slowest} s`);
  });

  test('lists Phonon as a speech model', async () => {
    const { data } = await (await fetch(`${server.url}/v1/models`)).json();
    const model = data.find((m) => m.id.startsWith(PHONON));
    assert.ok(model, JSON.stringify(data.map((m) => m.id)));
  });
});

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { after, before, describe, test } from 'node:test';
import { openCatalog, startServer } from '../../src/index.js';

const MODEL = 'ai-ecoverse/kev-ministral/kev-ministral-3b';
const png = readFileSync(new URL('./fixtures/red-circle.png', import.meta.url)).toString('base64');

async function cached() {
  try {
    const catalog = await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } });
    return (await catalog.resolve(MODEL)).info.inputModalities.includes('image');
  } catch {
    return false;
  }
}

const skip =
  !(await cached()) &&
  !process.env.ORTLLAMA_REQUIRE_MODEL &&
  `${MODEL} is not in the Hugging Face cache`;

const MENU = {
  state: [
    'Goal: Read the comments of the top story',
    'Page: Hacker News (https://news.ycombinator.com/)',
    'Page text:',
    '  1. Show HN: A tiny ONNX server (example.com) 312 points by ada 4 hours ago | 87 comments',
    '  2. The history of the lighthouse (example.org) 120 points by bob 6 hours ago | 31 comments',
    'Controls:',
    '  [e12] link "Show HN: A tiny ONNX server"',
    '  [e17] link "87 comments"',
    '  [e23] link "The history of the lighthouse"',
    '  [e29] link "31 comments"',
    '  [e40] link "login"',
  ].join('\n'),
  questions: {
    action: {
      type: 'choice',
      instructions: 'Which action makes progress toward the goal?',
      criteria: {
        'click:e12': 'link "Show HN: A tiny ONNX server"',
        'click:e17': 'link "87 comments"',
        'click:e23': 'link "The history of the lighthouse"',
        'click:e29': 'link "31 comments"',
        'click:e40': 'link "login"',
      },
    },
  },
};

let server;
const post = async (body) => {
  const response = await fetch(`${server.url}/v1/systemone`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: response.status, body: await response.json() };
};

describe('kev-ministral on /v1/systemone through kev.js', { skip }, () => {
  before(async () => {
    server = await startServer({
      env: { ...process.env, ORTLLAMA_API_KEY: '', ORTLLAMA_KEEP_ALIVE: '-1' },
      port: 0,
    });
  });
  after(() => server?.close());

  test('is listed as a System One model that takes images', async () => {
    const { data } = await (await fetch(`${server.url}/v1/models`)).json();
    const entry = data.find((m) => m.id === MODEL);
    assert.ok(entry, JSON.stringify(data.map((m) => m.id)));
    assert.deepEqual(entry.architecture.input_modalities, ['text', 'image']);
    assert.deepEqual(entry.architecture.output_modalities, ['decisions']);
  });

  test('picks the comments link on a webrunner menu', async () => {
    const { status, body } = await post({ model: MODEL, ...MENU });
    assert.equal(status, 200, JSON.stringify(body));
    assert.equal(body.model, MODEL);
    assert.equal(body.answers.action.choice, 'click:e17', JSON.stringify(body.answers));
    assert.ok(body.usage.input_tokens > 100);
  });

  test('answers about an image', async () => {
    const { status, body } = await post({
      model: MODEL,
      image: `data:image/png;base64,${png}`,
      state: 'Look at the picture.',
      questions: {
        circle: { type: 'noul', instructions: 'Does the picture show a circle?' },
        color: {
          type: 'choice',
          instructions: 'What color is the shape?',
          criteria: { red: null, green: null, blue: null },
        },
      },
    });
    assert.equal(status, 200, JSON.stringify(body));
    assert.ok(body.answers.circle.noul > 0.9, JSON.stringify(body.answers));
    assert.equal(body.answers.color.choice, 'red');
  });

  test('refuses an image URL it would have to fetch', async () => {
    const { status, body } = await post({
      model: MODEL,
      image: 'https://example.com/a.png',
      ...MENU,
    });
    assert.equal(status, 400, JSON.stringify(body));
  });
});

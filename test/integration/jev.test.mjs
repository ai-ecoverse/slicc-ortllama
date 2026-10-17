import assert from 'node:assert/strict';
import { after, before, describe, test } from 'node:test';
import { classify } from '@earendil-works/pi-ai/api/typesafe-system-one';
import { openCatalog, startServer } from '../../src/index.js';

const JEV = 'ai-ecoverse/jev-omni.js/jev-omni:q8f32';

async function cached() {
  try {
    await (await openCatalog({ env: { ...process.env, ORTLLAMA_MODEL_PATH: '' } })).resolve(JEV);
    return true;
  } catch {
    return false;
  }
}

const have = await cached();
const skip =
  !have && !process.env.ORTLLAMA_REQUIRE_MODEL && `${JEV} is not in the Hugging Face cache`;

describe('System One on Jev-Omni through pi-ai', { skip }, () => {
  let server;
  let model;

  before(async () => {
    server = await startServer({ env: { ...process.env, ORTLLAMA_API_KEY: '' }, port: 0 });
    model = {
      type: 'classifier',
      id: JEV,
      name: 'Jev-Omni',
      api: 'typesafe-system-one',
      provider: 'ortllama',
      baseUrl: `${server.url}/v1`,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 65536,
    };
  });
  after(() => server?.close());

  test('answers the kev.js README request', async () => {
    const result = await classify(
      model,
      {
        state: 'I was charged twice. Please fix this ASAP.',
        questions: {
          billing: {
            type: 'bool',
            instructions: 'Is this ticket about billing?',
            criteria: { true: 'Yes', false: 'No' },
          },
          urgency: {
            type: 'score',
            instructions: 'How urgent is this ticket?',
            criteria: ['can wait', 'this week', 'today'],
          },
        },
      },
      { apiKey: 'ortllama' }
    );
    assert.equal(result.stopReason, 'stop', result.errorMessage);
    assert.equal(result.answers.billing.type, 'bool');
    assert.ok(result.answers.billing.probability > 0.8);
    assert.ok(result.answers.urgency.score > 1 && result.answers.urgency.score <= 2);
    assert.ok(result.usage.input > 0);
  });

  test('reports the loaded model', async () => {
    const { models } = await (await fetch(`${server.url}/ortllama/ps`)).json();
    assert.equal(models.length, 1);
    assert.equal(models[0].id, 'ai-ecoverse/jev-omni.js/jev-omni');
  });
});

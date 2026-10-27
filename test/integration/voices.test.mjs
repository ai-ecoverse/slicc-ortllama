import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import { request } from 'node:http';
import { dirname, join, relative } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { hubCache, startServer } from '../../src/index.js';
import { fakeCache, fakeHome } from './cache.mjs';

const MAIN = 'c'.repeat(40);
const OLD = 'd'.repeat(40);
const KOKORO = 'onnx-community/Kokoro-82M-v1.0-ONNX';
const PHONON = 'ai-ecoverse/phonon-31m-en-ONNX';
const MUTE = 'test/phonon-mute';

async function put(cache, repo, sha, rel, content) {
  const dir = join(cache, `models--${repo.replace('/', '--')}`);
  const blob = join(dir, 'blobs', createHash('sha256').update(content).digest('hex'));
  await mkdir(dirname(blob), { recursive: true });
  await writeFile(blob, content);
  const link = join(dir, 'snapshots', sha, rel);
  await mkdir(dirname(link), { recursive: true });
  await symlink(relative(dirname(link), blob), link);
}

async function ref(cache, repo, sha) {
  const dir = join(cache, `models--${repo.replace('/', '--')}`, 'refs');
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'main'), sha);
}

async function kokoro(cache) {
  await ref(cache, KOKORO, MAIN);
  const add = (rel, content) => put(cache, KOKORO, MAIN, rel, content);
  await add('config.json', JSON.stringify({ model_type: 'style_text_to_speech_2' }));
  await add('tokenizer.json', '{}');
  await add('onnx/model.onnx', 'not a graph');
  for (const voice of ['af_heart', 'bm_george', 'ef_dora', 'jf_alpha', 'zm_yunxi', 'xx_nobody'])
    await add(`voices/${voice}.bin`, voice);
}

const PHONON_VOICES = [
  { name: 'easton', voice_id: 'E', locale: 'en-us', gender: 'male' },
  { name: 'maeve', voice_id: 'M' },
  { name: 'azgorath', voice_id: 'A', locale: 'characters', gender: null },
  { name: 'gone', voice_id: 'G', locale: 'en-us', gender: 'female' },
];

async function phonon(cache, sha, voices) {
  const add = (rel, content) => put(cache, PHONON, sha, rel, content);
  const onnx = { voices, default_voice: voices[0].name };
  await add('config.json', JSON.stringify({ languages: ['en'], flow_lm: {}, mimi: {}, onnx }));
  await add('tokenizer.json', '{}');
  for (const graph of ['text_conditioner', 'flow_lm_main', 'flow_lm_flow', 'mimi_decoder'])
    await add(`onnx/${graph}.onnx`, `${sha} ${graph}`);
  for (const v of voices) if (v.name !== 'gone') await add(`voices/${v.name}.bin`, v.name);
}

async function gradium(cache) {
  const lines = [
    { name: 'Maeve', voice_id: 'M', language: 'en', locale: 'en-gb', gender: 'female' },
    { name: 'Easton', voice_id: 'E', language: 'en', locale: 'en-us', gender: 'female' },
  ];
  const text = `${lines.map((l) => JSON.stringify(l)).join('\n')}\nnot json\n`;
  await put(cache, 'gradium/phonon-voices', MAIN, 'voices.jsonl', text);
}

function get(server, path, headers = {}) {
  return new Promise((resolve, reject) => {
    const r = request(`${server.url}${path}`, { headers }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        body += chunk;
      });
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body }));
    });
    r.on('error', reject);
    r.end();
  });
}

const json = async (server, path) => {
  const response = await get(server, path);
  return { status: response.status, ...JSON.parse(response.body) };
};

const voice = (data, id, model) => data.find((v) => v.id === id && v.model === model);

describe('GET /v1/audio/voices on a fake cache', () => {
  let server;
  let cache;
  let home;

  before(async () => {
    cache = await fakeCache();
    await kokoro(cache);
    await ref(cache, PHONON, MAIN);
    await phonon(cache, MAIN, PHONON_VOICES);
    await phonon(cache, OLD, [{ name: 'easton', voice_id: 'E', locale: 'en-gb', gender: 'male' }]);
    await gradium(cache);
    await ref(cache, MUTE, MAIN);
    await put(
      cache,
      MUTE,
      MAIN,
      'config.json',
      JSON.stringify({ flow_lm: {}, mimi: {}, onnx: {} })
    );
    await put(cache, MUTE, MAIN, 'tokenizer.json', '{}');
    for (const graph of ['text_conditioner', 'flow_lm_main', 'flow_lm_flow', 'mimi_decoder'])
      await put(cache, MUTE, MAIN, `onnx/${graph}.onnx`, `mute ${graph}`);
    await put(cache, MUTE, MAIN, 'voices/easton.bin', 'easton');
    home = await fakeHome({
      narrator: PHONON,
      'old-narrator': `${PHONON}@${OLD}`,
      decisions: 'test/kev/kev-tiny',
    });
    server = await startServer({
      env: { HF_HUB_CACHE: cache, ORTLLAMA_HOME: home, ORTLLAMA_API_KEY: 'sekrit' },
      port: 0,
    });
  });
  after(async () => {
    await server.close();
    await rm(cache, { recursive: true, force: true });
    await rm(home, { recursive: true, force: true });
  });

  const auth = { authorization: 'Bearer sekrit' };
  const list = async (query = '') => {
    const response = await get(server, `/v1/audio/voices${query}`, auth);
    return { status: response.status, ...JSON.parse(response.body) };
  };

  test('lists every cached speech model once, aliases included', async () => {
    const { status, object, data } = await list();
    assert.equal(status, 200);
    assert.equal(object, 'list');
    assert.deepEqual(
      data.map((v) => `${v.model} ${v.id}`),
      [
        `${PHONON} easton`,
        `${PHONON} maeve`,
        `${PHONON} azgorath`,
        `${KOKORO} af_heart`,
        `${KOKORO} bm_george`,
        `${KOKORO} ef_dora`,
        'old-narrator easton',
      ]
    );
    assert.deepEqual(voice(data, 'af_heart', KOKORO), {
      id: 'af_heart',
      model: KOKORO,
      engine: 'kokoro',
      language: 'en-US',
      gender: 'female',
    });
    assert.deepEqual(voice(data, 'bm_george', KOKORO), {
      id: 'bm_george',
      model: KOKORO,
      engine: 'kokoro',
      language: 'en-GB',
      gender: 'male',
    });
    assert.equal(voice(data, 'ef_dora', KOKORO).language, 'es-ES');
    assert.deepEqual(voice(data, 'easton', PHONON), {
      id: 'easton',
      model: PHONON,
      engine: 'phonon',
      language: 'en-US',
      gender: 'male',
    });
    assert.deepEqual(voice(data, 'maeve', PHONON), {
      id: 'maeve',
      model: PHONON,
      engine: 'phonon',
      language: 'en-GB',
      gender: 'female',
    });
    assert.deepEqual(voice(data, 'azgorath', PHONON), {
      id: 'azgorath',
      model: PHONON,
      engine: 'phonon',
      language: 'en',
      gender: null,
    });
    assert.equal(voice(data, 'easton', 'old-narrator').language, 'en-GB');
  });

  test('lists one model when asked, under the name it was asked by', async () => {
    const kokoroVoices = await list('?model=speech');
    assert.equal(kokoroVoices.status, 200);
    assert.deepEqual(
      kokoroVoices.data.map((v) => [v.id, v.model, v.engine]),
      [
        ['af_heart', 'speech', 'kokoro'],
        ['bm_george', 'speech', 'kokoro'],
        ['ef_dora', 'speech', 'kokoro'],
      ]
    );
    const phononVoices = await list(`?model=${encodeURIComponent(`${PHONON}:fp32`)}`);
    assert.deepEqual(
      phononVoices.data.map((v) => v.id),
      ['easton', 'maeve', 'azgorath']
    );
    assert.equal(phononVoices.data[0].model, `${PHONON}:fp32`);
  });

  test('lists no voices for a Phonon export whose onnx section names none', async () => {
    const { status, data } = await list(`?model=${MUTE}`);
    assert.equal(status, 200);
    assert.deepEqual(data, []);
  });

  test('answers as the speech route does for models it cannot speak with', async () => {
    const missing = await list('?model=nobody/none');
    assert.equal(missing.status, 404);
    assert.equal(missing.error.code, 'model_not_found');
    assert.match(missing.error.message, /hf download nobody\/none$/);
    const variant = await list(`?model=${encodeURIComponent(`${KOKORO}:q4`)}`);
    assert.equal(variant.status, 404);
    assert.equal(variant.error.code, 'variant_not_found');
    const other = await list('?model=decisions');
    assert.equal(other.status, 400);
    assert.equal(
      other.error.message,
      'model test/kev/kev-tiny does not synthesize speech; it answers POST /v1/systemone'
    );
    const empty = await list('?model=');
    assert.equal(empty.status, 400);
    assert.equal(empty.error.param, 'model');
  });

  test('loads no model to answer', async () => {
    await list();
    await list('?model=narrator');
    const ps = JSON.parse((await get(server, '/ortllama/ps', auth)).body);
    assert.deepEqual(ps.models, []);
  });

  test('keeps the API key, Host and CORS checks of the other routes', async () => {
    const denied = await get(server, '/v1/audio/voices');
    assert.equal(denied.status, 401);
    assert.equal(JSON.parse(denied.body).error.code, 'invalid_api_key');
    const rebound = await get(server, '/v1/audio/voices', { ...auth, host: 'rebind.example' });
    assert.equal(rebound.status, 403);
    const kernel = await get(server, '/v1/audio/voices?model=speech', {
      ...auth,
      origin: 'http://11435.kernel.localhost',
    });
    assert.equal(kernel.status, 200);
    assert.equal(kernel.headers['access-control-allow-origin'], 'http://11435.kernel.localhost');
    const other = await get(server, '/v1/audio/voices?model=speech', {
      ...auth,
      origin: 'https://example.com',
    });
    assert.equal(other.headers['access-control-allow-origin'], undefined);
  });
});

const real = hubCache(process.env);
const cached = (repo) =>
  existsSync(join(real, `models--${repo.replace('/', '--')}`, 'refs', 'main'));

describe('GET /v1/audio/voices on the real cache', () => {
  let server;
  let home;

  before(async () => {
    home = await fakeHome();
    server = await startServer({ env: { HF_HUB_CACHE: real, ORTLLAMA_HOME: home }, port: 0 });
  });
  after(async () => {
    await server.close();
    await rm(home, { recursive: true, force: true });
  });

  test('lists Kokoro voices a phonemizer can speak', { skip: !cached(KOKORO) }, async () => {
    const { status, data } = await json(server, `/v1/audio/voices?model=${KOKORO}`);
    assert.equal(status, 200);
    assert.ok(data.length > 0);
    assert.deepEqual(voice(data, 'af_heart', KOKORO), {
      id: 'af_heart',
      model: KOKORO,
      engine: 'kokoro',
      language: 'en-US',
      gender: 'female',
    });
    assert.equal(
      data.some((v) => /^[jz]/.test(v.id)),
      false
    );
    for (const v of data)
      assert.match(`${v.language} ${v.gender}`, /^[a-z]{2}-[A-Z]{2} (fe)?male$/);
  });

  test('lists the Phonon export voices', { skip: !cached(PHONON) }, async () => {
    const { status, data } = await json(server, `/v1/audio/voices?model=${PHONON}`);
    assert.equal(status, 200);
    assert.equal(voice(data, 'easton', PHONON).gender, 'male');
    assert.equal(voice(data, 'easton', PHONON).language, 'en-US');
    assert.equal(voice(data, 'freya', PHONON).language, 'en-GB');
    for (const v of data) {
      assert.equal(v.engine, 'phonon');
      assert.match(v.language, /^en(-US|-GB)?$/);
      assert.ok([null, 'female', 'male'].includes(v.gender));
    }
  });
});

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, describe, test } from 'node:test';
import { createAacCodec, PEERS } from '../../src/audio/aac-codec.js';
import { adtsFrames, readAdts } from '../../src/audio/adts.js';
import { decodeFile } from '../../src/audio/decode.js';
import { ENCODERS } from '../../src/audio/encode.js';
import { mp4Audio } from '../../src/audio/mp4.js';
import { concat, decodeWav, resample } from '../../src/audio/pcm.js';
import { HttpError } from '../../src/http/errors.js';
import { createRuntime, startServer } from '../../src/index.js';
import { fixture, foxWav } from './audio.mjs';

const ROOT = new URL('../../', import.meta.url);
const ALL_PEERS = [...PEERS.decode.flat(), ...PEERS.encode.flat()];

const missing = (name) =>
  Promise.reject(new HttpError(501, `no ${name}`, { code: 'engine_unavailable' }));

const only =
  (...names) =>
  (name) =>
    names.includes(name) ? import(name) : missing(name);

const none = () => createAacCodec({ scope: {}, load: missing });

const FRAME = 1024;

function fakeCodec(calls) {
  return {
    require: async () => {},
    async encoder(config) {
      calls.push(['encoder', config]);
      let held = new Float32Array(0);
      const cut = (all) => {
        const frames = [];
        let at = 0;
        for (; at + FRAME <= held.length || (all && at < held.length); at += FRAME) {
          const block = new Int16Array(FRAME);
          held.subarray(at, at + FRAME).forEach((x, i) => {
            block[i] = Math.round(x * 32767);
          });
          frames.push(new Uint8Array(block.buffer));
        }
        held = held.slice(at);
        return frames;
      };
      return {
        async encode([plane]) {
          held = concat([held, plane], Float32Array);
          return cut(false);
        },
        flush: async () => cut(true),
      };
    },
    async decoder({ asc }) {
      calls.push(['decoder', [...asc]]);
      return {
        sampleRate: 24000,
        async decode(frame) {
          calls.push(['frame', frame.length]);
          const block = new Int16Array(frame.slice().buffer, 0, frame.length >> 1);
          return [Float32Array.from(block, (x) => x / 32767)];
        },
        flush: async () => [],
      };
    },
  };
}

function similarity(a, b) {
  let dot = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < b.length; i++) {
    dot += a[i] * b[i];
    x += a[i] ** 2;
    y += b[i] ** 2;
  }
  return dot / Math.sqrt(x * y);
}

function snr(reference, decoded) {
  let best = -Infinity;
  for (let lag = 0; lag < 4096; lag += 1) {
    let signal = 0;
    let noise = 0;
    for (let i = 0; i < reference.length; i += 2) {
      signal += reference[i] ** 2;
      noise += (reference[i] - (decoded[i + lag] ?? 0)) ** 2;
    }
    best = Math.max(best, 10 * Math.log10(signal / noise));
  }
  return best;
}

async function encodeWith(open, samples) {
  const encoder = await open(24000);
  const parts = [];
  for (let at = 0; at < samples.length; at += 3000)
    parts.push(await encoder.push(samples.subarray(at, at + 3000)));
  parts.push(await encoder.end());
  return concat(parts);
}

async function fox24() {
  const { samples, rate } = decodeWav(await foxWav());
  return resample(samples, rate, 24000);
}

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(path)));
    else out.push(path);
  }
  return out;
}

describe('AAC goes through a codec seam', () => {
  test('src holds no AAC codec of its own', async () => {
    const src = new URL('src/', ROOT).pathname;
    const files = (await walk(src)).filter((f) => f.endsWith('.js'));
    const named = files.map((f) => f.slice(src.length)).filter((f) => /aac/i.test(f));
    assert.deepEqual(named, ['audio/aac-codec.js']);
    const signatures =
      /createAac(De|En)coder|SCALEFACTOR|SPECTRAL|TNS_(LONG|SHORT)|EIGHT_SHORT|LONG_START|INTENSITY_OUT|POW43|\bkbd\(|pairElement|skipPce|spectral\(|scalefactors\(bits, ch/;
    const users = [];
    for (const file of files) {
      const text = await readFile(file, 'utf8');
      assert.doesNotMatch(text, signatures, file);
      if (/from '\.\/mdct\.js'/.test(text)) users.push(file.slice(src.length));
    }
    assert.deepEqual(users.sort(), ['audio/layer3.js', 'audio/mp3-encoder.js', 'audio/vorbis.js']);
    const pkg = JSON.parse(await readFile(new URL('package.json', ROOT), 'utf8'));
    for (const name of ALL_PEERS) {
      assert.equal(pkg.dependencies[name], undefined, name);
      assert.equal(pkg.peerDependenciesMeta[name]?.optional, true, name);
      assert.equal(pkg.peerDependencies[name], pkg.devDependencies[name], name);
    }
  });

  test('a fake codec decodes M4A and ADTS frames and its frames come out as ADTS', async () => {
    const calls = [];
    const codec = fakeCodec(calls);
    const m4a = await fixture('m4a');
    const { frames } = mp4Audio(m4a);
    await decodeFile(m4a, { rate: 24000, aac: codec });
    assert.deepEqual(calls[0], ['decoder', [0x14, 0x08]]);
    assert.deepEqual(
      calls.filter(([what]) => what === 'frame').map(([, n]) => n),
      frames.map((f) => f.length)
    );
    calls.length = 0;
    const tone = Float32Array.from({ length: 5000 }, (_, i) => 0.5 * Math.sin(i * 0.05));
    const adts = await encodeWith(await ENCODERS.aac.open({ aac: codec }), tone);
    assert.deepEqual(calls[0], ['encoder', { sampleRate: 24000, channels: 1, bitrate: 64000 }]);
    const parsed = adtsFrames(adts);
    assert.deepEqual(parsed.config, { object: 2, index: 6, channels: 1, rate: 24000 });
    assert.equal(parsed.frames.length, 5);
    assert.ok(parsed.frames.every((f) => f.length === 2 * FRAME));
    calls.length = 0;
    const back = await decodeFile(adts, { rate: 24000, type: 'audio/aac', aac: codec });
    assert.deepEqual(calls[0], ['decoder', [0x13, 0x08]]);
    assert.equal(back.length, 5 * FRAME);
    assert.ok(similarity(back.subarray(0, tone.length), tone) > 0.9999);
  });

  test('WebCodecs goes first when it supports the config, and falls through when not', async (t) => {
    const log = [];
    const answers = { decode: true, encode: true };
    class AudioData {
      constructor(init) {
        Object.assign(this, init);
      }
      copyTo(target, { planeIndex }) {
        target.set(this.planes[planeIndex]);
      }
      close() {
        log.push('data closed');
      }
    }
    class EncodedAudioChunk {
      constructor({ data }) {
        this.data = data;
      }
    }
    class AudioDecoder {
      static isConfigSupported = async (config) => {
        log.push(['decode supported?', config.codec, config.sampleRate, config.numberOfChannels]);
        return { supported: answers.decode };
      };
      constructor({ output }) {
        this.output = output;
        this.state = 'configured';
      }
      configure() {}
      decode(chunk) {
        const planes = [new Float32Array(FRAME).fill(chunk.data.length / 1000)];
        this.output(
          new AudioData({ sampleRate: 16000, numberOfChannels: 1, numberOfFrames: FRAME, planes })
        );
      }
      flush = async () => {};
      close() {
        this.state = 'closed';
      }
    }
    class AudioEncoder {
      static isConfigSupported = async (config) => {
        log.push(['encode supported?', config.codec, config.sampleRate, config.bitrate]);
        return { supported: answers.encode && [44100, 48000].includes(config.sampleRate) };
      };
      constructor({ output }) {
        this.output = output;
        this.state = 'configured';
      }
      configure() {}
      encode(data) {
        const bytes = Uint8Array.of(data.numberOfFrames & 0xff, data.numberOfFrames >> 8);
        this.output({ byteLength: 2, copyTo: (target) => target.set(bytes) });
      }
      flush = async () => {};
      close() {
        this.state = 'closed';
      }
    }
    const stubs = { AudioData, EncodedAudioChunk, AudioDecoder, AudioEncoder };
    Object.assign(globalThis, stubs);
    t.after(() => {
      for (const name of Object.keys(stubs)) delete globalThis[name];
    });
    const codec = createAacCodec({ load: only(...ALL_PEERS) });
    assert.deepEqual(await codec.providers('decode'), ['webcodecs', '@audio/decode-aac']);
    assert.deepEqual(await codec.providers('encode'), ['webcodecs', '@audio/encode-aac']);
    const m4a = await fixture('m4a');
    const FRAME_COUNT = mp4Audio(m4a).frames.length;
    const web = await decodeFile(m4a, { rate: 16000, aac: codec });
    assert.deepEqual(log[0], ['decode supported?', 'mp4a.40.2', 44100, 2]);
    assert.deepEqual(log.at(-FRAME_COUNT - 1), ['decode supported?', 'mp4a.40.2', 16000, 1]);
    assert.equal(log.filter((l) => l === 'data closed').length, mp4Audio(m4a).frames.length);
    const middle = web.length >> 1;
    const frame = mp4Audio(m4a).frames[Math.floor((middle + 2112) / FRAME)];
    assert.ok(Math.abs(web[middle] - frame.length / 1000) < 1e-6);
    const open = await ENCODERS.aac.open({ aac: codec });
    const tone = Float32Array.from({ length: 3000 }, (_, i) => Math.sin(i / 10));
    const asked = log.length;
    const stub = adtsFrames(await encodeWith(open, tone));
    assert.deepEqual(
      log.slice(asked).filter((l) => l[0] === 'encode supported?'),
      [
        ['encode supported?', 'mp4a.40.2', 24000, 64000],
        ['encode supported?', 'mp4a.40.2', 24000, 96000],
        ['encode supported?', 'mp4a.40.2', 48000, 64000],
      ]
    );
    assert.equal(stub.config.rate, 48000);
    assert.equal(
      stub.frames.reduce((n, f) => n + f[0] + f[1] * 256, 0),
      6000
    );
    answers.decode = false;
    answers.encode = false;
    const { samples, rate } = decodeWav(await foxWav());
    const peer = await decodeFile(m4a, { rate, aac: codec });
    assert.ok(similarity(peer, samples) > 0.99);
    const fdk = adtsFrames(await encodeWith(open, await fox24()));
    assert.ok(fdk.frames.length > 60);
  });

  test('without WebCodecs or a peer, AAC answers 400 and the other formats still work', async () => {
    const aac = none();
    for (const [ext, type] of [
      ['m4a', 'audio/mp4'],
      ['aac', 'audio/aac'],
    ])
      await assert.rejects(
        decodeFile(await fixture(ext), { type, aac }),
        (error) =>
          error.name === 'AudioFormatError' &&
          /AAC input needs an AAC codec: this runtime has no WebCodecs AudioDecoder for mp4a\.40\.2, so install @audio\/decode-aac next to ortllama/.test(
            error.message
          )
      );
    await assert.rejects(
      ENCODERS.aac.open({ aac }),
      /aac output needs an AAC codec: .* install @audio\/encode-aac, or mediabunny with @mediabunny\/aac-encoder next to ortllama \(npm install @audio\/encode-aac; pnpm add -g @audio\/encode-aac in slicc-kernel\)/
    );
    for (const ext of ['wav', 'mp3', 'flac', 'opus', 'ogg', 'webm'])
      assert.ok((await decodeFile(await fixture(ext), { aac })).length > 40000, ext);
  });

  test('a host without the peers answers 400 through the default resolution', async () => {
    const script = `
import { registerHooks } from 'node:module';
const blocked = ${JSON.stringify(ALL_PEERS)};
registerHooks({
  resolve(specifier, context, next) {
    if (blocked.includes(specifier)) throw Object.assign(new Error("Cannot find package '" + specifier + "'"), { code: 'ERR_MODULE_NOT_FOUND' });
    return next(specifier, context);
  },
});
const { decodeFile } = await import(${JSON.stringify(new URL('src/audio/decode.js', ROOT).href)});
const { ENCODERS } = await import(${JSON.stringify(new URL('src/audio/encode.js', ROOT).href)});
const { readFileSync } = await import('node:fs');
const m4a = new Uint8Array(readFileSync(${JSON.stringify(new URL('test/integration/fixtures/fox.m4a', ROOT).pathname)}));
const wav = new Uint8Array(readFileSync(${JSON.stringify(new URL('test/integration/fixtures/fox.wav', ROOT).pathname)}));
const errors = [];
await decodeFile(m4a).catch((e) => errors.push(e.name + ': ' + e.message));
await ENCODERS.aac.open().catch((e) => errors.push(e.name + ': ' + e.message));
console.log(JSON.stringify({ errors, wav: (await decodeFile(wav)).length }));
`;
    const home = new URL('test/integration/fixtures/', ROOT).pathname;
    const child = spawnSync(process.execPath, ['--input-type=module', '-e', script], {
      encoding: 'utf8',
      env: { ...process.env, HOME: home, PNPM_HOME: home, ORTLLAMA_MODULE_PATH: '' },
    });
    assert.equal(child.status, 0, child.stderr);
    const { errors, wav } = JSON.parse(child.stdout);
    assert.equal(wav, 42732);
    assert.match(errors[0], /^AudioFormatError: AAC input needs an AAC codec/);
    assert.match(errors[1], /^AudioFormatError: aac output needs an AAC codec/);
  });
});

describe('AAC over HTTP', () => {
  const servers = {};
  const engine = (capability, instance) => ({
    key: `test/${capability}`,
    id: `test/${capability}`,
    variant: 'fp32',
    info: {},
    engine: { name: capability, capability, load: async () => instance },
  });
  const listener = {
    async transcribe(audio) {
      return { text: String(audio.length), language: 'english', duration: 1, segments: [] };
    },
  };
  const speaker = {
    sampleRate: 24000,
    tokens: async (text) => [[text.length]],
    synthesize: async () => Float32Array.from({ length: 4800 }, (_, i) => 0.4 * Math.sin(i / 7)),
  };

  const encoders = [];
  const peers = createAacCodec({ scope: {}, load: only(...ALL_PEERS) });
  const tracked = {
    require: (role) => peers.require(role),
    decoder: (config) => peers.decoder(config),
    async encoder(config) {
      const inner = await peers.encoder(config);
      const seen = { flushed: false, closed: false };
      encoders.push(seen);
      return {
        sampleRate: inner.sampleRate,
        encode: (input) => inner.encode(input),
        flush() {
          seen.flushed = true;
          return inner.flush();
        },
        close() {
          seen.closed = true;
          inner.close();
        },
      };
    },
  };

  before(async () => {
    const resolved = {
      transcribe: engine('transcription', listener),
      speech: engine('speech', speaker),
    };
    for (const [name, aac] of [
      ['none', none()],
      ['peers', peers],
      ['tracked', tracked],
    ]) {
      const catalog = { aliases: {}, list: async () => [], resolve: async (m) => resolved[m] };
      const runtime = createRuntime({
        env: {},
        backend: async () => ({ track: 'node', executionProviders: ['cpu'] }),
      });
      servers[name] = await startServer({ env: {}, port: 0, catalog, runtime, aac });
    }
  });
  after(async () => {
    for (const server of Object.values(servers)) await server.close();
  });

  const transcribe = async (server, ext) => {
    const form = new FormData();
    form.set('file', new Blob([await fixture(ext)]), `clip.${ext}`);
    form.set('model', 'transcribe');
    form.set('response_format', 'text');
    return fetch(`${server.url}/v1/audio/transcriptions`, { method: 'POST', body: form });
  };
  const speak = (server, format) =>
    fetch(`${server.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'speech', input: 'One. Two.', response_format: format }),
    });

  test('without a codec, AAC input and aac output answer 400 naming the peers', async () => {
    for (const ext of ['m4a', 'aac']) {
      const response = await transcribe(servers.none, ext);
      assert.equal(response.status, 400, ext);
      const { error } = await response.json();
      assert.equal(error.param, 'file');
      assert.match(error.message, /install @audio\/decode-aac/);
    }
    for (const ext of ['wav', 'mp3', 'flac', 'opus'])
      assert.equal((await transcribe(servers.none, ext)).status, 200, ext);
    const aac = await speak(servers.none, 'aac');
    assert.equal(aac.status, 400);
    const { error } = await aac.json();
    assert.equal(error.param, 'response_format');
    assert.match(
      error.message,
      /install @audio\/encode-aac, or mediabunny with @mediabunny\/aac-encoder/
    );
    for (const format of ['wav', 'mp3', 'opus', 'flac'])
      assert.equal((await speak(servers.none, format)).status, 200, format);
  });

  test('with the peers, AAC input transcribes and aac output streams ADTS', async () => {
    const m4a = await transcribe(servers.peers, 'm4a');
    assert.equal(m4a.status, 200);
    assert.equal((await m4a.text()).trim(), '42732');
    const response = await speak(servers.peers, 'aac');
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'audio/aac');
    const bytes = new Uint8Array(await response.arrayBuffer());
    const { config, frames } = adtsFrames(bytes);
    assert.deepEqual(config, { object: 2, index: 6, channels: 1, rate: 24000 });
    assert.ok(frames.length >= 9, `${frames.length} frames`);
  });

  test('a client that hangs up mid-stream closes the AAC encoder', async () => {
    const controller = new AbortController();
    const response = await fetch(`${servers.tracked.url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'speech',
        input: 'One more sentence. '.repeat(200),
        response_format: 'aac',
      }),
      signal: controller.signal,
    });
    assert.equal(response.status, 200);
    await response.body.getReader().read();
    controller.abort();
    const until = Date.now() + 5000;
    while (!encoders[0]?.closed && Date.now() < until)
      await new Promise((resolve) => setTimeout(resolve, 20));
    assert.deepEqual(encoders, [{ flushed: false, closed: true }]);
  });
});

const ffmpeg = spawnSync('ffmpeg', ['-version']).status === 0;

describe('the real AAC peers', () => {
  for (const set of PEERS.encode) {
    test(`${set.join(' with ')} encodes speech that ffmpeg and the decoder read back`, async (t) => {
      const codec = createAacCodec({ scope: {}, load: only(...set, '@audio/decode-aac') });
      assert.deepEqual(await codec.providers('encode'), [set[0]]);
      const fox = await fox24();
      const adts = await encodeWith(await ENCODERS.aac.open({ aac: codec }), fox);
      const { frames } = readAdts(adts);
      assert.ok(Math.abs(frames.length - fox.length / FRAME) < 4, `${frames.length} frames`);
      const decoded = await decodeFile(adts, { rate: 24000, type: 'audio/aac', aac: codec });
      const ours = snr(fox, decoded);
      t.diagnostic(`${adts.length} bytes, ${ours.toFixed(1)} dB`);
      assert.ok(ours > 15, `${ours} dB`);
      if (!ffmpeg) return;
      const dir = await mkdtemp(join(tmpdir(), 'ortllama-aac-'));
      try {
        const file = join(dir, 'fox.aac');
        await writeFile(file, adts);
        const probe = spawnSync('ffprobe', ['-v', 'error', '-show_streams', '-of', 'json', file], {
          encoding: 'utf8',
        });
        const [stream] = JSON.parse(probe.stdout).streams;
        assert.equal(stream.codec_name, 'aac');
        assert.equal(stream.profile, 'LC');
        assert.equal(stream.sample_rate, '24000');
        assert.equal(stream.channels, 1);
        const raw = spawnSync('ffmpeg', ['-v', 'error', '-i', file, '-f', 'f32le', '-']).stdout;
        const theirs = snr(fox, new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4));
        t.diagnostic(`ffmpeg reads it at ${theirs.toFixed(1)} dB`);
        assert.ok(theirs > 15, `${theirs} dB`);
      } finally {
        await rm(dir, { recursive: true });
      }
    });
  }
});

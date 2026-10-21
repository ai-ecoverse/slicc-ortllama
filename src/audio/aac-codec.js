import { optional } from '../engines/optional.js';
import { audioSpecificConfig, readAdts } from './adts.js';
import { AudioFormatError, concat, createResampler } from './pcm.js';

export const CODEC = 'mp4a.40.2';

export const PEERS = {
  decode: [['@audio/decode-aac']],
  encode: [['@audio/encode-aac'], ['mediabunny', '@mediabunny/aac-encoder']],
};

const SBR_DELAY = 962;
const WINDOWS_BITRATE = 96000;

const names = (role) => PEERS[role].map((set) => set.join(' with ')).join(', or ');

export function unavailable(role) {
  const what = role === 'decode' ? 'AAC input' : 'aac output';
  const api = role === 'decode' ? 'AudioDecoder' : 'AudioEncoder';
  const install = PEERS[role][0].join(' ');
  return new AudioFormatError(
    `${what} needs an AAC codec: this runtime has no WebCodecs ${api} for ${CODEC}, so install ${names(role)} next to ortllama (npm install ${install}; pnpm add -g ${install} in slicc-kernel), or add its project folder to ORTLLAMA_MODULE_PATH`
  );
}

function merge(sets) {
  if (sets.length <= 1) return sets[0] ?? [];
  return sets[0].map((_, c) =>
    concat(
      sets.map((set) => set[c]),
      Float32Array
    )
  );
}

function planes(data) {
  const out = [];
  for (let c = 0; c < data.numberOfChannels; c++) {
    const plane = new Float32Array(data.numberOfFrames);
    data.copyTo(plane, { planeIndex: c, format: 'f32-planar' });
    out.push(plane);
  }
  data.close();
  return out;
}

function planar(input, length) {
  const data = new Float32Array(length * input.length);
  for (const [c, plane] of input.entries()) data.set(plane, c * length);
  return data;
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

async function held(promise) {
  const keep = setInterval(() => {}, 1000);
  try {
    return await promise;
  } finally {
    clearInterval(keep);
  }
}

async function supported(api, config) {
  if (typeof api?.isConfigSupported !== 'function') return false;
  try {
    return Boolean((await api.isConfigSupported(config)).supported);
  } catch {
    return false;
  }
}

async function first(api, configs) {
  for (const config of configs) if (await supported(api, config)) return config;
  return null;
}

const zeros = (channels, length) =>
  Array.from({ length: channels }, () => new Float32Array(length));

function timeline(core, offset) {
  let frames = 0;
  let emitted = 0;
  let rate = 0;
  let channels = 0;
  let width = 1024;
  let shift = 0;
  return {
    get rate() {
      return rate;
    },
    count() {
      frames++;
    },
    place(input, sampleRate) {
      if (!input[0]?.length) return [];
      if (!rate) {
        rate = sampleRate;
        channels = input.length;
        const sbr = rate > core * 1.5;
        width = sbr ? 2048 : 1024;
        shift = offset(sbr);
      }
      let out = input;
      if (shift > 0) out = merge([zeros(channels, shift), input]);
      const cut = Math.min(Math.max(0, -shift), input[0].length);
      if (cut) out = input.map((plane) => plane.subarray(cut));
      shift = shift > 0 ? 0 : shift + cut;
      emitted += out[0].length;
      return out[0].length ? out : [];
    },
    tail() {
      const length = frames * width - emitted;
      return rate && length > 0 ? zeros(channels, length) : [];
    },
  };
}

function webDecoder(scope, config, core) {
  const sets = [];
  const line = timeline(core, (sbr) => (sbr ? -SBR_DELAY : 0));
  let failure = null;
  let sampleRate = core;
  let timestamp = 0;
  const decoder = new scope.AudioDecoder({
    output: (data) => {
      sampleRate = data.sampleRate;
      sets.push(planes(data));
    },
    error: (error) => {
      failure = error;
    },
  });
  decoder.configure(config);
  const take = () => {
    if (failure) throw failure;
    return line.place(merge(sets.splice(0)), sampleRate);
  };
  return {
    get sampleRate() {
      return line.rate || sampleRate;
    },
    async decode(frame) {
      line.count();
      decoder.decode(new scope.EncodedAudioChunk({ type: 'key', timestamp, data: frame }));
      timestamp += Math.round((1024 / core) * 1e6);
      await tick();
      return take();
    },
    async flush() {
      if (!failure && decoder.state !== 'closed') await held(decoder.flush());
      if (decoder.state !== 'closed') decoder.close();
      const rest = take();
      const tail = line.tail();
      return rest[0]?.length ? merge([rest, tail].filter((set) => set.length)) : tail;
    },
  };
}

function webEncoder(scope, config, from) {
  const { sampleRate, numberOfChannels: channels } = config;
  const resamplers = Array.from({ length: channels }, () => createResampler(from, sampleRate));
  const frames = [];
  let failure = null;
  let timestamp = 0;
  const encoder = new scope.AudioEncoder({
    output: (chunk) => {
      const frame = new Uint8Array(chunk.byteLength);
      chunk.copyTo(frame);
      frames.push(frame);
    },
    error: (error) => {
      failure = error;
    },
  });
  encoder.configure(config);
  const take = () => {
    if (failure) throw failure;
    return frames.splice(0);
  };
  const send = (input) => {
    const length = input[0].length;
    if (!length || failure) return;
    const audio = new scope.AudioData({
      format: 'f32-planar',
      sampleRate,
      numberOfFrames: length,
      numberOfChannels: channels,
      timestamp: Math.round(timestamp),
      data: planar(input, length),
    });
    encoder.encode(audio);
    audio.close();
    timestamp += (length / sampleRate) * 1e6;
  };
  return {
    sampleRate,
    async encode(input) {
      send(input.map((plane, c) => resamplers[c](plane)));
      await tick();
      return take();
    },
    async flush() {
      send(resamplers.map((resample) => resample.end()));
      if (!failure && encoder.state !== 'closed') await held(encoder.flush());
      if (encoder.state !== 'closed') encoder.close();
      return take();
    },
  };
}

export const PROBES = {
  decode: [{ codec: CODEC, sampleRate: 44100, numberOfChannels: 2 }],
  encode: [48000, 44100].map((sampleRate) => ({
    codec: CODEC,
    sampleRate,
    numberOfChannels: 1,
    bitrate: 128000,
  })),
};

export function webcodecs(scope) {
  return {
    name: 'webcodecs',
    async decoder({ asc }) {
      const { rate, channels } = audioSpecificConfig(asc);
      const config = {
        codec: CODEC,
        sampleRate: rate,
        numberOfChannels: channels,
        description: asc,
      };
      if (!(await supported(scope.AudioDecoder, config))) return null;
      return webDecoder(scope, config, rate);
    },
    async encoder({ sampleRate, channels, bitrate }) {
      const rates = [...new Set([sampleRate, sampleRate * 2, 48000, 44100])];
      const rated = [...new Set([bitrate, WINDOWS_BITRATE])];
      const configs = rates.flatMap((rate) =>
        rated.map((bits) => ({
          codec: CODEC,
          sampleRate: rate,
          numberOfChannels: channels,
          bitrate: bits,
        }))
      );
      const config = await first(scope.AudioEncoder, configs);
      return config && webEncoder(scope, config, sampleRate);
    },
  };
}

export function faad([library]) {
  return {
    name: '@audio/decode-aac',
    async decoder({ asc }) {
      const line = timeline(audioSpecificConfig(asc).rate, (sbr) =>
        sbr ? 2048 - SBR_DELAY : 1024
      );
      const decoder = await library.decoder({ asc });
      return {
        get sampleRate() {
          return line.rate;
        },
        async decode(frame) {
          line.count();
          const { channelData, sampleRate } = decoder.decode(frame);
          return line.place(channelData, sampleRate);
        },
        async flush() {
          decoder.free();
          return line.tail();
        },
      };
    },
  };
}

const unwrap = (bytes) => (bytes.length ? readAdts(bytes).frames : []);

export function fdk([library]) {
  return {
    name: '@audio/encode-aac',
    async encoder({ sampleRate, channels, bitrate }) {
      const encoder = await library.default({
        sampleRate,
        channels,
        bitrate: bitrate / 1000,
        profile: 'lc',
      });
      return {
        sampleRate,
        encode: async (input) => unwrap(await encoder.encode(input)),
        flush: async () => unwrap(await encoder.flush()),
      };
    },
  };
}

export function bunny([mediabunny, extension]) {
  let registered = false;
  return {
    name: 'mediabunny',
    async encoder({ sampleRate, channels, bitrate }) {
      if (!registered) extension.registerAacEncoder();
      registered = true;
      let pending = new Uint8Array(0);
      const output = new mediabunny.Output({
        format: new mediabunny.AdtsOutputFormat(),
        target: new mediabunny.AppendOnlyStreamTarget(
          new WritableStream({
            write: (chunk) => {
              pending = concat([pending, chunk]);
            },
          })
        ),
      });
      const source = new mediabunny.AudioSampleSource({ codec: 'aac', bitrate });
      output.addAudioTrack(source);
      await output.start();
      let timestamp = 0;
      const take = () => {
        const { frames, rest } = readAdts(pending);
        pending = rest.slice();
        return frames.map((frame) => frame.slice());
      };
      return {
        sampleRate,
        async encode(input) {
          const length = input[0].length;
          if (length) {
            const data = planar(input, length);
            const sample = new mediabunny.AudioSample({
              data,
              format: 'f32-planar',
              numberOfChannels: channels,
              sampleRate,
              timestamp,
            });
            await source.add(sample);
            sample.close();
            timestamp += length / sampleRate;
          }
          return take();
        },
        async flush() {
          await output.finalize();
          return take();
        },
      };
    },
  };
}

const ADAPTERS = { '@audio/decode-aac': faad, '@audio/encode-aac': fdk, mediabunny: bunny };

async function peer(load, set) {
  try {
    return ADAPTERS[set[0]](await Promise.all(set.map((name) => load(name))));
  } catch (error) {
    if (error?.code === 'engine_unavailable') return null;
    throw error;
  }
}

async function providers(role, scope, load) {
  const api = role === 'decode' ? scope.AudioDecoder : scope.AudioEncoder;
  const found = (await first(api, PROBES[role])) ? [webcodecs(scope)] : [];
  for (const set of PEERS[role]) {
    const adapter = await peer(load, set);
    if (adapter) return [...found, adapter];
  }
  return found;
}

export function createAacCodec({ scope = globalThis, load = optional } = {}) {
  const roles = new Map();
  const chain = (role) => {
    if (!roles.has(role))
      roles.set(
        role,
        providers(role, scope, load).catch((error) => {
          roles.delete(role);
          throw error;
        })
      );
    return roles.get(role);
  };
  const open = async (role, method, config) => {
    for (const provider of await chain(role)) {
      const made = await provider[method](config);
      if (made) return made;
    }
    throw unavailable(role);
  };
  return {
    async providers(role) {
      return (await chain(role)).map((provider) => provider.name);
    },
    async require(role) {
      if ((await chain(role)).length === 0) throw unavailable(role);
    },
    encoder: (config) => open('encode', 'encoder', config),
    decoder: (config) => open('decode', 'decoder', config),
  };
}

let shared = null;

export function aacCodec() {
  shared ??= createAacCodec();
  return shared;
}

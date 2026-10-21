import { aacCodec } from './aac-codec.js';
import { adtsFrame, RATES } from './adts.js';
import { createFlacEncoder } from './flac.js';
import { createMp3Encoder } from './mp3-encoder.js';
import { opusEncoder } from './opus.js';
import { AudioFormatError, concat, encodePcm16, wavHeader } from './pcm.js';

const AAC_BITRATE = 64000;

function pcmEncoder() {
  return { push: encodePcm16, end: () => new Uint8Array(0) };
}

function wavEncoder(rate) {
  let header = wavHeader(rate);
  const take = () => {
    const out = header;
    header = new Uint8Array(0);
    return out;
  };
  return {
    push: (samples) => concat([take(), encodePcm16(samples)]),
    end: take,
  };
}

export async function aacOutput(aac = aacCodec()) {
  await aac.require('encode');
  return async (rate) => {
    if (!RATES.includes(rate)) throw new AudioFormatError(`aac output does not support ${rate} Hz`);
    const encoder = await aac.encoder({ sampleRate: rate, channels: 1, bitrate: AAC_BITRATE });
    const index = RATES.indexOf(encoder.sampleRate ?? rate);
    const wrap = (frames) =>
      concat(frames.map((f) => adtsFrame(f, { object: 2, index, channels: 1 })));
    return {
      push: async (samples) => wrap(await encoder.encode([samples])),
      end: async () => wrap(await encoder.flush()),
    };
  };
}

export const ENCODERS = {
  wav: { type: 'audio/wav', open: async () => wavEncoder },
  pcm: { type: 'audio/pcm', open: async () => pcmEncoder },
  mp3: { type: 'audio/mpeg', open: async () => createMp3Encoder },
  opus: { type: 'audio/opus', open: () => opusEncoder() },
  flac: { type: 'audio/flac', open: async () => createFlacEncoder },
  aac: { type: 'audio/aac', open: ({ aac } = {}) => aacOutput(aac) },
};

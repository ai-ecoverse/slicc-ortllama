import { createFlacEncoder } from './flac.js';
import { createMp3Encoder } from './mp3-encoder.js';
import { opusEncoder } from './opus.js';
import { concat, encodePcm16, wavHeader } from './pcm.js';

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

export const ENCODERS = {
  wav: { type: 'audio/wav', open: async () => wavEncoder },
  pcm: { type: 'audio/pcm', open: async () => pcmEncoder },
  mp3: { type: 'audio/mpeg', open: async () => createMp3Encoder },
  opus: { type: 'audio/opus', open: () => opusEncoder() },
  flac: { type: 'audio/flac', open: async () => createFlacEncoder },
};

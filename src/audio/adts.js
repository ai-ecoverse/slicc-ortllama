import { AudioFormatError } from './pcm.js';

export const RATES = [
  96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];
const TRAILERS = ['TAG', 'ID3', 'APETAGEX'];

function corrupt() {
  return new AudioFormatError('the AAC stream is corrupt');
}

function reader(bytes) {
  let pos = 0;
  return (n) => {
    let v = 0;
    for (let i = 0; i < n; i++, pos++) {
      if (pos >= bytes.length * 8) throw corrupt();
      v = v * 2 + ((bytes[pos >> 3] >> (7 - (pos & 7))) & 1);
    }
    return v;
  };
}

export function audioSpecificConfig(bytes) {
  const read = reader(bytes);
  const object = () => {
    const o = read(5);
    return o === 31 ? 32 + read(6) : o;
  };
  const rate = () => {
    const index = read(4);
    return index === 15 ? read(24) : (RATES[index] ?? 0);
  };
  const type = object();
  const core = rate();
  const channels = read(4);
  if (!core) throw corrupt();
  return { object: type, index: RATES.indexOf(core), rate: core, channels };
}

export function ascBytes({ object, index, channels }) {
  return Uint8Array.of((object << 3) | (index >> 1), ((index & 1) << 7) | (channels << 3));
}

const sync = (bytes, at) => bytes[at] === 0xff && (bytes[at + 1] & 0xf6) === 0xf0;

const frameLength = (b, at) => ((b[at + 3] & 3) << 11) | (b[at + 4] << 3) | (b[at + 5] >> 5);

export function isAdts(bytes) {
  if (bytes.length < 7 || !sync(bytes, 0) || ((bytes[2] >> 2) & 15) > 12) return false;
  const length = frameLength(bytes, 0);
  return length >= 7 && (length === bytes.length || sync(bytes, length));
}

const trailer = (bytes) =>
  TRAILERS.some((tag) => String.fromCharCode(...bytes.subarray(0, tag.length)) === tag);

function header(bytes, at) {
  return {
    object: (bytes[at + 2] >> 6) + 1,
    index: (bytes[at + 2] >> 2) & 15,
    channels: ((bytes[at + 2] & 1) << 2) | (bytes[at + 3] >> 6),
  };
}

export function readAdts(bytes) {
  const frames = [];
  let config = null;
  let at = 0;
  while (at + 7 <= bytes.length && sync(bytes, at)) {
    const size = bytes[at + 1] & 1 ? 7 : 9;
    const length = frameLength(bytes, at);
    const found = header(bytes, at);
    if (length < size || found.index > 12) throw corrupt();
    if (at + length > bytes.length) break;
    if ((bytes[at + 6] & 3) > 0)
      throw new AudioFormatError('ADTS frames with several raw data blocks are not supported');
    config ??= found;
    if (found.index !== config.index) throw corrupt();
    frames.push(bytes.subarray(at + size, at + length));
    at += length;
  }
  return { config, frames, rest: bytes.subarray(at) };
}

export function adtsFrames(bytes) {
  const { config, frames, rest } = readAdts(bytes);
  if (!config || (rest.length && !trailer(rest))) throw corrupt();
  return { config: { ...config, rate: RATES[config.index] }, frames };
}

export function adtsFrame(payload, { object, index, channels }) {
  const length = payload.length + 7;
  const out = new Uint8Array(length);
  out.set([
    0xff,
    0xf1,
    ((object - 1) << 6) | (index << 2) | (channels >> 2),
    ((channels & 3) << 6) | (length >> 11),
    (length >> 3) & 0xff,
    ((length & 7) << 5) | 0x1f,
    0xfc,
  ]);
  out.set(payload, 7);
  return out;
}

import { readFile } from 'node:fs/promises';
import { optional } from '../engines/optional.js';
import { encodePcm16 } from './pcm.js';

const STEP_SECONDS = 60;
const BITRATE = 64;
const DECODER_DELAY = 529;
const LAME = new Set(['LAME', 'Lavf', 'Lavc']);
const TAGS = new Set(['Xing', 'Info']);
const KBPS = [
  [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
  [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];
const RATES = [44100, 48000, 32000];
const SIDE = [
  [17, 32],
  [9, 17],
];

let compiled;

export function id3Size(bytes) {
  if (bytes.length < 10 || bytes[0] !== 0x49 || bytes[1] !== 0x44 || bytes[2] !== 0x33) return 0;
  const size = ((bytes[6] & 0x7f) << 21) | ((bytes[7] & 0x7f) << 14) | ((bytes[8] & 0x7f) << 7);
  return 10 + (size | (bytes[9] & 0x7f)) + (bytes[5] & 0x10 ? 10 : 0);
}

export function isMp3(bytes) {
  if (bytes.length < 4) return false;
  if (bytes[0] === 0x49 && bytes[1] === 0x44 && bytes[2] === 0x33) return true;
  return bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0 && (bytes[1] & 0x06) !== 0;
}

const text = (bytes, at) => String.fromCharCode(...bytes.subarray(at, at + 4));

export function gapless(bytes) {
  const at = id3Size(bytes);
  const plain = { body: bytes, skip: 0, keep: Infinity };
  const [sync, b1, b2, b3] = bytes.subarray(at, at + 4);
  const version = (b1 >> 3) & 3;
  const index = b2 >> 4;
  const rateIndex = (b2 >> 2) & 3;
  if (sync !== 0xff || (b1 & 0xe6) !== 0xe2 || version === 1 || index % 15 === 0 || rateIndex === 3)
    return plain;
  const mpeg1 = version === 3;
  const spf = mpeg1 ? 1152 : 576;
  const rate = RATES[rateIndex] / (mpeg1 ? 1 : 4 - version);
  const end = at + Math.floor((spf * KBPS[mpeg1 ? 0 : 1][index] * 125) / rate) + ((b2 >> 1) & 1);
  const side = at + 4 + SIDE[mpeg1 ? 0 : 1][b3 >> 6 === 3 ? 0 : 1];
  const tag = b1 & 1 || TAGS.has(text(bytes, side)) ? side : side + 2;
  if (end > bytes.length || tag + 8 > end || !TAGS.has(text(bytes, tag))) return plain;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const flags = view.getUint32(tag + 4);
  const frames = flags & 1 && tag + 12 <= end ? view.getUint32(tag + 8) : 0;
  const lame = [4, 4, 100, 4].reduce((n, size, i) => n + (flags & (1 << i) ? size : 0), tag + 8);
  const body = bytes.subarray(end);
  if (lame + 24 > end || !LAME.has(text(bytes, lame))) return { ...plain, body };
  const delay = (bytes[lame + 21] << 4) | (bytes[lame + 22] >> 4);
  const padding = ((bytes[lame + 22] & 15) << 8) | bytes[lame + 23];
  return {
    body,
    skip: delay + DECODER_DELAY,
    keep: frames ? frames * spf - delay - padding : Infinity,
  };
}

async function decoderModule({
  resolve = (specifier) => import.meta.resolve(specifier),
  readBinary = readFile,
} = {}) {
  compiled ??= (async () => {
    const bytes = await readBinary(new URL('decoder.opt.wasm', resolve('minimp3-wasm')));
    return WebAssembly.compile(bytes);
  })().catch((error) => {
    compiled = null;
    throw error;
  });
  return compiled;
}

export async function* decodeMp3(bytes, options) {
  const { Decoder } = await import('minimp3-wasm');
  const instance = await WebAssembly.instantiate(await decoderModule(options), {});
  const { body, skip, keep } = gapless(bytes);
  const decoder = new Decoder(instance.exports, body);
  for (let at = 0; ; ) {
    const { pcm, numChannels: channels, samplingRate: rate } = decoder.decode(STEP_SECONDS);
    if (pcm.length === 0) return;
    const count = pcm.length / channels;
    const from = Math.min(count, Math.max(0, skip - at));
    const to = Math.max(from, Math.min(count, skip + keep - at));
    at += count;
    const samples = new Float32Array(to - from);
    for (let i = 0; i < samples.length; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) sum += pcm[(from + i) * channels + c];
      samples[i] = sum / channels / 32768;
    }
    yield { samples, rate };
  }
}

const copy = (out) => new Uint8Array(out.buffer, out.byteOffset, out.byteLength).slice();

export async function mp3Encoder(load = () => optional('@breezystack/lamejs')) {
  const { Mp3Encoder } = await load();
  return (rate) => {
    const encoder = new Mp3Encoder(1, rate, BITRATE);
    return {
      push: (samples) => copy(encoder.encodeBuffer(new Int16Array(encodePcm16(samples).buffer))),
      end: () => copy(encoder.flush()),
    };
  };
}

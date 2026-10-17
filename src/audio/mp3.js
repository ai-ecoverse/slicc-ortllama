import { readFile } from 'node:fs/promises';
import { optional } from '../engines/optional.js';
import { encodePcm16 } from './pcm.js';

const STEP_SECONDS = 60;
const BITRATE = 64;

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
  const decoder = new Decoder(instance.exports, bytes);
  for (;;) {
    const { pcm, numChannels: channels, samplingRate: rate } = decoder.decode(STEP_SECONDS);
    if (pcm.length === 0) return;
    const samples = new Float32Array(pcm.length / channels);
    for (let i = 0; i < samples.length; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) sum += pcm[i * channels + c];
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

import { readFile } from 'node:fs/promises';
import { decodeWav, resample } from '../../src/audio/pcm.js';

export const FOX = /the quick brown fox jumps? over the lazy dog/;

export async function foxWav() {
  return fixture('wav');
}

export async function fixture(ext, name = 'fox') {
  return new Uint8Array(await readFile(new URL(`fixtures/${name}.${ext}`, import.meta.url)));
}

export function pcm24(wav, { before = 0, after = 0, endOnSpeech = false } = {}) {
  const { samples, rate } = decodeWav(wav);
  let speech = resample(samples, rate, 24000);
  if (endOnSpeech) {
    let end = speech.length;
    while (end > 0 && Math.abs(speech[end - 1]) < 0.02) end--;
    speech = speech.subarray(0, end);
  }
  const lead = Math.round(before * 24000);
  const out = new Float32Array(lead + speech.length + Math.round(after * 24000));
  out.set(speech, lead);
  const bytes = new Uint8Array(out.length * 2);
  const view = new DataView(bytes.buffer);
  out.forEach((x, i) => {
    view.setInt16(i * 2, Math.max(-32768, Math.min(32767, Math.round(x * 32768))), true);
  });
  return bytes;
}

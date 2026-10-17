import { decodeFlac, isFlac } from './flac.js';
import { decodeMp3, isMp3 } from './mp3.js';
import { isOgg, oggPackets } from './ogg.js';
import { decodeOpus, opusRate } from './opus.js';
import {
  AudioFormatError,
  audioParams,
  concat,
  createResampler,
  decodeAudio,
  decodeWav,
  isWav,
  PCM_TYPES,
} from './pcm.js';
import { isWebm, webmAudio } from './webm.js';

export const MAX_SECONDS = 1800;

const SUPPORTED = 'send wav, mp3, flac, ogg or webm with opus, or 16-bit PCM (audio/pcm;rate=N)';

const TYPES = {
  mp3: ['audio/mpeg', 'audio/mp3', 'audio/mpga', 'audio/x-mp3', 'audio/x-mpeg'],
  flac: ['audio/flac', 'audio/x-flac'],
  ogg: ['audio/ogg', 'audio/opus', 'application/ogg'],
  webm: ['audio/webm', 'video/webm'],
};

const EXTENSIONS = {
  mp3: ['mp3', 'mpga', 'mpeg'],
  flac: ['flac'],
  ogg: ['ogg', 'oga', 'opus'],
  webm: ['webm'],
  wav: ['wav', 'wave'],
};

const find = (table, key) => Object.keys(table).find((f) => table[f].includes(key)) ?? null;

export function sniff(bytes, contentType = '', name = '') {
  if (isWav(bytes)) return 'wav';
  if (isFlac(bytes)) return 'flac';
  if (isOgg(bytes)) return 'ogg';
  if (isWebm(bytes)) return 'webm';
  if (isMp3(bytes)) return 'mp3';
  const { type } = audioParams(contentType);
  const ext = /\.([a-z0-9]+)$/i.exec(name)?.[1].toLowerCase() ?? '';
  return find(TYPES, type) ?? find(EXTENSIONS, ext);
}

function opusStream(head, packets, granule, rate) {
  const text = String.fromCharCode(...(head?.subarray(0, 8) ?? []));
  if (!text.startsWith('OpusHead'))
    throw new AudioFormatError(`only Opus is supported in Ogg and WebM; ${SUPPORTED}`);
  return decodeOpus({ head, packets, granule, rate: opusRate(rate) });
}

function chunks(bytes, format, contentType, rate) {
  if (format === 'mp3') return decodeMp3(bytes);
  if (format === 'flac') return decodeFlac(bytes);
  if (format === 'ogg') {
    const { packets, granule } = oggPackets(bytes);
    return opusStream(packets[0], packets.slice(2), granule, rate);
  }
  if (format === 'webm') {
    const { codec, head, packets } = webmAudio(bytes);
    if (codec !== 'A_OPUS')
      throw new AudioFormatError(`WebM codec ${codec} is not supported; ${SUPPORTED}`);
    return opusStream(head, packets, -1, rate);
  }
  if (format === 'wav') return [decodeWav(bytes)];
  return [decodeAudio(bytes, contentType)];
}

export async function decodeFile(
  bytes,
  { type = '', name = '', rate = 16000, maxSeconds = MAX_SECONDS } = {}
) {
  const format = sniff(bytes, type, name);
  const { type: mime } = audioParams(type);
  const label = mime || /\.[a-z0-9]+$/i.exec(name)?.[0] || 'unknown';
  if (!format && !PCM_TYPES.has(mime))
    throw new AudioFormatError(`audio type ${label} is not supported; ${SUPPORTED}`);
  const parts = [];
  let source = 0;
  let from = 0;
  let resampler = null;
  for await (const chunk of chunks(bytes, format, type, rate)) {
    if (!resampler) {
      from = chunk.rate;
      resampler = createResampler(from, rate);
    }
    source += chunk.samples.length;
    if (source / from > maxSeconds)
      throw new AudioFormatError(`the audio is longer than ${maxSeconds} seconds`);
    parts.push(resampler(chunk.samples));
  }
  if (resampler) parts.push(resampler.end());
  return concat(parts, Float32Array);
}

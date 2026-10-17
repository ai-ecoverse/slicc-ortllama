import { decodeFlac, isFlac } from './flac.js';
import { decodeMp3, id3Size, isMp3 } from './mp3.js';
import { isOgg, oggPackets } from './ogg.js';
import { decodeOpus, opusRate, packetSamples } from './opus.js';
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

const BUDGET_RATE = 48000;

const SUPPORTED = 'send wav, mp3, flac, ogg or webm with opus, or 16-bit PCM (audio/pcm;rate=N)';

const TYPES = {
  mp3: ['audio/mpeg', 'audio/mp3', 'audio/mpga', 'audio/x-mp3', 'audio/x-mpeg'],
  flac: ['audio/flac', 'audio/x-flac'],
  ogg: ['audio/ogg', 'audio/opus', 'application/ogg'],
  webm: ['audio/webm', 'video/webm'],
  wav: ['audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave'],
};

const EXTENSIONS = {
  mp3: ['mp3', 'mpga', 'mpeg'],
  flac: ['flac'],
  ogg: ['ogg', 'oga', 'opus'],
  webm: ['webm'],
  wav: ['wav', 'wave'],
};

const find = (table, key) => Object.keys(table).find((f) => table[f].includes(key)) ?? null;

function detect(bytes, contentType, name) {
  const tag = id3Size(bytes);
  const body = bytes.subarray(Math.min(tag, bytes.length));
  for (const [format, test] of [
    ['wav', isWav],
    ['flac', isFlac],
    ['ogg', isOgg],
    ['webm', isWebm],
  ])
    if (test(body)) return { format, body };
  const { type } = audioParams(contentType);
  const ext = /\.([a-z0-9]+)$/i.exec(name)?.[1].toLowerCase() ?? '';
  const labelled = find(TYPES, type) ?? find(EXTENSIONS, ext);
  if (tag > 0 || (isMp3(body) && (!labelled || labelled === 'mp3')))
    return { format: 'mp3', body: bytes };
  return { format: labelled, body: bytes };
}

export function sniff(bytes, contentType = '', name = '') {
  return detect(bytes, contentType, name).format;
}

function endGranule(packets, padding) {
  if (!padding) return -1;
  return Math.max(0, packets.reduce((n, p) => n + packetSamples(p), 0) - padding);
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
    const { codec, head, packets, padding } = webmAudio(bytes);
    if (codec !== 'A_OPUS')
      throw new AudioFormatError(`WebM codec ${codec} is not supported; ${SUPPORTED}`);
    return opusStream(head, packets, endGranule(packets, padding), rate);
  }
  if (format === 'wav') return [decodeWav(bytes)];
  return [decodeAudio(bytes, contentType)];
}

export async function decodeFile(
  bytes,
  { type = '', name = '', rate = 16000, maxSeconds = MAX_SECONDS } = {}
) {
  const { format, body } = detect(bytes, type, name);
  const { type: mime } = audioParams(type);
  const label = mime || /\.[a-z0-9]+$/i.exec(name)?.[0] || 'unknown';
  if (!format && !PCM_TYPES.has(mime))
    throw new AudioFormatError(`audio type ${label} is not supported; ${SUPPORTED}`);
  const parts = [];
  let source = 0;
  let from = 0;
  let resampler = null;
  for await (const chunk of chunks(body, format, type, rate)) {
    if (!resampler) {
      from = chunk.rate;
      resampler = createResampler(from, rate);
    }
    if (chunk.rate !== from) throw new AudioFormatError('the audio changes its sample rate');
    source += chunk.samples.length;
    if (source / from > maxSeconds || source > maxSeconds * BUDGET_RATE)
      throw new AudioFormatError(`the audio is longer than ${maxSeconds} seconds`);
    parts.push(resampler(chunk.samples));
  }
  if (resampler) parts.push(resampler.end());
  return concat(parts, Float32Array);
}

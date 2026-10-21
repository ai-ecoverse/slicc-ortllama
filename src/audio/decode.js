import { aacCodec } from './aac-codec.js';
import { adtsFrames, ascBytes, isAdts } from './adts.js';
import { decodeFlac, isFlac } from './flac.js';
import { decodeMp3, id3Size, isMp3 } from './mp3.js';
import { isMp4, mp4Audio } from './mp4.js';
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
import { createVorbisDecoder, vorbisHeaders } from './vorbis.js';
import { isWebm, webmAudio } from './webm.js';

export const MAX_SECONDS = 1800;

const BUDGET_RATE = 48000;

const AAC_OBJECTS = new Set([0x40, 0x66, 0x67]);

const SUPPORTED =
  'send wav, mp3, flac, m4a, mp4 or aac, ogg with opus or vorbis, webm with opus, or 16-bit PCM (audio/pcm;rate=N)';

const TYPES = {
  mp3: ['audio/mpeg', 'audio/mp3', 'audio/mpga', 'audio/x-mp3', 'audio/x-mpeg'],
  flac: ['audio/flac', 'audio/x-flac'],
  ogg: ['audio/ogg', 'audio/opus', 'audio/vorbis', 'application/ogg'],
  webm: ['audio/webm', 'video/webm'],
  wav: ['audio/wav', 'audio/x-wav', 'audio/wave', 'audio/vnd.wave'],
  mp4: ['audio/mp4', 'audio/m4a', 'audio/x-m4a', 'video/mp4', 'video/quicktime'],
  aac: ['audio/aac', 'audio/aacp', 'audio/x-aac', 'audio/adts'],
};

const EXTENSIONS = {
  mp3: ['mp3', 'mpga', 'mpeg'],
  flac: ['flac'],
  ogg: ['ogg', 'oga', 'opus'],
  webm: ['webm'],
  wav: ['wav', 'wave'],
  mp4: ['m4a', 'mp4', 'm4b', 'mov'],
  aac: ['aac', 'adts'],
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
    ['mp4', isMp4],
    ['aac', isAdts],
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

function opusStream(head, packets, granule, rate, container) {
  const text = String.fromCharCode(...(head?.subarray(0, 8) ?? []));
  if (!text.startsWith('OpusHead'))
    throw new AudioFormatError(`this ${container} codec is not supported; ${SUPPORTED}`);
  return decodeOpus({ head, packets, granule, rate: opusRate(rate) });
}

async function* trimmed(chunks, edit) {
  let skip = null;
  let keep = null;
  for await (const chunk of chunks) {
    skip ??= edit ? Math.round(edit.skip * chunk.rate) : 0;
    keep ??= edit ? Math.round(edit.keep * chunk.rate) : Infinity;
    const cut = Math.min(skip, chunk.samples.length);
    skip -= cut;
    const samples = chunk.samples.subarray(cut, cut + Math.min(keep, chunk.samples.length - cut));
    keep -= samples.length;
    if (samples.length) yield { samples, rate: chunk.rate };
    if (keep <= 0) return;
  }
}

function mono(planes) {
  if (planes.length === 1) return planes[0];
  const out = new Float32Array(planes[0].length);
  for (const plane of planes)
    for (let i = 0; i < out.length; i++) out[i] += plane[i] / planes.length;
  return out;
}

async function* aacChunks(aac, asc, frames) {
  let decoder = null;
  try {
    decoder = await aac.decoder({ asc });
    for (const frame of frames) {
      const planes = await decoder.decode(frame);
      if (planes[0]?.length) yield { samples: mono(planes), rate: decoder.sampleRate };
    }
    const done = decoder;
    decoder = null;
    const tail = await done.flush();
    if (tail[0]?.length) yield { samples: mono(tail), rate: done.sampleRate };
  } catch (error) {
    if (error instanceof AudioFormatError) throw error;
    throw new AudioFormatError(`the AAC stream could not be decoded: ${error?.message ?? error}`);
  } finally {
    await decoder?.flush().catch(() => {});
  }
}

function mp4Stream(bytes, aac) {
  const { codec, esds, frames, edit } = mp4Audio(bytes);
  if (codec !== 'mp4a' || !esds?.config || !AAC_OBJECTS.has(esds.object)) {
    const name = codec === 'mp4a' && esds ? `mp4a.${esds.object.toString(16)}` : codec;
    throw new AudioFormatError(`MP4 codec ${name} is not supported; ${SUPPORTED}`);
  }
  return trimmed(aacChunks(aac, esds.config, frames), edit);
}

function* vorbisChunks(decode, packets, rate) {
  for (const packet of packets) yield { samples: decode(packet), rate };
}

function vorbisStream(packets, granule) {
  const setup = vorbisHeaders(packets[0], packets[2]);
  const decode = createVorbisDecoder(setup);
  return trimmed(
    vorbisChunks(decode, packets.slice(3), setup.rate),
    granule >= 0 && { skip: 0, keep: granule / setup.rate }
  );
}

function chunks(bytes, format, contentType, rate, aac) {
  if (format === 'mp3') return decodeMp3(bytes);
  if (format === 'flac') return decodeFlac(bytes);
  if (format === 'mp4') return mp4Stream(bytes, aac);
  if (format === 'aac') {
    const { config, frames } = adtsFrames(bytes);
    return aacChunks(aac, ascBytes(config), frames);
  }
  if (format === 'ogg') {
    const { packets, granule } = oggPackets(bytes);
    if (packets[0]?.[0] === 1 && String.fromCharCode(...packets[0].subarray(1, 7)) === 'vorbis')
      return vorbisStream(packets, granule);
    return opusStream(packets[0], packets.slice(2), granule, rate, 'Ogg');
  }
  if (format === 'webm') {
    const { codec, head, packets, padding } = webmAudio(bytes);
    if (codec !== 'A_OPUS')
      throw new AudioFormatError(`WebM codec ${codec} is not supported; ${SUPPORTED}`);
    return opusStream(head, packets, endGranule(packets, padding), rate, 'WebM');
  }
  if (format === 'wav') return [decodeWav(bytes)];
  return [decodeAudio(bytes, contentType)];
}

export async function decodeFile(
  bytes,
  { type = '', name = '', rate = 16000, maxSeconds = MAX_SECONDS, aac = aacCodec() } = {}
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
  for await (const chunk of chunks(body, format, type, rate, aac)) {
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

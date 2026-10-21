import { createLayer3, frameHeader } from './layer3.js';
import { AudioFormatError } from './pcm.js';

const DECODER_DELAY = 529;
const FREE_LIMIT = 2884;
const LAME = new Set(['LAME', 'Lavf', 'Lavc']);
const TAGS = new Set(['Xing', 'Info']);
const SIDE = [
  [17, 32],
  [9, 17],
];

const sideSize = (head) => SIDE[head.mpeg1 ? 0 : 1][head.channels - 1];

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
  const head = frameHeader(bytes, at);
  if (head?.layer !== 3) return plain;
  const spf = head.samples;
  const end = at + (head.size || freeSize(bytes, at, head));
  const side = at + 4 + sideSize(head);
  const tag = !head.crc || TAGS.has(text(bytes, side)) ? side : side + 2;
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

function freeSize(bytes, at, head) {
  const last = Math.min(bytes.length - 4, at + FREE_LIMIT);
  for (let next = at + 4 + sideSize(head); next <= last; next++)
    if (frameHeader(bytes, next)?.key === head.key) return next - at;
  return bytes.length - at <= FREE_LIMIT ? bytes.length - at : 0;
}

function freeSizes(bytes) {
  const seen = new Map();
  let scanned = 0;
  return (at, head) => {
    const first = at + 4 + sideSize(head);
    const last = Math.min(bytes.length - 4, at + FREE_LIMIT);
    for (; scanned <= last; scanned++) {
      const found = bytes[scanned] === 0xff && frameHeader(bytes, scanned);
      if (found && !found.size) {
        if (!seen.has(found.key)) seen.set(found.key, { list: [], start: 0 });
        seen.get(found.key).list.push(scanned);
      }
    }
    const entry = seen.get(head.key) ?? { list: [], start: 0 };
    while (entry.start < entry.list.length && entry.list[entry.start] < first) entry.start++;
    if (entry.start > 1024) {
      entry.list = entry.list.slice(entry.start);
      entry.start = 0;
    }
    const next = entry.list[entry.start];
    if (next !== undefined && next <= last) return next - at;
    return bytes.length - at <= FREE_LIMIT ? bytes.length - at : 0;
  };
}

function* frames(bytes) {
  const free = freeSizes(bytes);
  let lock = null;
  for (let at = id3Size(bytes); at + 4 <= bytes.length; ) {
    const head = frameHeader(bytes, at);
    if (head) {
      const size = head.size || free(at, head);
      const next = at + size;
      const follows = next === bytes.length || frameHeader(bytes, next)?.key === head.key;
      if (size && (head.key === lock || follows)) {
        if (head.layer !== 3)
          throw new AudioFormatError(
            `MPEG audio layer ${'I'.repeat(head.layer)} is not supported; send mp3 (layer III)`
          );
        lock = head.key;
        yield [head, at, size];
        at = next;
        continue;
      }
    }
    at++;
  }
  if (lock === null) throw new AudioFormatError('the mp3 stream is corrupt');
}

export function* mp3Channels(bytes, options) {
  const { body, skip, keep } = gapless(bytes);
  const decode = createLayer3(options);
  let at = 0;
  for (const [head, start, size] of frames(body)) {
    const pcm = decode(body, start, head, size);
    const from = Math.min(head.samples, Math.max(0, skip - at));
    const to = Math.max(from, Math.min(head.samples, skip + keep - at));
    at += head.samples;
    if (to > from) yield { channels: pcm.map((c) => c.subarray(from, to)), rate: head.rate };
    if (at >= skip + keep) return;
  }
}

export function* decodeMp3(bytes) {
  for (const { channels, rate } of mp3Channels(bytes, { mono: true }))
    yield { samples: channels[0], rate };
}

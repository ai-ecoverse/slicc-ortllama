import { AudioFormatError } from './pcm.js';

const MASTERS = new Set([0x18538067, 0x1654ae6b, 0xae, 0x1f43b675, 0xa0]);

export function isWebm(bytes) {
  return (
    bytes.length >= 4 &&
    bytes[0] === 0x1a &&
    bytes[1] === 0x45 &&
    bytes[2] === 0xdf &&
    bytes[3] === 0xa3
  );
}

function corrupt() {
  return new AudioFormatError('the WebM file is corrupt');
}

function vint(bytes, at, keep = false) {
  const first = bytes[at];
  if (!first) throw corrupt();
  const length = Math.clz32(first) - 23;
  if (at + length > bytes.length) throw corrupt();
  let value = keep ? first : first & (0xff >> length);
  let ones = value === 0xff >> length;
  for (let i = 1; i < length; i++) {
    value = value * 256 + bytes[at + i];
    ones &&= bytes[at + i] === 0xff;
  }
  return { value, length, unknown: !keep && ones };
}

function uint(bytes) {
  let value = 0;
  for (const b of bytes) value = value * 256 + b;
  return value;
}

function laced(data, lacing) {
  if (lacing === 0) return [data];
  const count = data[0] + 1;
  let at = 1;
  const sizes = [];
  if (lacing === 1) {
    for (let i = 0; i < count - 1; i++) {
      let size = 0;
      while (data[at] === 255) size += data[at++];
      sizes.push(size + data[at++]);
    }
  } else if (lacing === 3) {
    const first = vint(data, at);
    at += first.length;
    sizes.push(first.value);
    for (let i = 1; i < count - 1; i++) {
      const diff = vint(data, at);
      at += diff.length;
      sizes.push(sizes[i - 1] + diff.value - (2 ** (7 * diff.length - 1) - 1));
    }
  } else {
    const size = (data.length - 1) / count;
    if (!Number.isInteger(size)) throw corrupt();
    sizes.push(...Array(count - 1).fill(size));
  }
  const frames = [];
  for (const size of sizes) {
    if (size < 0 || at + size > data.length) throw corrupt();
    frames.push(data.subarray(at, at + size));
    at += size;
  }
  frames.push(data.subarray(at));
  return frames;
}

function signed(bytes) {
  const value = uint(bytes);
  return bytes.length > 0 && bytes[0] & 0x80 ? value - 2 ** (8 * bytes.length) : value;
}

function audioTrack(entries) {
  return entries.find((e) => e.codec?.startsWith('A_') && e.number !== undefined) ?? null;
}

export function webmAudio(bytes) {
  const decoder = new TextDecoder();
  const entries = [];
  let track = null;
  const packets = [];
  let padding = 0;
  for (let at = 0; at < bytes.length; ) {
    const id = vint(bytes, at, true);
    const size = vint(bytes, at + id.length);
    const body = at + id.length + size.length;
    if (MASTERS.has(id.value)) {
      if (id.value === 0xae) entries.push({});
      at = body;
      continue;
    }
    if (size.unknown) throw corrupt();
    const data = bytes.subarray(body, Math.min(body + size.value, bytes.length));
    const entry = entries.at(-1) ?? {};
    if (id.value === 0xd7) entry.number = uint(data);
    else if (id.value === 0x86) entry.codec = decoder.decode(data);
    else if (id.value === 0x63a2) entry.head = data;
    else if (id.value === 0x75a2) padding += Math.round((signed(data) * 48000) / 1e9);
    if (id.value === 0xa3 || id.value === 0xa1) {
      track ??= audioTrack(entries);
      const number = vint(data, 0);
      if (number.value === track?.number)
        packets.push(
          ...laced(data.subarray(number.length + 3), (data[number.length + 2] >> 1) & 3)
        );
    }
    at = body + size.value;
  }
  track ??= audioTrack(entries);
  if (!track) throw new AudioFormatError('the WebM file has no audio track');
  return { codec: track.codec, head: track.head, packets, padding };
}

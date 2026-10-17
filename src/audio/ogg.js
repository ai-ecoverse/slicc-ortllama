import { AudioFormatError } from './pcm.js';

const TABLE = new Uint32Array(256).map((_, i) => {
  let c = i << 24;
  for (let k = 0; k < 8; k++) c = c & 0x80000000 ? (c << 1) ^ 0x04c11db7 : c << 1;
  return c >>> 0;
});

export function oggCrc(bytes) {
  let crc = 0;
  for (const b of bytes) crc = ((crc << 8) ^ TABLE[((crc >>> 24) ^ b) & 0xff]) >>> 0;
  return crc;
}

export function isOgg(bytes) {
  return (
    bytes.length >= 4 &&
    bytes[0] === 0x4f &&
    bytes[1] === 0x67 &&
    bytes[2] === 0x67 &&
    bytes[3] === 0x53
  );
}

export function oggPackets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const packets = [];
  let serial = null;
  let granule = -1;
  let parts = [];
  for (let at = 0; at + 27 <= bytes.length; ) {
    if (!isOgg(bytes.subarray(at, at + 4))) throw new AudioFormatError('the Ogg stream is corrupt');
    const count = bytes[at + 26];
    const lacing = bytes.subarray(at + 27, at + 27 + count);
    let body = at + 27 + count;
    const page = view.getUint32(at + 14, true);
    serial ??= page;
    const mine = page === serial;
    if (mine && !(bytes[at + 5] & 1)) parts = [];
    for (const size of lacing) {
      if (mine) parts.push(bytes.subarray(body, Math.min(body + size, bytes.length)));
      body += size;
      if (mine && size < 255) {
        const length = parts.reduce((n, p) => n + p.length, 0);
        const packet = new Uint8Array(length);
        let o = 0;
        for (const p of parts) {
          packet.set(p, o);
          o += p.length;
        }
        packets.push(packet);
        parts = [];
      }
    }
    if (mine) {
      const g = view.getBigInt64(at + 6, true);
      if (g >= 0n) granule = Number(g);
    }
    at = body;
  }
  return { packets, granule };
}

export function oggPage({ serial, sequence, granule, flags = 0, packets }) {
  const lacing = [];
  for (const p of packets) {
    for (let n = p.length; ; n -= 255) {
      lacing.push(Math.min(n, 255));
      if (n < 255) break;
    }
  }
  const size = packets.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(27 + lacing.length + size);
  const view = new DataView(out.buffer);
  out.set([0x4f, 0x67, 0x67, 0x53, 0, flags]);
  view.setBigInt64(6, BigInt(granule), true);
  view.setUint32(14, serial, true);
  view.setUint32(18, sequence, true);
  out[26] = lacing.length;
  out.set(lacing, 27);
  let at = 27 + lacing.length;
  for (const p of packets) {
    out.set(p, at);
    at += p.length;
  }
  view.setUint32(22, oggCrc(out), true);
  return out;
}

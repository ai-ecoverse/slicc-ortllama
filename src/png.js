import { Unzlib } from 'fflate';
import { badRequest } from './http/errors.js';

const KEEP = new Set(['IHDR', 'PLTE', 'tRNS', 'IDAT', 'IEND']);
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
const ADAM7 = [
  [0, 0, 8, 8],
  [4, 0, 8, 8],
  [0, 4, 4, 8],
  [2, 0, 4, 4],
  [0, 2, 2, 4],
  [1, 0, 2, 2],
  [0, 1, 1, 2],
];
const SLICE = 4096;

const u32 = (b, i) => b[i] * 2 ** 24 + ((b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]);
const type = (b, i) => String.fromCharCode(b[i], b[i + 1], b[i + 2], b[i + 3]);

export function pngChunks(bytes, param = 'messages') {
  const kept = [bytes.subarray(0, 8)];
  const idat = [];
  let ihdr = null;
  let exif = null;
  let at = 8;
  while (at < bytes.length) {
    if (at + 12 > bytes.length) throw badRequest('the png image is cut off', param);
    const length = u32(bytes, at);
    const name = type(bytes, at + 4);
    const end = at + 12 + length;
    if (end > bytes.length) throw badRequest('the png image is cut off', param);
    const data = bytes.subarray(at + 8, at + 8 + length);
    if (KEEP.has(name)) kept.push(bytes.subarray(at, end));
    if (name === 'IHDR')
      ihdr = {
        width: u32(data, 0),
        height: u32(data, 4),
        depth: data[8],
        color: data[9],
        interlace: data[12],
      };
    if (name === 'IDAT') idat.push(data);
    if (name === 'eXIf') exif = data;
    at = end;
    if (name === 'IEND') break;
  }
  if (!ihdr || idat.length === 0) throw badRequest('the png image has no image data', param);
  const file = new Uint8Array(kept.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of kept) {
    file.set(c, offset);
    offset += c.length;
  }
  return { ihdr, idat, exif, file };
}

export function rawSize({ width, height, depth, color, interlace }, param = 'messages') {
  const channels = CHANNELS[color];
  if (!channels || ![1, 2, 4, 8, 16].includes(depth))
    throw badRequest('the png image has an unknown color type or bit depth', param);
  const row = (w) => 1 + Math.ceil((w * channels * depth) / 8);
  if (!interlace) return height * row(width);
  let size = 0;
  for (const [x0, y0, dx, dy] of ADAM7) {
    const w = Math.ceil((width - x0) / dx);
    const h = Math.ceil((height - y0) / dy);
    if (w > 0 && h > 0) size += h * row(w);
  }
  return size;
}

export function checkInflate(parts, limit, param = 'messages') {
  let total = 0;
  const stream = new Unzlib((chunk) => {
    total += chunk.length;
    if (total > limit)
      throw badRequest('the png image data inflates past the size its header declares', param);
  });
  const input = parts.flatMap((p) => {
    const out = [];
    for (let i = 0; i < p.length; i += SLICE) out.push(p.subarray(i, i + SLICE));
    return out;
  });
  for (const [i, slice] of input.entries()) stream.push(slice, i === input.length - 1);
  return total;
}

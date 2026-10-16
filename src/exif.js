const ORIENTATION = 0x0112;
const EXIF_HEADER = [0x45, 0x78, 0x69, 0x66, 0, 0];

const startsWith = (b, prefix, at = 0) => prefix.every((v, i) => b[at + i] === v);

function tiffOf(data) {
  return startsWith(data, EXIF_HEADER) ? data.subarray(EXIF_HEADER.length) : data;
}

export function orientation(data) {
  if (!data) return 1;
  const t = tiffOf(data);
  if (t.length < 8) return 1;
  const little = t[0] === 0x49 && t[1] === 0x49;
  if (!little && !(t[0] === 0x4d && t[1] === 0x4d)) return 1;
  const u16 = (i) => (little ? t[i] | (t[i + 1] << 8) : (t[i] << 8) | t[i + 1]);
  const u32 = (i) => (little ? u16(i) + u16(i + 2) * 65536 : u16(i) * 65536 + u16(i + 2));
  const ifd = u32(4);
  if (ifd + 2 > t.length) return 1;
  const count = u16(ifd);
  for (let k = 0; k < count; k++) {
    const entry = ifd + 2 + k * 12;
    if (entry + 12 > t.length) return 1;
    if (u16(entry) !== ORIENTATION) continue;
    const value = u16(entry + 8);
    return value >= 1 && value <= 8 ? value : 1;
  }
  return 1;
}

export function jpegExif(b) {
  let i = 2;
  while (i + 4 <= b.length && b[i] === 0xff) {
    const marker = b[i + 1];
    if (marker === 0xda || marker === 0xd9) return null;
    const length = (b[i + 2] << 8) | b[i + 3];
    if (marker === 0xe1 && startsWith(b, EXIF_HEADER, i + 4))
      return b.subarray(i + 4, i + 2 + length);
    i += 2 + length;
  }
  return null;
}

export function webpExif(b) {
  let i = 12;
  while (i + 8 <= b.length) {
    const name = String.fromCharCode(b[i], b[i + 1], b[i + 2], b[i + 3]);
    const size = b[i + 4] | (b[i + 5] << 8) | (b[i + 6] << 16) | (b[i + 7] * 2 ** 24);
    if (name === 'EXIF') return b.subarray(i + 8, i + 8 + size);
    i += 8 + size + (size % 2);
  }
  return null;
}

const SOURCE = {
  2: (x, y, w) => [w - 1 - x, y],
  3: (x, y, w, h) => [w - 1 - x, h - 1 - y],
  4: (x, y, w, h) => [x, h - 1 - y],
  5: (x, y) => [y, x],
  6: (x, y, w, h) => [y, h - 1 - x],
  7: (x, y, w, h) => [w - 1 - y, h - 1 - x],
  8: (x, y, w) => [w - 1 - y, x],
};

export function orient(image, value) {
  const map = SOURCE[value];
  if (!map) return image;
  const { width: w, height: h, data } = image;
  const turned = value >= 5;
  const W = turned ? h : w;
  const H = turned ? w : h;
  const out = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const [sx, sy] = map(x, y, w, h);
      const from = (sy * w + sx) * 4;
      out.set(data.subarray(from, from + 4), (y * W + x) * 4);
    }
  return { width: W, height: H, data: out };
}

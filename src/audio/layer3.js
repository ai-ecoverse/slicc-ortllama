import { imdct } from './mdct.js';
import { HUFFMAN, LONG, SHORT, WINDOW } from './mp3-tables.js';

const KBPS = [
  [
    [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
  ],
  [
    [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
  ],
];
const RATES = [44100, 48000, 32000];
const RESERVOIR = 512;
const PRETAB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0];
const SLEN = [
  [0, 0],
  [0, 1],
  [0, 2],
  [0, 3],
  [3, 0],
  [1, 1],
  [1, 2],
  [1, 3],
  [2, 1],
  [2, 2],
  [2, 3],
  [3, 1],
  [3, 2],
  [3, 3],
  [4, 2],
  [4, 3],
];
const PARTS = [
  [
    [6, 5, 5, 5],
    [9, 9, 9, 9],
    [6, 9, 9, 9],
  ],
  [
    [6, 5, 7, 3],
    [9, 9, 12, 6],
    [6, 9, 12, 6],
  ],
  [
    [11, 10, 0, 0],
    [18, 18, 0, 0],
    [15, 18, 0, 0],
  ],
  [
    [7, 7, 7, 0],
    [12, 12, 12, 0],
    [6, 15, 12, 0],
  ],
  [
    [6, 6, 6, 3],
    [12, 9, 9, 6],
    [6, 12, 9, 6],
  ],
  [
    [8, 8, 5, 0],
    [15, 12, 9, 0],
    [6, 18, 9, 0],
  ],
];
const GROUPS = [0, 6, 11, 16, 21];
const ALIAS = [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037];
const CS = ALIAS.map((c) => 1 / Math.sqrt(1 + c * c));
const CA = ALIAS.map((c) => c / Math.sqrt(1 + c * c));
const LINBITS = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 6, 8, 10, 13, 4, 5, 6, 7, 8, 9, 11,
  13,
];
const ROOT = 8;

export class Bits {
  constructor(bytes, pos = 0) {
    this.bytes = bytes;
    this.pos = pos;
  }

  peek(n) {
    const b = this.bytes;
    const i = this.pos >> 3;
    const word = ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) << (this.pos & 7);
    return word >>> (32 - n);
  }

  read(n) {
    if (n === 0) return 0;
    const value = this.peek(n);
    this.pos += n;
    return value;
  }
}

export function huffmanCodes(text) {
  const leaves = [];
  let code = -1;
  let len = 0;
  for (let i = 0; i < text.length; i += 3) {
    const next = Number.parseInt(text[i + 2], 36);
    code =
      next >= len ? (code + 1) * 2 ** (next - len) : Math.floor((code + 1) / 2 ** (len - next));
    len = next;
    leaves.push({ symbol: Number.parseInt(text.slice(i, i + 2), 16), code, len });
  }
  return leaves;
}

function lookup(leaves) {
  const data = [];
  const fill = (items, depth) => {
    const width = Math.min(ROOT, Math.max(...items.map((l) => l.len)) - depth);
    const start = data.length;
    for (let i = 0; i < 1 << width; i++) data.push(0);
    const groups = new Map();
    for (const leaf of items) {
      const rest = leaf.len - depth;
      const tail = leaf.code % 2 ** rest;
      if (rest <= width) {
        const base = tail << (width - rest);
        for (let k = 0; k < 1 << (width - rest); k++)
          data[start + base + k] = (leaf.symbol << 8) | rest;
      } else {
        const top = Math.floor(tail / 2 ** (rest - width));
        if (!groups.has(top)) groups.set(top, []);
        groups.get(top).push(leaf);
      }
    }
    for (const [top, group] of groups) {
      const sub = fill(group, depth + width);
      data[start + top] = -((sub.start << 5) | sub.width);
    }
    return { start, width };
  };
  const { width } = fill(leaves, 0);
  return { data: Int32Array.from(data), width };
}

let tables;

function codebooks() {
  if (tables) return tables;
  const books = {};
  for (const [n, text] of Object.entries(HUFFMAN)) books[n] = lookup(huffmanCodes(text));
  const pair = (n) => (n < 16 ? books[n] : n < 24 ? books[16] : books[24]);
  const pow = new Float64Array(8207);
  for (let i = 0; i < pow.length; i++) pow[i] = i ** (4 / 3);
  const window = new Float64Array(512);
  for (let i = 0; i <= 256; i++) window[i] = WINDOW[i] / 65536;
  for (let i = 1; i < 256; i++) window[512 - i] = (i % 64 === 0 ? 1 : -1) * window[i];
  tables = {
    pairs: Array.from({ length: 32 }, (_, n) => pair(n) ?? null),
    quads: [books[32], books[33]],
    pow,
    window,
    layouts: new Map(),
  };
  return tables;
}

function decode(bits, book) {
  const { data } = book;
  let at = 0;
  let width = book.width;
  for (;;) {
    const entry = data[at + bits.peek(width)];
    if (entry >= 0) {
      bits.pos += entry & 255;
      return entry >> 8;
    }
    bits.pos += width;
    at = -entry >> 5;
    width = -entry & 31;
  }
}

const edges = (widths) => widths.reduce((list, w) => [...list, list.at(-1) + w], [0]);

function layout(table, kind) {
  const key = table * 3 + kind;
  const cache = codebooks().layouts;
  if (cache.has(key)) return cache.get(key);
  const long = LONG[table];
  const short = SHORT[table];
  const split = 3 * (short[0] + short[1] + short[2]);
  const start = [];
  const width = [];
  const win = [];
  const sfb = [];
  let at = 0;
  const push = (w, window, band) => {
    start.push(at);
    width.push(w);
    win.push(window);
    sfb.push(band);
    at += w;
  };
  let first = 0;
  if (kind === 0) for (let i = 0; i < 22; i++) push(long[i], -1, i);
  if (kind === 2) for (let i = 0; at < split; i++) push(long[i], -1, i);
  if (kind === 2) first = 3;
  if (kind !== 0) for (let s = first; s < 13; s++) for (let w = 0; w < 3; w++) push(short[s], w, s);
  const result = {
    start,
    width,
    win,
    sfb,
    longs: kind === 2 ? split / 18 : kind === 0 ? 32 : 0,
    split: kind === 2 ? split : kind === 0 ? 576 : 0,
  };
  cache.set(key, result);
  return result;
}

function sfbTable(version, rateIndex) {
  if (version === 3) return 5 + rateIndex;
  if (version === 2) return 2 + rateIndex;
  return rateIndex === 2 ? 1 : 0;
}

function frameSize(layer, mpeg1, kbps, rate, padding) {
  if (!kbps) return 0;
  if (layer === 1) return (Math.floor((12000 * kbps) / rate) + padding) * 4;
  return Math.floor(((layer === 3 && !mpeg1 ? 72000 : 144000) * kbps) / rate) + padding;
}

export function frameHeader(bytes, at) {
  if (at + 4 > bytes.length || bytes[at] !== 0xff || (bytes[at + 1] & 0xe0) !== 0xe0) return null;
  const b1 = bytes[at + 1];
  const b2 = bytes[at + 2];
  const b3 = bytes[at + 3];
  const version = (b1 >> 3) & 3;
  const layer = 4 - ((b1 >> 1) & 3);
  const index = b2 >> 4;
  const rateIndex = (b2 >> 2) & 3;
  if (version === 1 || layer === 4 || index === 15 || rateIndex === 3) return null;
  const mpeg1 = version === 3;
  const rate = RATES[rateIndex] >> (mpeg1 ? 0 : version === 2 ? 1 : 2);
  const mode = b3 >> 6;
  return {
    layer,
    mpeg1,
    rate,
    key: ((b1 & 0xfe) << 8) | (b2 & 0x0c) | (index ? 0 : 1),
    crc: (b1 & 1) === 0,
    channels: mode === 3 ? 1 : 2,
    ms: mode === 1 && (b3 & 0x20) !== 0,
    intensity: mode === 1 && (b3 & 0x10) !== 0,
    table: sfbTable(version, rateIndex),
    size: frameSize(layer, mpeg1, KBPS[mpeg1 ? 0 : 1][layer - 1][index], rate, (b2 >> 1) & 1),
    samples: mpeg1 ? 1152 : 576,
  };
}

function granule(bits, head, scfsi) {
  const longEdges = edges(LONG[head.table]);
  const g = {
    part23: bits.read(12),
    big: bits.read(9),
    gain: bits.read(8),
    compress: bits.read(head.mpeg1 ? 4 : 9),
    type: 0,
    mixed: false,
    tables: [0, 0, 0],
    subgain: [0, 0, 0],
    region1: 0,
    region2: 576,
    valid: true,
    scfsi,
  };
  if (bits.read(1)) {
    g.type = bits.read(2);
    g.mixed = bits.read(1) === 1;
    g.tables = [bits.read(5), bits.read(5), 0];
    g.subgain = [bits.read(3), bits.read(3), bits.read(3)];
    g.region1 = g.type === 2 ? 3 * edges(SHORT[head.table])[3] : longEdges[8];
    g.valid = g.type !== 0;
  } else {
    g.tables = [bits.read(5), bits.read(5), bits.read(5)];
    const r0 = bits.read(4);
    const r1 = bits.read(3);
    g.region1 = longEdges[r0 + 1];
    g.region2 = longEdges[Math.min(22, r0 + r1 + 2)];
  }
  g.preflag = head.mpeg1 ? bits.read(1) : 0;
  g.scale = bits.read(1);
  g.count1 = bits.read(1);
  g.kind = g.type === 2 ? (g.mixed ? 2 : 1) : 0;
  g.valid &&= g.big <= 288;
  return g;
}

function lsfSlen(head, g, ch) {
  const is = head.intensity && ch === 1;
  let c = is ? g.compress >> 1 : g.compress;
  if (!is) {
    if (c < 400) return [0, [Math.floor((c >> 4) / 5), (c >> 4) % 5, (c & 15) >> 2, c & 3]];
    if (c < 500) {
      c -= 400;
      return [1, [Math.floor((c >> 2) / 5), (c >> 2) % 5, c & 3, 0]];
    }
    c -= 500;
    g.preflag = 1;
    return [2, [Math.floor(c / 3), c % 3, 0, 0]];
  }
  if (c < 180) return [3, [Math.floor(c / 36), Math.floor((c % 36) / 6), c % 6, 0]];
  if (c < 244) {
    c -= 180;
    return [4, [(c & 63) >> 4, (c & 15) >> 2, c & 3, 0]];
  }
  c -= 244;
  return [5, [Math.floor(c / 3), c % 3, 0, 0]];
}

function mpeg1Scalefactors(bits, g, saved, sf, count) {
  const [s1, s2] = SLEN[g.compress];
  if (g.kind === 0) {
    for (let k = 0; k < 4; k++)
      for (let i = GROUPS[k]; i < GROUPS[k + 1]; i++)
        sf[i] = g.scfsi & (8 >> k) ? saved[i] : bits.read(k < 2 ? s1 : s2);
    saved.set(sf.subarray(0, 22));
    return;
  }
  const split = g.kind === 2 ? 17 : 18;
  for (let i = 0; i < count - 3; i++) sf[i] = bits.read(i < split ? s1 : s2);
}

function scalefactors(bits, head, g, ch, saved, sf, illegal) {
  const bands = layout(head.table, g.kind);
  const count = bands.start.length;
  sf.fill(0, 0, count);
  illegal.fill(0, 0, count);
  if (head.mpeg1) {
    mpeg1Scalefactors(bits, g, saved, sf, count);
    for (let i = 0; i < count; i++) illegal[i] = sf[i] >= 7 ? 1 : 0;
    return bands;
  }
  const [table, slen] = lsfSlen(head, g, ch);
  const parts = PARTS[table][g.kind];
  let i = 0;
  for (let k = 0; k < 4; k++)
    for (let j = 0; j < parts[k]; j++, i++) {
      sf[i] = bits.read(slen[k]);
      illegal[i] = slen[k] && sf[i] === (1 << slen[k]) - 1 ? 1 : 0;
    }
  return bands;
}

function value(bits, v, linbits) {
  const big = v === 15 && linbits ? v + bits.read(linbits) : v;
  return big && bits.read(1) ? -big : big;
}

function pairs(bits, g, q) {
  const books = codebooks().pairs;
  const big = g.big * 2;
  let i = 0;
  for (let r = 0; r < 3; r++) {
    const stop = Math.min(big, [g.region1, g.region2, 576][r]);
    const book = books[g.tables[r]];
    const linbits = LINBITS[g.tables[r]];
    for (; book && i < stop; i += 2) {
      const symbol = decode(bits, book);
      q[i] = value(bits, symbol >> 4, linbits);
      q[i + 1] = value(bits, symbol & 15, linbits);
    }
    i = Math.max(i, stop);
  }
  return i;
}

function huffman(bits, g, end, q) {
  q.fill(0);
  let i = pairs(bits, g, q);
  const book = codebooks().quads[g.count1];
  while (i + 4 <= 576 && bits.pos < end) {
    const symbol = decode(bits, book);
    for (let k = 0; k < 4; k++) q[i + k] = value(bits, (symbol >> (3 - k)) & 1, 0);
    if (bits.pos > end) {
      q.fill(0, i, i + 4);
      break;
    }
    i += 4;
  }
  bits.pos = end;
  return i;
}

function requantize(q, xr, g, bands, sf, count) {
  const { pow } = codebooks();
  xr.fill(0);
  const mult = g.scale ? 1 : 0.5;
  const { start, width, win, sfb } = bands;
  for (let b = 0; b < start.length && start[b] < count; b++) {
    const w = win[b];
    const pre = g.preflag && w < 0 ? PRETAB[sfb[b]] : 0;
    const scale = 2 ** ((g.gain - 210) / 4 - mult * (sf[b] + pre) - (w < 0 ? 0 : 2 * g.subgain[w]));
    const stop = Math.min(start[b] + width[b], count);
    for (let i = start[b]; i < stop; i++) {
      const v = q[i];
      if (v > 0) xr[i] = pow[v] * scale;
      else if (v < 0) xr[i] = -pow[-v] * scale;
    }
  }
}

function midSide(left, right, from, to) {
  for (let i = from; i < to; i++) {
    const m = left[i];
    const s = right[i];
    left[i] = (m + s) * Math.SQRT1_2;
    right[i] = (m - s) * Math.SQRT1_2;
  }
}

function ratios(head, pos, g) {
  if (head.mpeg1) {
    const s = Math.sin((pos * Math.PI) / 12);
    const c = Math.cos((pos * Math.PI) / 12);
    return [s / (s + c), c / (s + c)];
  }
  const k = 2 ** (-((pos + 1) >> 1) / (g.compress & 1 ? 2 : 4));
  if (pos === 0) return [1, 1];
  return pos & 1 ? [k, 1] : [1, k];
}

function stereo(head, xr, g, bands, sf, illegal) {
  const [left, right] = xr;
  if (!head.intensity) {
    if (head.ms) midSide(left, right, 0, 576);
    return;
  }
  const { start, width, win } = bands;
  const count = start.length;
  const top = [-1, -1, -1, -1];
  for (let b = 0; b < count; b++) {
    for (let i = start[b]; i < start[b] + width[b]; i++)
      if (right[i] !== 0) {
        top[win[b] + 1] = b;
        break;
      }
  }
  const shorts = Math.max(top[1], top[2], top[3]);
  const step = g.kind === 0 ? 1 : 3;
  for (let b = count - step; b < count; b++) {
    sf[b] = sf[b - step];
    illegal[b] = illegal[b - step];
  }
  for (let b = 0; b < count; b++) {
    const w = win[b];
    const above = w < 0 ? b > top[0] && shorts < 0 : b > top[w + 1];
    const from = start[b];
    const to = from + width[b];
    if (above && !illegal[b]) {
      const [kl, kr] = ratios(head, sf[b], g);
      for (let i = from; i < to; i++) {
        right[i] = left[i] * kr;
        left[i] *= kl;
      }
    } else if (head.ms) midSide(left, right, from, to);
  }
}

const sine = (i, n) => Math.sin((Math.PI / n) * (i + 0.5));

function startWindow(i) {
  if (i < 18) return sine(i, 36);
  if (i < 24) return 1;
  return i < 30 ? sine(i - 18, 12) : 0;
}

const LONG_WINDOWS = [
  Float64Array.from({ length: 36 }, (_, i) => sine(i, 36)),
  Float64Array.from({ length: 36 }, (_, i) => startWindow(i)),
  null,
  Float64Array.from({ length: 36 }, (_, i) => startWindow(35 - i)),
];
const SHORT_WINDOW = Float64Array.from({ length: 12 }, (_, i) =>
  Math.sin((Math.PI / 12) * (i + 0.5))
);

function reorder(xr, bands, scratch) {
  const { start, width, win } = bands;
  const from = bands.split;
  for (let b = 0; b < start.length; b++) {
    if (win[b] < 0) continue;
    const base = start[b] - win[b] * width[b];
    for (let i = 0; i < width[b]; i++) scratch[base + 3 * i + win[b]] = xr[start[b] + i];
  }
  xr.set(scratch.subarray(from, 576), from);
}

function antialias(xr, longs) {
  for (let sb = 1; sb < longs; sb++) {
    const at = sb * 18;
    for (let i = 0; i < 8; i++) {
      const lower = xr[at - 1 - i];
      const upper = xr[at + i];
      xr[at - 1 - i] = lower * CS[i] - upper * CA[i];
      xr[at + i] = upper * CS[i] + lower * CA[i];
    }
  }
}

function longBlock(xr, at, window, work) {
  const { block, long } = work;
  for (let i = 0; i < 18; i++) long[i] = xr[at + i];
  imdct(long, block, 36);
  for (let i = 0; i < 36; i++) block[i] *= window[i];
}

function shortBlock(xr, at, work) {
  const { block, small, input } = work;
  block.fill(0);
  for (let w = 0; w < 3; w++) {
    for (let k = 0; k < 6; k++) input[k] = xr[at + 3 * k + w];
    imdct(input, small, 12);
    for (let i = 0; i < 12; i++) block[6 + 6 * w + i] += small[i] * SHORT_WINDOW[i];
  }
}

function silent(xr, at) {
  for (let i = at; i < at + 18; i++) if (xr[i] !== 0) return false;
  return true;
}

function hybrid(xr, g, bands, overlap, work) {
  const { block, out } = work;
  for (let sb = 0; sb < 32; sb++) {
    const at = sb * 18;
    if (silent(xr, at)) block.fill(0);
    else if (g.kind === 0 || sb < bands.longs)
      longBlock(xr, at, LONG_WINDOWS[g.kind === 0 && !(g.mixed && sb < 2) ? g.type : 0], work);
    else shortBlock(xr, at, work);
    for (let i = 0; i < 18; i++) {
      out[at + i] = block[i] + overlap[at + i];
      overlap[at + i] = block[18 + i];
    }
    if (sb & 1) for (let i = 1; i < 18; i += 2) out[at + i] = -out[at + i];
  }
}

const DCT_SCALES = [32, 16, 8, 4, 2].map((n) =>
  Float64Array.from({ length: n >> 1 }, (_, i) => 0.5 / Math.cos((Math.PI * (2 * i + 1)) / (2 * n)))
);

export function dct32(input, one, two) {
  let a = one;
  let b = two;
  a.set(input);
  for (let n = 32, level = 0; n > 1; n >>= 1, level++) {
    const half = n >> 1;
    const scale = DCT_SCALES[level];
    for (let o = 0; o < 32; o += n)
      for (let i = 0; i < half; i++) {
        const u = a[o + i];
        const w = a[o + n - 1 - i];
        b[o + i] = u + w;
        b[o + half + i] = (u - w) * scale[i];
      }
    [a, b] = [b, a];
  }
  for (let n = 2; n <= 32; n <<= 1) {
    const half = n >> 1;
    for (let o = 0; o < 32; o += n) {
      for (let k = 0; k < half - 1; k++) {
        b[o + 2 * k] = a[o + k];
        b[o + 2 * k + 1] = a[o + half + k] + a[o + half + k + 1];
      }
      b[o + n - 2] = a[o + half - 1];
      b[o + n - 1] = a[o + n - 1];
    }
    [a, b] = [b, a];
  }
  return a;
}

function synthesize(state, out, pcm, offset) {
  const { window } = codebooks();
  const { v, s, one, two } = state;
  for (let t = 0; t < 18; t++) {
    for (let k = 0; k < 32; k++) s[k] = out[k * 18 + t];
    const x = dct32(s, one, two);
    state.off = (state.off + 960) & 1023;
    const off = state.off;
    for (let i = 0; i < 64; i++) {
      const value = i <= 16 ? (i === 16 ? 0 : x[16 + i]) : i < 48 ? -x[48 - i] : -x[i - 48];
      v[off + i] = value;
      v[off + i + 1024] = value;
    }
    for (let j = 0; j < 32; j++) {
      let sum = 0;
      for (let i = 0; i < 8; i++) {
        sum += window[64 * i + j] * v[off + 128 * i + j];
        sum += window[64 * i + 32 + j] * v[off + 128 * i + 96 + j];
      }
      pcm[offset + 32 * t + j] = sum;
    }
  }
}

function channelState() {
  return {
    overlap: new Float64Array(576),
    saved: new Int32Array(22),
    v: new Float64Array(2048),
    off: 0,
    s: new Float64Array(32),
    one: new Float64Array(32),
    two: new Float64Array(32),
  };
}

function sideInfo(bytes, at, head) {
  const nch = head.channels;
  const side = new Bits(bytes, (at + 4 + (head.crc ? 2 : 0)) * 8);
  const begin = side.read(head.mpeg1 ? 9 : 8);
  side.pos += head.mpeg1 ? 7 - 2 * nch : nch;
  const scfsi = [0, 0];
  if (head.mpeg1) for (let ch = 0; ch < nch; ch++) scfsi[ch] = side.read(4);
  const granules = [];
  for (let gr = 0; gr < (head.mpeg1 ? 2 : 1); gr++) {
    const list = [];
    for (let ch = 0; ch < nch; ch++) list.push(granule(side, head, gr ? scfsi[ch] : 0));
    granules.push(list);
  }
  const valid = granules.every((list) => list.every((g) => g.valid));
  return { begin, granules, valid, end: side.pos >> 3 };
}

export function createLayer3({ mono = false } = {}) {
  const channels = [channelState(), channelState()];
  const xr = [new Float64Array(576), new Float64Array(576)];
  const q = new Int32Array(576);
  const sf = [new Int32Array(39), new Int32Array(39)];
  const illegal = [new Uint8Array(39), new Uint8Array(39)];
  const work = {
    block: new Float64Array(36),
    small: new Float64Array(12),
    input: new Float64Array(6),
    long: new Float64Array(18),
    out: new Float64Array(576),
    short: new Float64Array(576),
    mix: new Float64Array(576),
  };
  let reservoir = new Uint8Array(0);
  const spectrum = (bits, head, list, usable) => {
    const bands = [];
    for (let ch = 0; ch < list.length; ch++) {
      const g = list[ch];
      const end = bits.pos + g.part23;
      bands.push(layout(head.table, g.kind));
      xr[ch].fill(0);
      if (usable) {
        scalefactors(bits, head, g, ch, channels[ch].saved, sf[ch], illegal[ch]);
        requantize(q, xr[ch], g, bands[ch], sf[ch], huffman(bits, g, end, q));
      }
      bits.pos = end;
    }
    if (list.length === 2 && usable) stereo(head, xr, list[1], bands[1], sf[1], illegal[1]);
    return bands;
  };
  return (bytes, at, head, size) => {
    const { begin, granules, valid, end } = sideInfo(bytes, at, head);
    const own = bytes.subarray(end, at + size);
    const total = reservoir.length + own.length;
    const main = new Uint8Array(total + 4);
    main.set(reservoir);
    main.set(own, reservoir.length);
    const bits = new Bits(main, (reservoir.length - begin) * 8);
    const outputs = mono ? 1 : head.channels;
    const pcm = Array.from({ length: outputs }, () => new Float32Array(head.samples));
    granules.forEach((list, gr) => {
      const bands = spectrum(bits, head, list, valid && bits.pos >= 0);
      list.forEach((g, ch) => {
        if (g.kind !== 0) reorder(xr[ch], bands[ch], work.short);
        antialias(xr[ch], bands[ch].longs);
        hybrid(xr[ch], g, bands[ch], channels[ch].overlap, work);
        if (outputs === list.length) synthesize(channels[ch], work.out, pcm[ch], gr * 576);
        else if (ch === 0) work.mix.set(work.out);
        else {
          for (let i = 0; i < 576; i++) work.mix[i] = (work.mix[i] + work.out[i]) / 2;
          synthesize(channels[0], work.mix, pcm[0], gr * 576);
        }
      });
    });
    const keep = Math.min(RESERVOIR, total, Math.max(0, (total * 8 - bits.pos) >> 3));
    reservoir = main.slice(total - keep, total);
    return pcm;
  };
}

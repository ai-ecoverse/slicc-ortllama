import { huffmanCodes } from './layer3.js';
import { mdct } from './mdct.js';
import { HUFFMAN, LONG, WINDOW } from './mp3-tables.js';
import { createResampler } from './pcm.js';

const RATES = [44100, 48000, 32000, 22050, 24000, 16000, 11025, 12000, 8000];
const KBPS = [
  [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
  [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];
const SHAPES = [
  [1, 1],
  [2, 2],
  [3, 2],
  [5, 3],
  [6, 3],
  [7, 5],
  [8, 5],
  [9, 5],
  [10, 7],
  [11, 7],
  [12, 7],
  [13, 15],
  [15, 15],
];
const LINBITS = [1, 2, 3, 4, 6, 8, 10, 13];
const ALIAS = [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037];
const CS = ALIAS.map((c) => 1 / Math.sqrt(1 + c * c));
const CA = ALIAS.map((c) => c / Math.sqrt(1 + c * c));
const MAX_VALUE = 8206;
const MIN_GAIN = 140;

let shared;

function tables() {
  if (shared) return shared;
  const lengths = {};
  const codes = {};
  for (const n of [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24, 32, 33]) {
    lengths[n] = new Int32Array(256);
    codes[n] = new Int32Array(256);
    for (const leaf of huffmanCodes(HUFFMAN[n])) {
      lengths[n][leaf.symbol] = leaf.len;
      codes[n][leaf.symbol] = leaf.code;
    }
  }
  const window = new Float64Array(512);
  for (let i = 0; i <= 256; i++) window[i] = WINDOW[i] / 65536 / 32;
  for (let i = 1; i < 256; i++) window[512 - i] = (i % 64 === 0 ? 1 : -1) * window[i];
  const matrix = new Float64Array(32 * 64);
  for (let k = 0; k < 32; k++)
    for (let i = 0; i < 64; i++)
      matrix[k * 64 + i] = Math.cos(((2 * k + 1) * (i - 16) * Math.PI) / 64);
  const sine = Float64Array.from(
    { length: 36 },
    (_, i) => Math.sin((Math.PI / 36) * (i + 0.5)) / 9
  );
  shared = { lengths, codes, window, matrix, sine };
  return shared;
}

export function mp3Rate(rate) {
  return RATES.reduce((best, r) => (Math.abs(r - rate) < Math.abs(best - rate) ? r : best));
}

class Writer {
  constructor(size) {
    this.bytes = new Uint8Array(size);
    this.pos = 0;
  }

  put(value, n) {
    for (let i = n - 1; i >= 0; i--) {
      if ((value >>> i) & 1) this.bytes[this.pos >> 3] |= 0x80 >> (this.pos & 7);
      this.pos++;
    }
  }
}

function analysis(state, input, at, out) {
  const { window, matrix } = tables();
  const { x, y } = state;
  for (let t = 0; t < 18; t++) {
    x.copyWithin(32, 0, 480);
    for (let i = 0; i < 32; i++) x[31 - i] = input[at + 32 * t + i];
    for (let i = 0; i < 64; i++) {
      let sum = 0;
      for (let j = i; j < 512; j += 64) sum += window[j] * x[j];
      y[i] = sum;
    }
    for (let k = 0; k < 32; k++) {
      let sum = 0;
      for (let i = 0, row = k * 64; i < 64; i++) sum += matrix[row + i] * y[i];
      out[k * 18 + t] = k & 1 && t & 1 ? -sum : sum;
    }
  }
}

function hybrid(state, bands, xr) {
  const { sine } = tables();
  const { previous, block, coefficients } = state;
  for (let sb = 0; sb < 32; sb++) {
    for (let i = 0; i < 18; i++) {
      block[i] = previous[sb * 18 + i] * sine[i];
      block[18 + i] = bands[sb * 18 + i] * sine[18 + i];
      previous[sb * 18 + i] = bands[sb * 18 + i];
    }
    mdct(block, coefficients, 36);
    for (let k = 0; k < 18; k++) xr[sb * 18 + k] = coefficients[k];
  }
  for (let sb = 1; sb < 32; sb++) {
    const at = sb * 18;
    for (let i = 0; i < 8; i++) {
      const lower = xr[at - 1 - i];
      const upper = xr[at + i];
      xr[at - 1 - i] = lower * CS[i] + upper * CA[i];
      xr[at + i] = upper * CS[i] - lower * CA[i];
    }
  }
}

function quantize(magnitude, gain, ix) {
  const factor = 2 ** ((-3 * (gain - 210)) / 16);
  let max = 0;
  for (let i = 0; i < 576; i++) {
    const q = Math.floor(magnitude[i] * factor + 0.4054);
    ix[i] = q;
    if (q > max) max = q;
  }
  return max;
}

const NARROW = 13;
const INVALID = 1e9;
const WIDE = [4, 5, 6, 7, 8, 9, 11, 13];

function prefixes(ix, big, cum) {
  const { lengths } = tables();
  const pairs = big >> 1;
  for (const row of cum) row[0] = 0;
  for (let p = 0; p < pairs; p++) {
    const x = ix[2 * p];
    const y = ix[2 * p + 1];
    const signs = (x ? 1 : 0) + (y ? 1 : 0);
    const peak = Math.max(x, y);
    for (let c = 0; c < NARROW; c++) {
      const [table, limit] = SHAPES[c];
      cum[c][p + 1] = cum[c][p] + (peak > limit ? INVALID : lengths[table][(x << 4) | y] + signs);
    }
    const ex = Math.min(x, 15);
    const ey = Math.min(y, 15);
    const symbol = (ex << 4) | ey;
    cum[NARROW][p + 1] = cum[NARROW][p] + lengths[16][symbol] + signs;
    cum[NARROW + 1][p + 1] = cum[NARROW + 1][p] + lengths[24][symbol] + signs;
    cum[NARROW + 2][p + 1] = cum[NARROW + 2][p] + (x >= 15 ? 1 : 0) + (y >= 15 ? 1 : 0);
    cum[NARROW + 3][p + 1] = cum[NARROW + 3][p] + (peak ? 1 : 0);
  }
}

function escapes(max) {
  const narrow = LINBITS.findIndex((l) => 15 + 2 ** l - 1 >= max);
  const wide = WIDE.findIndex((l) => 15 + 2 ** l - 1 >= max);
  return [Math.max(0, narrow), Math.max(0, wide)];
}

function region(cum, from, to, wide) {
  const a = from >> 1;
  const b = to >> 1;
  if (b <= a || cum[NARROW + 3][b] === cum[NARROW + 3][a]) return { select: 0, bits: 0 };
  let best = { select: 0, bits: Infinity };
  for (let c = 0; c < NARROW; c++) {
    const bits = cum[c][b] - cum[c][a];
    if (bits < best.bits) best = { select: SHAPES[c][0], bits };
  }
  const escaped = cum[NARROW + 2][b] - cum[NARROW + 2][a];
  const narrow = cum[NARROW][b] - cum[NARROW][a] + escaped * LINBITS[wide[0]];
  if (narrow < best.bits) best = { select: 16 + wide[0], bits: narrow };
  const broad = cum[NARROW + 1][b] - cum[NARROW + 1][a] + escaped * WIDE[wide[1]];
  if (broad < best.bits) best = { select: 24 + wide[1], bits: broad };
  return best;
}

function layout(ix, edges, cum) {
  let top = 576;
  while (top > 0 && ix[top - 1] === 0) top--;
  top += top & 1;
  let big = top;
  while (big >= 4 && ix[big - 1] <= 1 && ix[big - 2] <= 1 && ix[big - 3] <= 1 && ix[big - 4] <= 1)
    big -= 4;
  const { lengths } = tables();
  const quads = [0, 0];
  for (let i = big; i < top; i += 4) {
    const v = (ix[i] << 3) | (ix[i + 1] << 2) | (ix[i + 2] << 1) | ix[i + 3];
    const signs = ix[i] + ix[i + 1] + ix[i + 2] + ix[i + 3];
    quads[0] += lengths[32][v] + signs;
    quads[1] += 4 + signs;
  }
  const count1 = quads[1] < quads[0] ? 1 : 0;
  let max = 0;
  for (let i = 0; i < big; i++) if (ix[i] > max) max = ix[i];
  const wide = escapes(max);
  prefixes(ix, big, cum);
  let best = null;
  for (let r0 = 0; r0 < 16; r0++) {
    const a = Math.min(big, edges[r0 + 1]);
    const first = region(cum, 0, a, wide);
    for (let r1 = 0; r1 < 8; r1++) {
      const b = Math.min(big, edges[Math.min(22, r0 + r1 + 2)]);
      const second = region(cum, a, b, wide);
      const third = region(cum, b, big, wide);
      const bits = first.bits + second.bits + third.bits;
      if (!best || bits < best.bits)
        best = { bits, r0, r1, tables: [first.select, second.select, third.select] };
      if (b >= big) break;
    }
    if (a >= big) break;
  }
  return { ...best, bits: best.bits + quads[count1], big, top, count1 };
}

function writePair(w, ix, sign, i, select) {
  const { lengths, codes } = tables();
  const table = select < 16 ? select : select < 24 ? 16 : 24;
  const linbits = select < 16 ? 0 : select < 24 ? LINBITS[select - 16] : WIDE[select - 24];
  const x = linbits ? Math.min(ix[i], 15) : ix[i];
  const y = linbits ? Math.min(ix[i + 1], 15) : ix[i + 1];
  const symbol = (x << 4) | y;
  w.put(codes[table][symbol], lengths[table][symbol]);
  if (x === 15 && linbits) w.put(ix[i] - 15, linbits);
  if (x) w.put(sign[i], 1);
  if (y === 15 && linbits) w.put(ix[i + 1] - 15, linbits);
  if (y) w.put(sign[i + 1], 1);
}

function writeHuffman(w, ix, sign, g, edges) {
  const { lengths, codes } = tables();
  const bounds = [edges[g.r0 + 1], edges[Math.min(22, g.r0 + g.r1 + 2)]];
  for (let i = 0; i < g.big; i += 2) {
    const select = g.tables[i < bounds[0] ? 0 : i < bounds[1] ? 1 : 2];
    if (select) writePair(w, ix, sign, i, select);
  }
  const quad = g.count1 ? 33 : 32;
  for (let i = g.big; i < g.top; i += 4) {
    const v = (ix[i] << 3) | (ix[i + 1] << 2) | (ix[i + 2] << 1) | ix[i + 3];
    w.put(codes[quad][v], lengths[quad][v]);
    for (let k = 0; k < 4; k++) if (ix[i + k]) w.put(sign[i + k], 1);
  }
}

function fit(xr, budget, state) {
  const { magnitude, sign, ix } = state;
  let peak = 0;
  for (let i = 0; i < 576; i++) {
    const a = Math.abs(xr[i]);
    magnitude[i] = a ** 0.75;
    sign[i] = xr[i] < 0 ? 1 : 0;
    if (a > peak) peak = a;
  }
  if (peak === 0) {
    ix.fill(0);
    return { gain: 210, ...layout(ix, state.edges, state.cum) };
  }
  let lo = MIN_GAIN;
  let hi = 255;
  let found = null;
  while (lo <= hi) {
    const gain = (lo + hi) >> 1;
    const max = quantize(magnitude, gain, ix);
    const g = max > MAX_VALUE ? null : layout(ix, state.edges, state.cum);
    if (g && g.bits <= budget) {
      found = gain;
      hi = gain - 1;
    } else lo = gain + 1;
  }
  const gain = found ?? 255;
  quantize(magnitude, gain, ix);
  return { gain, ...layout(ix, state.edges, state.cum) };
}

export function createMp3Encoder(input, { kbps = 64 } = {}) {
  const rate = mp3Rate(input);
  const order = RATES.indexOf(rate);
  const mpeg1 = order < 3;
  const version = mpeg1 ? 3 : order < 6 ? 2 : 0;
  const index = KBPS[mpeg1 ? 0 : 1].indexOf(kbps);
  const granules = mpeg1 ? 2 : 1;
  const samples = 576 * granules;
  const slots = (mpeg1 ? 144000 : 72000) * kbps;
  const side = mpeg1 ? 17 : 9;
  const table = mpeg1 ? 5 + (order % 3) : version === 2 ? 2 + (order % 3) : order === 8 ? 1 : 0;
  const edges = LONG[table].reduce((list, w) => [...list, list.at(-1) + w], [0]);
  const resample = createResampler(input, rate);
  const state = {
    x: new Float64Array(512),
    y: new Float64Array(64),
    previous: new Float64Array(576),
    block: new Float64Array(36),
    coefficients: new Float64Array(18),
    magnitude: new Float64Array(576),
    sign: new Uint8Array(576),
    ix: new Int32Array(576),
    cum: Array.from({ length: NARROW + 4 }, () => new Float64Array(289)),
    edges,
  };
  const bands = new Float64Array(576);
  const xr = new Float64Array(576);
  let pending = new Float32Array(0);
  let rest = 0;
  let total = 0;
  let written = 0;
  const frame = (pcm) => {
    rest += slots % rate;
    const padding = rest >= rate ? 1 : 0;
    if (padding) rest -= rate;
    const size = Math.floor(slots / rate) + padding;
    const w = new Writer(size);
    w.put(0xffe | (version >> 1), 12);
    w.put(((version & 1) << 3) | (1 << 1) | 1, 4);
    w.put(index, 4);
    w.put(order % 3, 2);
    w.put(padding, 1);
    w.put(0, 1);
    w.put(3, 2);
    w.put(0, 2);
    w.put(0, 4);
    const budget = Math.min(4095, Math.floor(((size - 4 - side) * 8) / granules));
    const info = [];
    const main = new Writer(size);
    for (let gr = 0; gr < granules; gr++) {
      analysis(state, pcm, gr * 576, bands);
      hybrid(state, bands, xr);
      const g = fit(xr, budget, state);
      info.push(g);
      const start = main.pos;
      writeHuffman(main, state.ix, state.sign, g, edges);
      g.part23 = main.pos - start;
    }
    w.put(0, mpeg1 ? 9 : 8);
    w.put(0, mpeg1 ? 5 : 1);
    if (mpeg1) w.put(0, 4);
    for (const g of info) {
      w.put(g.part23, 12);
      w.put(g.big >> 1, 9);
      w.put(g.gain, 8);
      w.put(0, mpeg1 ? 4 : 9);
      w.put(0, 1);
      for (const t of g.tables) w.put(t, 5);
      w.put(g.r0, 4);
      w.put(g.r1, 3);
      if (mpeg1) w.put(0, 1);
      w.put(0, 1);
      w.put(g.count1, 1);
    }
    for (let i = 0; i < Math.ceil(main.pos / 8); i++) w.bytes[4 + side + i] = main.bytes[i];
    return w.bytes;
  };
  const drain = (samplesIn) => {
    const joined = new Float32Array(pending.length + samplesIn.length);
    joined.set(pending);
    joined.set(samplesIn, pending.length);
    const out = [];
    let at = 0;
    for (; at + samples <= joined.length; at += samples)
      out.push(frame(joined.subarray(at, at + samples)));
    pending = joined.slice(at);
    const bytes = new Uint8Array(out.reduce((n, f) => n + f.length, 0));
    let k = 0;
    for (const f of out) {
      bytes.set(f, k);
      k += f.length;
    }
    written += out.length * samples;
    return bytes;
  };
  return {
    rate,
    push(input) {
      const resampled = resample(input);
      total += resampled.length;
      return drain(resampled);
    },
    end() {
      const tail = resample.end();
      total += tail.length;
      const delay = 481 + 576;
      const need = Math.ceil((total + delay) / samples) * samples;
      return drain(
        Float32Array.from({ length: need - written - pending.length }, (_, i) =>
          i < tail.length ? tail[i] : 0
        )
      );
    },
  };
}

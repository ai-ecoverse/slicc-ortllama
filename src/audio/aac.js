import { LONG, SCALEFACTOR, SHORT, SPECTRAL, TNS_LONG, TNS_SHORT } from './aac-tables.js';
import { imdct } from './mdct.js';
import { AudioFormatError } from './pcm.js';

export const RATES = [
  96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];
const BAND_RATES = [92017, 75132, 55426, 46009, 37566, 27713, 23004, 18783, 13856, 11502, 9391, 0];
const TRAILERS = ['TAG', 'ID3', 'APETAGEX'];
const LC = new Set([1, 2, 4]);
const EIGHT_SHORT = 2;
const LONG_START = 1;
const LONG_STOP = 3;
const NOISE = 13;
const INTENSITY_OUT = 14;
const INTENSITY = 15;
const UNSIGNED = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1];
const MODS = [0, 3, 3, 3, 3, 9, 9, 8, 8, 13, 13, 17];
const OFFS = [0, 1, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0];
const TNS_ORDER = 32;

function corrupt() {
  return new AudioFormatError('the AAC stream is corrupt');
}

function unsupported(what) {
  return new AudioFormatError(`${what} is not supported; send AAC-LC or HE-AAC`);
}

class Bits {
  constructor(bytes) {
    this.bytes = bytes;
    this.pos = 0;
    this.limit = bytes.length * 8;
  }

  bit() {
    if (this.pos >= this.limit) throw corrupt();
    const b = (this.bytes[this.pos >> 3] >> (7 - (this.pos & 7))) & 1;
    this.pos++;
    return b;
  }

  read(n) {
    let v = 0;
    for (let i = 0; i < n; i++) v = v * 2 + this.bit();
    return v;
  }

  skip(n) {
    if (this.pos + n > this.limit) throw corrupt();
    this.pos += n;
  }

  align() {
    this.pos = (this.pos + 7) & ~7;
  }
}

function huffman(bits, { counts, symbols }) {
  let code = 0;
  let first = 0;
  let index = 0;
  for (let i = 0; ; i++) {
    code |= bits.bit();
    if (code - first < counts[i]) return symbols[index + code - first];
    index += counts[i];
    first = (first + counts[i]) << 1;
    code <<= 1;
  }
}

function bessel(x) {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 50; k++) {
    term *= (x / 2 / k) ** 2;
    sum += term;
  }
  return sum;
}

function kbd(n, alpha) {
  const half = n / 2;
  const kernel = Float64Array.from({ length: half + 1 }, (_, j) =>
    bessel(Math.PI * alpha * Math.sqrt(1 - ((4 * j) / n - 1) ** 2))
  );
  const total = kernel.reduce((a, b) => a + b, 0);
  const out = new Float64Array(half);
  let run = 0;
  for (let i = 0; i < half; i++) {
    run += kernel[i];
    out[i] = Math.sqrt(run / total);
  }
  return out;
}

const sine = (n) =>
  Float64Array.from({ length: n / 2 }, (_, i) => Math.sin((Math.PI / n) * (i + 0.5)));
const RISE_LONG = [sine(2048), kbd(2048, 4)];
const RISE_SHORT = [sine(256), kbd(256, 6)];
const POW43 = Float64Array.from({ length: 8192 }, (_, i) => i ** (4 / 3));
const windows = new Map();

function longWindow(sequence, prev, shape) {
  const key = sequence * 4 + prev * 2 + shape;
  if (windows.has(key)) return windows.get(key);
  const w = new Float64Array(2048);
  if (sequence === LONG_STOP) {
    w.set(RISE_SHORT[prev], 448);
    w.fill(1, 576, 1024);
  } else w.set(RISE_LONG[prev]);
  if (sequence === LONG_START) {
    w.fill(1, 1024, 1472);
    w.set(RISE_SHORT[shape].slice().reverse(), 1472);
  } else w.set(RISE_LONG[shape].slice().reverse(), 1024);
  windows.set(key, w);
  return w;
}

export function audioSpecificConfig(bytes) {
  const bits = new Bits(bytes);
  const object = () => {
    const o = bits.read(5);
    return o === 31 ? 32 + bits.read(6) : o;
  };
  const rate = () => {
    const index = bits.read(4);
    return index === 15 ? bits.read(24) : (RATES[index] ?? 0);
  };
  let type = object();
  const core = rate();
  const channels = bits.read(4);
  if (type === 5 || type === 29) {
    rate();
    type = object();
  }
  if (!LC.has(type)) throw unsupported(`AAC object type ${type}`);
  if (bits.read(1)) throw unsupported('AAC with 960-sample frames');
  if (!core) throw corrupt();
  return { object: type, index: bandIndex(core), rate: core, channels };
}

export function bandIndex(rate) {
  return BAND_RATES.findIndex((least) => rate >= least);
}

const sync = (bytes, at) => bytes[at] === 0xff && (bytes[at + 1] & 0xf6) === 0xf0;

const frameLength = (b, at) => ((b[at + 3] & 3) << 11) | (b[at + 4] << 3) | (b[at + 5] >> 5);

export function isAdts(bytes) {
  if (bytes.length < 7 || !sync(bytes, 0) || ((bytes[2] >> 2) & 15) > 12) return false;
  const length = frameLength(bytes, 0);
  return length >= 7 && (length === bytes.length || sync(bytes, length));
}

const trailer = (bytes) =>
  TRAILERS.some((tag) => String.fromCharCode(...bytes.subarray(0, tag.length)) === tag);

export function adtsFrames(bytes) {
  const frames = [];
  const blocks = [];
  let config = null;
  let at = 0;
  while (at + 7 <= bytes.length && sync(bytes, at)) {
    const header = bytes[at + 1] & 1 ? 7 : 9;
    const length = frameLength(bytes, at);
    const index = (bytes[at + 2] >> 2) & 15;
    const count = (bytes[at + 6] & 3) + 1;
    if (length < header || at + length > bytes.length || index > 12) throw corrupt();
    if (count > 1 && header === 9) throw unsupported('ADTS with CRC-protected blocks');
    config ??= { object: (bytes[at + 2] >> 6) + 1, index, rate: RATES[index], channels: 0 };
    if (!LC.has(config.object)) throw unsupported(`AAC object type ${config.object}`);
    if (index !== config.index) throw corrupt();
    frames.push(bytes.subarray(at + header, at + length));
    blocks.push(count);
    at += length;
  }
  if (!config || (at < bytes.length && !trailer(bytes.subarray(at)))) throw corrupt();
  return { config: { ...config, index: bandIndex(config.rate) }, frames, blocks };
}

function icsInfo(bits, ics, index) {
  bits.read(1);
  ics.sequence = bits.read(2);
  ics.shape = bits.read(1);
  ics.groups = [1];
  const short = ics.sequence === EIGHT_SHORT;
  ics.maxSfb = bits.read(short ? 4 : 6);
  ics.offsets = (short ? SHORT : LONG)[index];
  ics.tnsBands = (short ? TNS_SHORT : TNS_LONG)[index];
  if (short) {
    const grouping = bits.read(7);
    for (let i = 6; i >= 0; i--) {
      if ((grouping >> i) & 1) ics.groups[ics.groups.length - 1]++;
      else ics.groups.push(1);
    }
  } else if (bits.read(1)) throw unsupported('AAC prediction');
  if (ics.maxSfb > ics.offsets.length - 1) throw corrupt();
}

function eachBand(ics, fn) {
  let win = 0;
  for (let g = 0; g < ics.groups.length; g++) {
    for (let sfb = 0; sfb < ics.maxSfb; sfb++) {
      const at = g * 64 + sfb;
      for (let w = win; w < win + ics.groups[g]; w++)
        fn(at, w, ics.offsets[sfb], ics.offsets[sfb + 1], sfb);
    }
    win += ics.groups[g];
  }
}

function eachGroupBand(ics, fn) {
  for (let g = 0; g < ics.groups.length; g++)
    for (let sfb = 0; sfb < ics.maxSfb; sfb++) fn(g * 64 + sfb);
}

function channel() {
  return {
    ics: {},
    books: new Uint8Array(512),
    factors: new Int32Array(512),
    quant: new Int32Array(1024),
    spec: new Float64Array(1024),
    seeds: new Uint32Array(1024),
    tns: null,
  };
}

function sections(bits, ch) {
  const { ics } = ch;
  const width = ics.sequence === EIGHT_SHORT ? 3 : 5;
  const escape = (1 << width) - 1;
  ch.books.fill(0);
  for (let g = 0; g < ics.groups.length; g++) {
    for (let k = 0; k < ics.maxSfb; ) {
      const book = bits.read(4);
      let length = 0;
      for (let step = escape; step === escape; length += step) step = bits.read(width);
      if (book === 12 || length === 0 || k + length > ics.maxSfb) throw corrupt();
      ch.books.fill(book, g * 64 + k, g * 64 + k + length);
      k += length;
    }
  }
}

function scalefactors(bits, ch, gain) {
  const delta = () => huffman(bits, SCALEFACTOR) - 60;
  const state = { factor: gain, position: 0, noise: gain - 90, first: true };
  eachGroupBand(ch.ics, (at) => {
    const book = ch.books[at];
    if (book === 0) ch.factors[at] = 0;
    else if (book === INTENSITY || book === INTENSITY_OUT) {
      state.position += delta();
      ch.factors[at] = state.position;
    } else if (book === NOISE) {
      state.noise += state.first ? bits.read(9) - 256 : delta();
      state.first = false;
      ch.factors[at] = state.noise;
    } else {
      state.factor += delta();
      if (state.factor < 0 || state.factor > 255) throw corrupt();
      ch.factors[at] = state.factor;
    }
  });
}

function pulses(bits, ch) {
  const count = bits.read(2) + 1;
  let k = ch.ics.offsets[bits.read(6)];
  const out = [];
  for (let i = 0; i < count; i++) {
    k += bits.read(5);
    out.push([k, bits.read(4)]);
  }
  if (ch.ics.sequence === EIGHT_SHORT || !(k < 1024)) throw corrupt();
  return out;
}

function tnsFilter(bits, short, resolution) {
  const length = bits.read(short ? 4 : 6);
  const order = bits.read(short ? 3 : 5);
  const filter = { length, order, down: 0, coefs: [] };
  if (!order) return filter;
  filter.down = bits.read(1);
  const width = resolution - bits.read(1);
  const pos = ((1 << (resolution - 1)) - 0.5) / (Math.PI / 2);
  const neg = ((1 << (resolution - 1)) + 0.5) / (Math.PI / 2);
  for (let i = 0; i < order; i++) {
    let v = bits.read(width);
    if (v >= 1 << (width - 1)) v -= 1 << width;
    filter.coefs.push(Math.sin(v / (v >= 0 ? pos : neg)));
  }
  return filter;
}

function tnsData(bits, ics) {
  const short = ics.sequence === EIGHT_SHORT;
  return Array.from({ length: short ? 8 : 1 }, () => {
    const count = bits.read(short ? 1 : 2);
    const resolution = count ? bits.read(1) + 3 : 0;
    return Array.from({ length: count }, () => tnsFilter(bits, short, resolution));
  });
}

function values(bits, book, quant, at) {
  const dim = book < 5 ? 4 : 2;
  const mod = MODS[book];
  let symbol = huffman(bits, SPECTRAL[book]);
  for (let i = dim - 1; i >= 0; i--) {
    quant[at + i] = (symbol % mod) - OFFS[book];
    symbol = Math.floor(symbol / mod);
  }
  if (!UNSIGNED[book]) return;
  for (let i = 0; i < dim; i++) if (quant[at + i] && bits.bit()) quant[at + i] = -quant[at + i];
  if (book !== 11) return;
  for (let i = 0; i < dim; i++) {
    if (Math.abs(quant[at + i]) !== 16) continue;
    let n = 4;
    while (bits.bit()) if (++n > 12) throw corrupt();
    const value = (1 << n) + bits.read(n);
    quant[at + i] = quant[at + i] < 0 ? -value : value;
  }
}

function spectral(bits, ch) {
  ch.quant.fill(0);
  eachBand(ch.ics, (at, w, start, end) => {
    const book = ch.books[at];
    if (book === 0 || book >= NOISE) return;
    for (let k = start; k < end; k += book < 5 ? 4 : 2) values(bits, book, ch.quant, w * 128 + k);
  });
}

function random(state) {
  return (Math.imul(state, 1664525) + 1013904223) >>> 0;
}

function noise(ch, rng, band, shared) {
  const { at, w, start, end, sfb } = band;
  const { spec } = ch;
  const base = w * 128;
  let seed = shared ? shared.seeds[w * 64 + sfb] : rng.state;
  ch.seeds[w * 64 + sfb] = seed;
  let energy = 0;
  for (let k = base + start; k < base + end; k++) {
    seed = random(seed);
    spec[k] = (seed | 0) / 2147483648;
    energy += spec[k] ** 2;
  }
  if (!shared) rng.state = seed;
  const scale = 2 ** (0.25 * ch.factors[at]) / Math.sqrt(energy);
  for (let k = base + start; k < base + end; k++) spec[k] *= scale;
}

function dequantize(ch, rng, partner = null, used = null) {
  const { quant, spec } = ch;
  spec.fill(0);
  eachBand(ch.ics, (at, w, start, end, sfb) => {
    const book = ch.books[at];
    if (book === NOISE) {
      const shared = partner?.books[at] === NOISE && used(at) ? partner : null;
      noise(ch, rng, { at, w, start, end, sfb }, shared);
    } else if (book && book < NOISE) {
      const gain = 2 ** (0.25 * (ch.factors[at] - 100));
      for (let k = w * 128 + start; k < w * 128 + end; k++) {
        const q = quant[k];
        spec[k] = (q < 0 ? -POW43[-q] : POW43[q]) * gain;
      }
    }
  });
}

function stereo(left, right, mode, used) {
  eachBand(right.ics, (at, w, start, end) => {
    const book = right.books[at];
    const from = w * 128 + start;
    const to = w * 128 + end;
    if (book === INTENSITY || book === INTENSITY_OUT) {
      const sign = (book === INTENSITY ? 1 : -1) * (mode === 1 && used(at) ? -1 : 1);
      const scale = sign * 2 ** (-0.25 * right.factors[at]);
      for (let k = from; k < to; k++) right.spec[k] = left.spec[k] * scale;
    } else if (used(at) && left.books[at] !== NOISE) {
      for (let k = from; k < to; k++) {
        const l = left.spec[k];
        left.spec[k] = l + right.spec[k];
        right.spec[k] = l - right.spec[k];
      }
    }
  });
}

function lpcOf(coefs) {
  const lpc = [1];
  for (let m = 1; m <= coefs.length; m++) {
    const prev = lpc.slice();
    for (let i = 1; i < m; i++) lpc[i] = prev[i] + coefs[m - 1] * prev[m - i];
    lpc[m] = coefs[m - 1];
  }
  return lpc;
}

function filter(spec, from, step, count, lpc, state) {
  const order = lpc.length - 1;
  state.fill(0);
  for (let n = 0, at = from; n < count; n++, at += step) {
    let y = spec[at];
    for (let i = 0; i < order; i++) y -= lpc[i + 1] * state[i];
    state.copyWithin(1, 0, order - 1);
    state[0] = y;
    spec[at] = y;
  }
}

function tns(ch, state) {
  const { ics, spec } = ch;
  const limit = Math.min(ics.tnsBands, ics.maxSfb);
  ch.tns?.forEach((filters, w) => {
    let bottom = ics.offsets.length - 1;
    for (const { length, order, down, coefs } of filters) {
      const top = bottom;
      bottom = Math.max(top - length, 0);
      const start = ics.offsets[Math.min(bottom, limit)];
      const end = ics.offsets[Math.min(top, limit)];
      if (!order || end <= start) continue;
      const from = w * 128 + (down ? end - 1 : start);
      filter(spec, from, down ? -1 : 1, end - start, lpcOf(coefs), state);
    }
  });
}

function filterbank(ch, state, buffers) {
  const { ics, spec } = ch;
  const { long, short, windowed } = buffers;
  if (ics.sequence === EIGHT_SHORT) {
    windowed.fill(0);
    const fall = RISE_SHORT[ics.shape];
    for (let w = 0; w < 8; w++) {
      imdct(spec.subarray(w * 128, w * 128 + 128), short, 256);
      const rise = RISE_SHORT[w === 0 ? state.shape : ics.shape];
      const at = 448 + w * 128;
      for (let j = 0; j < 128; j++) {
        windowed[at + j] += (short[j] * rise[j]) / 128;
        windowed[at + 128 + j] += (short[128 + j] * fall[127 - j]) / 128;
      }
    }
  } else {
    imdct(spec, long, 2048);
    const w = longWindow(ics.sequence, state.shape, ics.shape);
    for (let n = 0; n < 2048; n++) windowed[n] = (long[n] * w[n]) / 1024;
  }
  const out = new Float64Array(1024);
  for (let n = 0; n < 1024; n++) out[n] = windowed[n] + state.overlap[n];
  state.overlap.set(windowed.subarray(1024));
  state.shape = ics.shape;
  return out;
}

function skipPce(bits) {
  bits.read(10);
  const counts = [
    bits.read(4),
    bits.read(4),
    bits.read(4),
    bits.read(2),
    bits.read(3),
    bits.read(4),
  ];
  for (let i = 0; i < 3; i++) if (bits.read(1)) bits.read(i === 2 ? 3 : 4);
  const widths = [5, 5, 5, 4, 4, 5];
  bits.skip(counts.reduce((n, c, i) => n + c * widths[i], 0));
  bits.align();
  bits.skip(bits.read(8) * 8);
}

function skipDse(bits) {
  bits.read(4);
  const align = bits.read(1);
  let count = bits.read(8);
  if (count === 255) count += bits.read(8);
  if (align) bits.align();
  bits.skip(count * 8);
}

function skipFil(bits) {
  let count = bits.read(4);
  if (count === 15) count += bits.read(8) - 1;
  bits.skip(count * 8);
}

function ics(bits, ch, common, index) {
  const gain = bits.read(8);
  if (!common) icsInfo(bits, ch.ics, index);
  sections(bits, ch);
  scalefactors(bits, ch, gain);
  const pulse = bits.read(1) ? pulses(bits, ch) : [];
  ch.tns = bits.read(1) ? tnsData(bits, ch.ics) : null;
  if (bits.read(1)) throw unsupported('AAC gain control');
  spectral(bits, ch);
  for (const [k, amp] of pulse) ch.quant[k] += ch.quant[k] < 0 ? -amp : amp;
}

function pairElement(bits, pair, rng, index) {
  const [left, right] = pair;
  const mask = pair.mask;
  bits.read(4);
  const common = bits.read(1);
  let mode = 0;
  if (common) {
    icsInfo(bits, left.ics, index);
    Object.assign(right.ics, left.ics);
    mode = bits.read(2);
    if (mode === 3) throw corrupt();
    if (mode === 1) eachGroupBand(left.ics, (at) => (mask[at] = bits.read(1)));
  }
  ics(bits, left, common, index);
  ics(bits, right, common, index);
  const used = (at) => mode === 2 || (mode === 1 && mask[at] === 1);
  dequantize(left, rng);
  dequantize(right, rng, left, used);
  if (common) stereo(left, right, mode, used);
  return [left, right];
}

export function createAacDecoder(config) {
  const states = [];
  const rng = { state: 1 };
  const buffers = {
    long: new Float64Array(2048),
    short: new Float64Array(256),
    windowed: new Float64Array(2048),
    tns: new Float64Array(TNS_ORDER),
  };
  const pair = [channel(), channel()];
  pair.mask = new Uint8Array(512);
  const single = (bits, lfe) => {
    bits.read(4);
    ics(bits, pair[0], false, config.index);
    dequantize(pair[0], rng);
    return lfe ? [] : [pair[0]];
  };
  const elements = [
    (bits) => single(bits, false),
    (bits) => pairElement(bits, pair, rng, config.index),
    () => {
      throw unsupported('AAC coupling channels');
    },
    (bits) => single(bits, true),
    skipDse,
    skipPce,
    skipFil,
  ];
  const block = (bits, mix) => {
    let slot = 0;
    for (let id = bits.read(3); id !== 7; id = bits.read(3)) {
      for (const ch of elements[id](bits) ?? []) {
        tns(ch, buffers.tns);
        states[slot] ??= { overlap: new Float64Array(1024), shape: 0 };
        const out = filterbank(ch, states[slot++], buffers);
        for (let n = 0; n < 1024; n++) mix[n] += out[n];
      }
    }
    bits.align();
    return slot;
  };
  return (frame, blocks = 1) => {
    const bits = new Bits(frame);
    const samples = new Float32Array(1024 * blocks);
    const mix = new Float64Array(1024);
    for (let b = 0; b < blocks; b++) {
      mix.fill(0);
      const scale = 1 / Math.max(1, block(bits, mix)) / 32768;
      for (let n = 0; n < 1024; n++) samples[b * 1024 + n] = mix[n] * scale;
    }
    return samples;
  };
}

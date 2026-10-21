import { longWindow, RATES, RISE_SHORT } from './aac.js';
import { LONG, SCALEFACTOR, SHORT, SPECTRAL } from './aac-tables.js';
import { mdct } from './mdct.js';
import { createResampler } from './pcm.js';

const FRAME = 1024;
const SHORT_FRAME = 128;
const MAX_BITS = 6144;
const HEADER_BITS = 56;
const MAX_QUANT = 8191;
const ESCAPE = 11;
const BOOKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const LIMITS = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 8191];
const UNSIGNED = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1];
const MODS = [0, 3, 3, 3, 3, 9, 9, 8, 8, 13, 13, 17];
const OFFS = [0, 1, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0];
const ROUNDING = 0.4054;
const FULL = 2 ** 53;
const HIGHEST = 120;
const STEPS = 10;
const ONLY_LONG = 0;
const LONG_START = 1;
const EIGHT_SHORT = 2;
const LONG_STOP = 3;
const ATTACK = 10;
const QUIET = 1e-5;
const GROUPING = 4;

function canonical({ counts, symbols }) {
  const size = Math.max(...symbols) + 1;
  const codes = new Uint32Array(size);
  const lengths = new Uint8Array(size);
  let first = 0;
  let index = 0;
  for (let i = 0; i < counts.length; i++) {
    for (let j = 0; j < counts[i]; j++) {
      codes[symbols[index + j]] = first + j;
      lengths[symbols[index + j]] = i + 1;
    }
    index += counts[i];
    first = (first + counts[i]) << 1;
  }
  return { codes, lengths };
}

let shared;

function tables() {
  if (shared) return shared;
  const rise = RISE_SHORT[0];
  shared = {
    scalefactor: canonical(SCALEFACTOR),
    spectral: SPECTRAL.map((book) => book && canonical(book)),
    windows: [ONLY_LONG, LONG_START, EIGHT_SHORT, LONG_STOP].map((s) => longWindow(s, 0, 0)),
    short: Float64Array.from({ length: 2 * SHORT_FRAME }, (_, n) =>
      n < SHORT_FRAME ? rise[n] : rise[2 * SHORT_FRAME - 1 - n]
    ),
    pow43: Float64Array.from({ length: MAX_QUANT + 1 }, (_, i) => i ** (4 / 3)),
  };
  return shared;
}

export function aacRate(rate) {
  return RATES.slice(0, 12).reduce((best, r) =>
    Math.abs(r - rate) < Math.abs(best - rate) ? r : best
  );
}

function hearing(f) {
  const k = Math.max(f, 50) / 1000;
  return 3.64 * k ** -0.8 - 6.5 * Math.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4;
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

  reset() {
    this.bytes.fill(0, 0, (this.pos + 7) >> 3);
    this.pos = 0;
  }
}

function escapeBits(v) {
  if (v < 16) return 0;
  return 2 * (31 - Math.clz32(v)) - 3;
}

function symbolOf(q, at, book) {
  const dim = book < 5 ? 4 : 2;
  const mod = MODS[book];
  let symbol = 0;
  for (let i = 0; i < dim; i++) {
    const v = q[at + i];
    symbol = symbol * mod + (UNSIGNED[book] ? Math.min(Math.abs(v), 16) : v + OFFS[book]);
  }
  return symbol;
}

function bandBits(q, start, end, book) {
  const { lengths } = tables().spectral[book];
  const dim = book < 5 ? 4 : 2;
  let bits = 0;
  for (let k = start; k < end; k += dim) {
    bits += lengths[symbolOf(q, k, book)];
    if (!UNSIGNED[book]) continue;
    for (let i = 0; i < dim; i++) {
      const a = Math.abs(q[k + i]);
      if (a) bits += 1 + escapeBits(a);
    }
  }
  return bits;
}

function writeBand(w, q, start, end, book) {
  const { codes, lengths } = tables().spectral[book];
  const dim = book < 5 ? 4 : 2;
  for (let k = start; k < end; k += dim) {
    const symbol = symbolOf(q, k, book);
    w.put(codes[symbol], lengths[symbol]);
    if (!UNSIGNED[book]) continue;
    for (let i = 0; i < dim; i++) if (q[k + i]) w.put(q[k + i] < 0 ? 1 : 0, 1);
    if (book !== ESCAPE) continue;
    for (let i = 0; i < dim; i++) {
      const a = Math.abs(q[k + i]);
      if (a < 16) continue;
      const n = 31 - Math.clz32(a);
      w.put((1 << (n - 3)) - 2, n - 3);
      w.put(a - (1 << n), n);
    }
  }
}

function sections(costs, starts, overhead) {
  const count = costs.length;
  const total = Array.from({ length: count }, () => new Float64Array(12));
  const from = Array.from({ length: count }, () => new Int8Array(12));
  for (let u = 0; u < count; u++) {
    let best = 0;
    let bestBook = -1;
    if (u > 0) {
      best = Number.POSITIVE_INFINITY;
      for (let c = 0; c < 12; c++)
        if (total[u - 1][c] < best) {
          best = total[u - 1][c];
          bestBook = c;
        }
    }
    for (let c = 0; c < 12; c++) {
      const stay = u > 0 && !starts[u] ? total[u - 1][c] : Number.POSITIVE_INFINITY;
      const open = best + overhead;
      total[u][c] = costs[u][c] + Math.min(stay, open);
      from[u][c] = stay <= open ? c : bestBook;
    }
  }
  const books = new Uint8Array(count);
  if (!count) return books;
  const last = total[count - 1];
  let c = last.indexOf(Math.min(...last));
  for (let u = count - 1; u >= 0; u--) {
    books[u] = c;
    c = from[u][c];
  }
  return books;
}

function quantizeUnit(state, u) {
  const { units, magnitude, sign, quant, factors, zero, peaks } = state;
  const scale = 2 ** ((-3 * (factors[u] - 100)) / 16);
  let peak = 0;
  for (let k = units[u].start; k < units[u].end; k++) {
    const q = Math.floor(magnitude[k] * scale + ROUNDING);
    quant[k] = sign[k] ? -q : q;
    if (q > peak) peak = q;
  }
  peaks[u] = peak;
  zero[u] = peak === 0 ? 1 : 0;
}

function raise(state, order) {
  const { factors, zero, units } = state;
  const count = units.length;
  let last = -1;
  for (let i = 0; i < count; i++) {
    const u = order > 0 ? i : count - 1 - i;
    if (zero[u]) continue;
    if (last >= 0) factors[u] = Math.max(factors[u], factors[last] - 60);
    last = u;
  }
}

function zeros(state) {
  let n = 0;
  for (let u = 0; u < state.units.length; u++) n += state.zero[u];
  return n;
}

function chain(state) {
  let before;
  do {
    before = zeros(state);
    raise(state, 1);
    raise(state, -1);
    for (let u = 0; u < state.units.length; u++) if (!state.zero[u]) quantizeUnit(state, u);
  } while (zeros(state) !== before);
}

function noiseOf(state, unit, factor) {
  const { magnitude, spec } = state;
  const { pow43 } = tables();
  const scale = 2 ** ((-3 * (factor - 100)) / 16);
  const gain = 2 ** ((factor - 100) / 4);
  let noise = 0;
  for (let k = unit.start; k < unit.end; k++) {
    const q = Math.floor(magnitude[k] * scale + ROUNDING);
    noise += (Math.abs(spec[k]) - pow43[q] * gain) ** 2;
  }
  return noise;
}

function fitUnit(state, u, target) {
  if (state.energy[u] <= target) return 255;
  let lo = state.floors[u];
  let hi = 255;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (noiseOf(state, state.units[u], mid) <= target) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function plan(state, quality) {
  const { units, energy, factors, quant, peaks, zero } = state;
  const ratio = 10 ** (-quality / 10);
  for (let u = 0; u < units.length; u++) {
    const target = ratio * Math.sqrt(energy[u] * (units[u].end - units[u].start) * FULL);
    factors[u] = fitUnit(state, u, Math.max(target, units[u].hearing));
    quantizeUnit(state, u);
  }
  chain(state);
  let top = 0;
  for (let u = 0; u < units.length; u++) if (!zero[u]) top = Math.max(top, units[u].sfb + 1);
  const included = [];
  for (let u = 0; u < units.length; u++) if (units[u].sfb < top) included.push(u);
  const sfBits = tables().scalefactor.lengths[60];
  const costs = included.map((u) => {
    const row = new Float64Array(12);
    row[0] = zero[u] ? 0 : Number.POSITIVE_INFINITY;
    for (const book of BOOKS)
      row[book] =
        peaks[u] > LIMITS[book]
          ? Number.POSITIVE_INFINITY
          : bandBits(quant, units[u].start, units[u].end, book) + (zero[u] ? sfBits : 0);
    return row;
  });
  const starts = included.map((u) => units[u].sfb === 0);
  const width = state.sequence === EIGHT_SHORT ? 3 : 5;
  return { top, included, books: sections(costs, starts, 4 + width) };
}

function writeIcsInfo(w, state, top) {
  w.put(0, 1);
  w.put(state.sequence, 2);
  w.put(0, 1);
  if (state.sequence !== EIGHT_SHORT) {
    w.put(top, 6);
    w.put(0, 1);
    return;
  }
  w.put(top, 4);
  const bits = state.groups.flatMap((size) => Array.from({ length: size }, (_, i) => (i ? 1 : 0)));
  for (const bit of bits.slice(1)) w.put(bit, 1);
}

function writeSections(w, state, included, books) {
  const width = state.sequence === EIGHT_SHORT ? 3 : 5;
  const escape = (1 << width) - 1;
  for (let i = 0; i < included.length; ) {
    let end = i + 1;
    while (end < included.length && books[end] === books[i] && state.units[included[end]].sfb !== 0)
      end++;
    w.put(books[i], 4);
    let length = end - i;
    while (length >= escape) {
      w.put(escape, width);
      length -= escape;
    }
    w.put(length, width);
    i = end;
  }
}

function writeFrame(state, w, { included, books, top }) {
  const { units, quant, factors, zero } = state;
  const { scalefactor } = tables();
  const sf = new Int32Array(included.length);
  let next = -1;
  for (let i = included.length - 1; i >= 0; i--) {
    if (!zero[included[i]]) next = factors[included[i]];
    sf[i] = next;
  }
  let previous = -1;
  for (let i = 0; i < included.length; i++) {
    if (!books[i]) continue;
    if (!zero[included[i]]) previous = factors[included[i]];
    if (previous >= 0) sf[i] = previous;
  }
  const first = books.findIndex((book) => book > 0);
  const gain = first >= 0 ? sf[first] : 100;
  w.put(0, 3);
  w.put(0, 4);
  w.put(gain, 8);
  writeIcsInfo(w, state, top);
  writeSections(w, state, included, books);
  let factor = gain;
  for (let i = 0; i < included.length; i++) {
    if (!books[i]) continue;
    const symbol = sf[i] - factor + 60;
    w.put(scalefactor.codes[symbol], scalefactor.lengths[symbol]);
    factor = sf[i];
  }
  w.put(0, 3);
  for (let i = 0; i < included.length; i++) {
    const unit = units[included[i]];
    if (books[i]) writeBand(w, quant, unit.start, unit.end, books[i]);
  }
  w.put(7, 3);
  w.pos = (w.pos + 7) & ~7;
}

function groupsOf(energies) {
  const groups = [1];
  for (let w = 1; w < 8; w++) {
    const a = energies[w] + QUIET;
    const b = energies[w - 1] + QUIET;
    if (a > b * GROUPING || b > a * GROUPING) groups.push(1);
    else groups[groups.length - 1]++;
  }
  return groups;
}

function layout(config, sequence, groups) {
  const short = sequence === EIGHT_SHORT;
  const offsets = (short ? SHORT : LONG)[config.index];
  const length = short ? SHORT_FRAME : FRAME;
  const units = [];
  let at = 0;
  let window = 0;
  for (let g = 0; g < groups.length; g++) {
    for (let sfb = 0; sfb < offsets.length - 1; sfb++) {
      const width = offsets[sfb + 1] - offsets[sfb];
      let low = Number.POSITIVE_INFINITY;
      for (let k = offsets[sfb]; k < offsets[sfb + 1]; k++)
        low = Math.min(low, config.hearing((k + 0.5) / (2 * length)));
      const sources = [];
      for (let w = window; w < window + groups[g]; w++)
        sources.push(w * SHORT_FRAME + offsets[sfb]);
      units.push({
        sfb,
        width,
        sources,
        start: at,
        end: at + width * groups[g],
        hearing: width * groups[g] * 2 * length * 10 ** (low / 10),
      });
      at += width * groups[g];
    }
    window += groups[g];
  }
  return units;
}

function transform(state, frame) {
  const { windows, short } = tables();
  const { input, raw, block } = state;
  if (state.sequence !== EIGHT_SHORT) {
    const w = windows[state.sequence];
    for (let n = 0; n < 2 * FRAME; n++) input[n] = frame[n] * w[n];
    mdct(input, raw, 2 * FRAME);
    return null;
  }
  const energies = new Float64Array(8);
  for (let w = 0; w < 8; w++) {
    const at = 448 + w * SHORT_FRAME;
    for (let n = 0; n < 2 * SHORT_FRAME; n++) block[n] = frame[at + n] * short[n];
    const out = raw.subarray(w * SHORT_FRAME, (w + 1) * SHORT_FRAME);
    mdct(block, out, 2 * SHORT_FRAME);
    for (const x of out) energies[w] += x * x;
  }
  return energies;
}

function analyse(state, frame) {
  const energies = transform(state, frame);
  const short = state.sequence === EIGHT_SHORT;
  state.groups = short ? groupsOf(energies.map((e) => e / FULL)) : [1];
  state.units = layout(state.config, state.sequence, state.groups);
  const { raw, spec, magnitude, sign, energy, floors, units } = state;
  for (let u = 0; u < units.length; u++) {
    const unit = units[u];
    let k = unit.start;
    let sum = 0;
    let peak = 0;
    for (const from of short ? unit.sources : [unit.sources[0]])
      for (let j = 0; j < unit.width; j++, k++) {
        const x = raw[from + j];
        spec[k] = x;
        magnitude[k] = Math.abs(x) ** 0.75;
        sign[k] = x < 0 ? 1 : 0;
        sum += x * x;
        if (magnitude[k] > peak) peak = magnitude[k];
      }
    energy[u] = sum;
    const need = peak > 0 ? 100 + (16 / 3) * Math.log2(peak / (MAX_QUANT - ROUNDING)) : 0;
    floors[u] = Math.min(255, Math.max(0, Math.ceil(need)));
  }
}

function header(bytes, index, length) {
  bytes[0] = 0xff;
  bytes[1] = 0xf1;
  bytes[2] = (1 << 6) | (index << 2);
  bytes[3] = (1 << 6) | ((length >> 11) & 3);
  bytes[4] = (length >> 3) & 0xff;
  bytes[5] = ((length & 7) << 5) | 0x1f;
  bytes[6] = 0xfc;
}

function detector() {
  const history = new Float64Array(8);
  let last = 0;
  return (block) => {
    const hits = [];
    for (let i = 0; i < 8; i++) {
      let e = 0;
      for (let n = i * SHORT_FRAME; n < (i + 1) * SHORT_FRAME; n++) {
        e += (block[n] - last) ** 2;
        last = block[n];
      }
      e /= SHORT_FRAME;
      const mean = history.reduce((a, b) => a + b, 0) / 8;
      if (e > QUIET && e > ATTACK * mean) hits.push(i);
      history.copyWithin(0, 1);
      history[7] = e;
    }
    return { here: hits.some((i) => i <= 4), next: hits.some((i) => i >= 3) };
  };
}

export function createAacEncoder(input, { kbps = 64 } = {}) {
  const rate = aacRate(input);
  const index = RATES.indexOf(rate);
  const config = { index, hearing: (fraction) => hearing(fraction * rate) };
  const state = {
    config,
    input: new Float64Array(2 * FRAME),
    block: new Float64Array(2 * SHORT_FRAME),
    raw: new Float64Array(FRAME),
    spec: new Float64Array(FRAME),
    magnitude: new Float64Array(FRAME),
    sign: new Uint8Array(FRAME),
    quant: new Int32Array(FRAME),
    energy: new Float64Array(128),
    floors: new Int32Array(128),
    factors: new Int32Array(128),
    peaks: new Int32Array(128),
    zero: new Uint8Array(128),
    sequence: ONLY_LONG,
  };
  const average = (kbps * 1000 * FRAME) / rate;
  const writer = new Writer(8192);
  const resample = createResampler(input, rate);
  const detect = detector();
  const window = new Float64Array(2 * FRAME);
  let ahead = null;
  let shortNow = false;
  let shortNext = false;
  let wasShort = false;
  let reservoir = 0;
  let pending = new Float32Array(0);
  let total = 0;
  let blocks = 0;
  const encode = () => {
    const short = shortNow || (wasShort && shortNext);
    state.sequence = short
      ? EIGHT_SHORT
      : shortNext
        ? LONG_START
        : wasShort
          ? LONG_STOP
          : ONLY_LONG;
    wasShort = short;
    analyse(state, window);
    const budget = Math.min(MAX_BITS, average + reservoir) - HEADER_BITS;
    let lo = 0;
    let hi = HIGHEST;
    let chosen = plan(state, lo);
    for (let i = 0; i < STEPS; i++) {
      const quality = (lo + hi) / 2;
      const attempt = plan(state, quality);
      writer.reset();
      writeFrame(state, writer, attempt);
      if (writer.pos <= budget) {
        lo = quality;
        chosen = attempt;
      } else hi = quality;
    }
    plan(state, lo);
    writer.reset();
    writeFrame(state, writer, chosen);
    const length = (writer.pos >> 3) + 7;
    reservoir = Math.max(0, Math.min(MAX_BITS - average, reservoir + average - length * 8));
    const out = new Uint8Array(length);
    header(out, index, length);
    out.set(writer.bytes.subarray(0, length - 7), 7);
    return out;
  };
  const step = (pcm) => {
    const block = Float64Array.from({ length: FRAME }, (_, n) => pcm[n] * 65536);
    const hit = detect(pcm);
    blocks++;
    if (!ahead) {
      ahead = block;
      shortNow = hit.here;
      shortNext = hit.next;
      return null;
    }
    const later = shortNext;
    shortNext = hit.here || later;
    window.copyWithin(0, FRAME);
    window.set(ahead, FRAME);
    const out = encode();
    ahead = block;
    shortNow = shortNext;
    shortNext = hit.next;
    return out;
  };
  const drain = (samples) => {
    const joined = new Float32Array(pending.length + samples.length);
    joined.set(pending);
    joined.set(samples, pending.length);
    const out = [];
    let at = 0;
    for (; at + FRAME <= joined.length; at += FRAME) {
      const frame = step(joined.subarray(at, at + FRAME));
      if (frame) out.push(frame);
    }
    pending = joined.slice(at);
    const bytes = new Uint8Array(out.reduce((n, f) => n + f.length, 0));
    let k = 0;
    for (const f of out) {
      bytes.set(f, k);
      k += f.length;
    }
    return bytes;
  };
  return {
    rate,
    push(samples) {
      const resampled = resample(samples);
      total += resampled.length;
      return drain(resampled);
    },
    end() {
      const tail = resample.end();
      total += tail.length;
      const need = (Math.ceil(total / FRAME) + 2 - blocks) * FRAME - pending.length;
      const padded = new Float32Array(need);
      padded.set(tail);
      return drain(padded);
    },
  };
}

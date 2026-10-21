import { imdct } from './mdct.js';
import { AudioFormatError } from './pcm.js';

const FAST = 10;
const RANGES = [256, 128, 86, 64];
const DB_STEP = Math.log(1 / 1.0649863e-7) / 255;
const INVERSE_DB = Float32Array.from(
  { length: 256 },
  (_, i) => 1.0649863e-7 * Math.exp(i * DB_STEP)
);
const MAX_VALUES = 1 << 22;

function corrupt() {
  return new AudioFormatError('the Vorbis stream is corrupt');
}

const ilog = (x) => (x > 0 ? 32 - Math.clz32(x) : 0);

class Bits {
  constructor(bytes) {
    this.bytes = bytes;
    this.pos = 0;
    this.limit = bytes.length * 8;
    this.eop = false;
  }

  peek(n) {
    const b = this.bytes;
    const at = this.pos >> 3;
    const word =
      (b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | (b[at + 3] << 24)) >>> (this.pos & 7);
    return word & ((1 << n) - 1);
  }

  advance(n) {
    this.pos += n;
    if (this.pos > this.limit) this.eop = true;
  }

  read(n) {
    if (n > 24) return this.read(16) + this.read(n - 16) * 65536;
    const v = this.peek(n);
    this.advance(n);
    return this.eop ? 0 : v;
  }
}

function header(bits) {
  if (bits.read(24) !== 0x564342) throw corrupt();
  return { dimensions: bits.read(16), entries: bits.read(24) };
}

function unpackFloat(x) {
  const mantissa = x & 0x1fffff;
  const exponent = (x >>> 21) & 0x3ff;
  return (x >= 0x80000000 ? -mantissa : mantissa) * 2 ** (exponent - 788);
}

function lookup1(entries, dimensions) {
  let r = Math.floor(entries ** (1 / dimensions));
  while ((r + 1) ** dimensions <= entries) r++;
  while (r > 0 && r ** dimensions > entries) r--;
  return r;
}

function assign(lengths) {
  const available = new Float64Array(33);
  const codes = new Float64Array(lengths.length);
  let first = true;
  lengths.forEach((length, entry) => {
    if (!length) return;
    if (first) {
      for (let i = 1; i <= length; i++) available[i] = 2 ** (32 - i);
      first = false;
      return;
    }
    let z = length;
    while (z > 0 && !available[z]) z--;
    if (z === 0) throw corrupt();
    const code = available[z];
    available[z] = 0;
    for (let i = length; i > z; i--) available[i] = code + 2 ** (32 - i);
    codes[entry] = code;
  });
  return codes;
}

function insert(tree, word, length, entry) {
  let node = 0;
  let reversed = 0;
  for (let i = 0; i < length; i++) {
    const bit = Math.floor(word / 2 ** (length - 1 - i)) & 1;
    reversed |= bit << i;
    const slot = node * 2 + bit;
    if (i === length - 1) tree[slot] = -(entry + 1);
    else {
      if (!(tree[slot] > 0)) {
        tree[slot] = tree.length / 2;
        tree.push(0, 0);
      }
      node = tree[slot];
    }
  }
  return reversed;
}

function build(lengths) {
  const fast = new Int32Array(1 << FAST).fill(-1);
  const tree = [0, 0];
  const codes = assign(lengths);
  lengths.forEach((length, entry) => {
    if (!length) return;
    const reversed = insert(tree, codes[entry] / 2 ** (32 - length), length, entry);
    if (length <= FAST)
      for (let fill = reversed; fill < 1 << FAST; fill += 1 << length) fast[fill] = entry;
  });
  return { fast, tree };
}

function readLengths(bits, entries) {
  const lengths = new Uint8Array(entries);
  if (bits.read(1)) {
    let length = bits.read(5) + 1;
    for (let entry = 0; entry < entries; length++) {
      const count = bits.read(ilog(entries - entry));
      if (entry + count > entries || length > 32) throw corrupt();
      lengths.fill(length, entry, entry + count);
      entry += count;
    }
    return lengths;
  }
  const sparse = bits.read(1);
  for (let i = 0; i < entries; i++) if (!sparse || bits.read(1)) lengths[i] = bits.read(5) + 1;
  return lengths;
}

function vectors(bits, type, entries, dimensions) {
  const min = unpackFloat(bits.read(32));
  const delta = unpackFloat(bits.read(32));
  const width = bits.read(4) + 1;
  const sequence = bits.read(1);
  const count = type === 1 ? lookup1(entries, dimensions) : entries * dimensions;
  if (count > MAX_VALUES || entries * dimensions > MAX_VALUES) throw corrupt();
  const values = Float64Array.from({ length: count }, () => bits.read(width) * delta + min);
  const out = new Float32Array(entries * dimensions);
  for (let e = 0; e < entries; e++) {
    let last = 0;
    let divisor = 1;
    for (let d = 0; d < dimensions; d++) {
      const v =
        last + (type === 1 ? values[Math.floor(e / divisor) % count] : values[e * dimensions + d]);
      out[e * dimensions + d] = v;
      if (sequence) last = v;
      divisor *= count;
    }
  }
  return out;
}

function codebook(bits) {
  const { dimensions, entries } = header(bits);
  const lengths = readLengths(bits, entries);
  const type = bits.read(4);
  if (type > 2) throw corrupt();
  const table = type ? vectors(bits, type, entries, dimensions) : null;
  return { dimensions, entries, lengths, vectors: table, ...build(lengths) };
}

function decode(bits, book) {
  const entry = book.fast[bits.peek(FAST)];
  if (entry >= 0) {
    bits.advance(book.lengths[entry]);
    return bits.eop ? -1 : entry;
  }
  let node = 0;
  for (;;) {
    const next = book.tree[node * 2 + bits.read(1)];
    if (bits.eop || !next) return -1;
    if (next < 0) return -next - 1;
    node = next;
  }
}

function floor1(bits, books) {
  const classes = Array.from({ length: bits.read(5) }, () => bits.read(4));
  const kinds = [];
  for (let c = 0; c <= Math.max(-1, ...classes); c++) {
    const kind = { dimensions: bits.read(3) + 1, sub: bits.read(2), master: 0, books: [] };
    if (kind.sub) kind.master = bits.read(8);
    for (let j = 0; j < 1 << kind.sub; j++) kind.books.push(bits.read(8) - 1);
    kinds.push(kind);
  }
  const multiplier = bits.read(2) + 1;
  const width = bits.read(4);
  const xs = [0, 1 << width];
  for (const c of classes) for (let j = 0; j < kinds[c].dimensions; j++) xs.push(bits.read(width));
  if (xs.length > 65 || new Set(xs).size !== xs.length) throw corrupt();
  for (const kind of kinds)
    if (kind.master >= books.length || kind.books.some((b) => b >= books.length)) throw corrupt();
  const low = [];
  const high = [];
  for (let i = 2; i < xs.length; i++) {
    let lo = 0;
    let hi = 1;
    for (let j = 0; j < i; j++) {
      if (xs[j] < xs[i] && xs[j] > xs[lo]) lo = j;
      if (xs[j] > xs[i] && xs[j] < xs[hi]) hi = j;
    }
    low[i] = lo;
    high[i] = hi;
  }
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  return { classes, kinds, multiplier, xs, low, high, order };
}

function residue(bits, books, type) {
  const r = {
    type,
    begin: bits.read(24),
    end: bits.read(24),
    size: bits.read(24) + 1,
    classifications: bits.read(6) + 1,
    classbook: bits.read(8),
    books: [],
  };
  const cascade = Array.from({ length: r.classifications }, () => {
    const low = bits.read(3);
    return (bits.read(1) ? bits.read(5) : 0) * 8 + low;
  });
  for (const c of cascade) {
    const set = [];
    for (let j = 0; j < 8; j++) set.push(c & (1 << j) ? bits.read(8) : -1);
    if (set.some((b) => b >= books.length || (b >= 0 && !books[b].vectors))) throw corrupt();
    r.books.push(set);
  }
  if (r.classbook >= books.length) throw corrupt();
  return r;
}

function mapping(bits, channels, floors, residues) {
  const submaps = bits.read(1) ? bits.read(4) + 1 : 1;
  const coupling = [];
  if (bits.read(1)) {
    const steps = bits.read(8) + 1;
    const width = ilog(channels - 1);
    for (let i = 0; i < steps; i++) {
      const magnitude = bits.read(width);
      const angle = bits.read(width);
      if (magnitude === angle || magnitude >= channels || angle >= channels) throw corrupt();
      coupling.push([magnitude, angle]);
    }
  }
  if (bits.read(2)) throw corrupt();
  const mux = new Array(channels).fill(0);
  if (submaps > 1) for (let c = 0; c < channels; c++) mux[c] = bits.read(4);
  const maps = Array.from({ length: submaps }, () => {
    bits.read(8);
    return { floor: bits.read(8), residue: bits.read(8) };
  });
  if (mux.some((m) => m >= submaps)) throw corrupt();
  if (maps.some((m) => m.floor >= floors.length || m.residue >= residues.length)) throw corrupt();
  return { coupling, mux, maps };
}

export function vorbisHeaders(id, setup) {
  const text = (p) => String.fromCharCode(...p.subarray(1, 7));
  if (id?.[0] !== 1 || text(id) !== 'vorbis') throw corrupt();
  if (setup?.[0] !== 5 || text(setup) !== 'vorbis') throw corrupt();
  const view = new DataView(id.buffer, id.byteOffset, id.byteLength);
  if (id.length < 30 || view.getUint32(7, true) !== 0) throw corrupt();
  const channels = id[11];
  const rate = view.getUint32(12, true);
  const blocks = [1 << (id[28] & 15), 1 << (id[28] >> 4)];
  if (!channels || !rate || blocks[0] < 64 || blocks[1] > 8192 || blocks[0] > blocks[1])
    throw corrupt();
  const bits = new Bits(setup.subarray(7));
  const books = Array.from({ length: bits.read(8) + 1 }, () => codebook(bits));
  const times = bits.read(6) + 1;
  for (let i = 0; i < times; i++) if (bits.read(16)) throw corrupt();
  const floors = Array.from({ length: bits.read(6) + 1 }, () => {
    const type = bits.read(16);
    if (type === 0) throw new AudioFormatError('Vorbis floor 0 is not supported');
    if (type !== 1) throw corrupt();
    return floor1(bits, books);
  });
  const residues = Array.from({ length: bits.read(6) + 1 }, () => {
    const type = bits.read(16);
    if (type > 2) throw corrupt();
    return residue(bits, books, type);
  });
  const mappings = Array.from({ length: bits.read(6) + 1 }, () => {
    if (bits.read(16)) throw corrupt();
    return mapping(bits, channels, floors, residues);
  });
  const modes = Array.from({ length: bits.read(6) + 1 }, () => {
    const mode = { block: bits.read(1), window: bits.read(16), transform: bits.read(16) };
    mode.mapping = bits.read(8);
    if (mode.window || mode.transform || mode.mapping >= mappings.length) throw corrupt();
    return mode;
  });
  if (!bits.read(1) || bits.eop) throw corrupt();
  return { channels, rate, blocks, books, floors, residues, mappings, modes };
}

function floorValues(bits, floor, books) {
  if (!bits.read(1)) return null;
  const width = ilog(RANGES[floor.multiplier - 1] - 1);
  const ys = [bits.read(width), bits.read(width)];
  for (const c of floor.classes) {
    const kind = floor.kinds[c];
    const mask = (1 << kind.sub) - 1;
    let value = kind.sub ? decode(bits, books[kind.master]) : 0;
    if (value < 0) return null;
    for (let j = 0; j < kind.dimensions; j++) {
      const book = kind.books[value & mask];
      value >>= kind.sub;
      const y = book >= 0 ? decode(bits, books[book]) : 0;
      if (y < 0) return null;
      ys.push(y);
    }
  }
  return bits.eop ? null : ys;
}

function line(x0, y0, x1, y1, out, limit) {
  const dy = y1 - y0;
  const adx = x1 - x0;
  const base = Math.trunc(dy / adx);
  const step = dy < 0 ? base - 1 : base + 1;
  const ady = Math.abs(dy) - Math.abs(base) * adx;
  let y = y0;
  let err = 0;
  if (x0 < limit) out[x0] *= INVERSE_DB[y];
  for (let x = x0 + 1; x < Math.min(x1, limit); x++) {
    err += ady;
    if (err >= adx) {
      err -= adx;
      y += step;
    } else y += base;
    out[x] *= INVERSE_DB[y];
  }
}

function floorCurve(floor, ys, out, half) {
  const range = RANGES[floor.multiplier - 1];
  const { xs, low, high } = floor;
  const final = ys.slice();
  const used = ys.map(() => false);
  used[0] = true;
  used[1] = true;
  for (let i = 2; i < xs.length; i++) {
    const lo = low[i];
    const hi = high[i];
    const dy = final[hi] - final[lo];
    const off = Math.floor((Math.abs(dy) * (xs[i] - xs[lo])) / (xs[hi] - xs[lo]));
    const predicted = dy < 0 ? final[lo] - off : final[lo] + off;
    const value = ys[i];
    const highroom = range - predicted;
    const lowroom = predicted;
    const room = (highroom < lowroom ? highroom : lowroom) * 2;
    if (value) {
      used[lo] = true;
      used[hi] = true;
      used[i] = true;
      if (value >= room)
        final[i] =
          highroom > lowroom ? value - lowroom + predicted : predicted - value + highroom - 1;
      else final[i] = value & 1 ? predicted - (value + 1) / 2 : predicted + value / 2;
    } else final[i] = predicted;
  }
  const clamp = (y) => Math.max(0, Math.min(255, y * floor.multiplier));
  let lx = 0;
  let ly = clamp(final[0]);
  for (const i of floor.order.slice(1)) {
    if (!used[i]) continue;
    const hy = clamp(final[i]);
    line(lx, ly, xs[i], hy, out, half);
    lx = xs[i];
    ly = hy;
  }
  if (lx < half) line(lx, ly, half, ly, out, half);
}

function vqDecode(bits, r, vq, target, offset) {
  const dims = vq.dimensions;
  if (r.type === 0) {
    const step = r.size / dims;
    for (let j = 0; j < step; j++) {
      const e = decode(bits, vq);
      if (e < 0) return false;
      for (let k = 0; k < dims; k++) target[offset + j + k * step] += vq.vectors[e * dims + k];
    }
    return true;
  }
  for (let i = 0; i < r.size; ) {
    const e = decode(bits, vq);
    if (e < 0) return false;
    for (let k = 0; k < dims && i < r.size; k++) target[offset + i++] += vq.vectors[e * dims + k];
  }
  return true;
}

function classify(bits, r, book, classes, p) {
  let temp = decode(bits, book);
  if (temp < 0) return false;
  for (let i = book.dimensions - 1; i >= 0; i--) {
    classes[p + i] = temp % r.classifications;
    temp = Math.floor(temp / r.classifications);
  }
  return true;
}

function partitions(bits, r, books, job) {
  const { targets, mute, begin, parts } = job;
  const book = books[r.classbook];
  const classes = targets.map(() => new Int32Array(parts + book.dimensions));
  const live = targets.map((_, c) => c).filter((c) => !mute[c]);
  for (let pass = 0; pass < 8; pass++) {
    for (let p = 0; p < parts; ) {
      if (pass === 0 && !live.every((c) => classify(bits, r, book, classes[c], p))) return;
      for (let i = 0; i < book.dimensions && p < parts; i++, p++) {
        for (const c of live) {
          const vq = r.books[classes[c][p]][pass];
          if (vq >= 0 && !vqDecode(bits, r, books[vq], targets[c], begin + p * r.size)) return;
        }
      }
    }
  }
}

function residueDecode(bits, r, books, vectors, skip, half) {
  const interleave = r.type === 2;
  const size = interleave ? half * vectors.length : half;
  const targets = interleave ? [new Float32Array(size)] : vectors;
  const mute = interleave ? [skip.every(Boolean)] : skip;
  const begin = Math.min(r.begin, size);
  const parts = Math.floor((Math.min(r.end, size) - begin) / r.size);
  partitions(bits, r, books, { targets, mute, begin, parts });
  if (!interleave) return;
  const [all] = targets;
  vectors.forEach((v, c) => {
    for (let i = 0; i < half; i++) v[i] = all[i * vectors.length + c];
  });
}

function slope(x, n) {
  return Math.sin((Math.PI / 2) * Math.sin(((x + 0.5) / n) * (Math.PI / 2)) ** 2);
}

function windowOf(cache, blocks, n, long, prev, next) {
  const key = `${n}:${prev}:${next}`;
  if (cache.has(key)) return cache.get(key);
  const w = new Float64Array(n);
  const short = blocks[0];
  const leftN = long && !prev ? short / 2 : n / 2;
  const leftStart = long && !prev ? n / 4 - short / 4 : 0;
  const rightN = long && !next ? short / 2 : n / 2;
  const rightStart = long && !next ? (n * 3) / 4 - short / 4 : n / 2;
  for (let i = 0; i < n; i++) {
    if (i < leftStart) w[i] = 0;
    else if (i < leftStart + leftN) w[i] = slope(i - leftStart, leftN);
    else if (i < rightStart) w[i] = 1;
    else if (i < rightStart + rightN) w[i] = slope(rightN - 1 - (i - rightStart), rightN);
  }
  cache.set(key, w);
  return w;
}

function uncouple(coupling, vectors, half) {
  for (let i = coupling.length - 1; i >= 0; i--) {
    const mag = vectors[coupling[i][0]];
    const ang = vectors[coupling[i][1]];
    for (let j = 0; j < half; j++) {
      const m = mag[j];
      const a = ang[j];
      if (a > 0) ang[j] = m > 0 ? m - a : m + a;
      else {
        ang[j] = m;
        mag[j] = m > 0 ? m + a : m - a;
      }
    }
  }
}

function spectrum(bits, setup, map, half) {
  const { channels, books, floors, residues } = setup;
  const floorOf = (c) => floors[map.maps[map.mux[c]].floor];
  const ys = Array.from({ length: channels }, (_, c) => floorValues(bits, floorOf(c), books));
  const nonzero = ys.map((y) => y !== null);
  for (const [m, a] of map.coupling) {
    nonzero[m] ||= nonzero[a];
    nonzero[a] = nonzero[m];
  }
  const vectors = Array.from({ length: channels }, () => new Float32Array(half));
  map.maps.forEach((sub, s) => {
    const members = vectors.map((_, c) => c).filter((c) => map.mux[c] === s);
    const targets = members.map((c) => vectors[c]);
    const skip = members.map((c) => !nonzero[c]);
    residueDecode(bits, residues[sub.residue], books, targets, skip, half);
  });
  uncouple(map.coupling, vectors, half);
  const sum = new Float64Array(half);
  for (let c = 0; c < channels; c++) {
    if (!ys[c]) continue;
    floorCurve(floorOf(c), ys[c], vectors[c], half);
    for (let i = 0; i < half; i++) sum[i] += vectors[c][i] / channels;
  }
  return sum;
}

function overlap(last, block) {
  const pn = last.length;
  const n = block.length;
  const out = new Float32Array(pn / 4 + n / 4);
  for (let i = 0; i < out.length; i++) {
    const j = i - pn / 4 + n / 4;
    out[i] = (pn / 2 + i < pn ? last[pn / 2 + i] : 0) + (j >= 0 && j < n ? block[j] : 0);
  }
  return out;
}

export function createVorbisDecoder(setup) {
  const { blocks, mappings, modes } = setup;
  const windows = new Map();
  const modeBits = ilog(modes.length - 1);
  let previous = null;
  return (packet) => {
    const bits = new Bits(packet);
    const type = bits.read(1);
    const mode = modes[bits.read(modeBits)];
    if (type || !mode || bits.eop) return new Float32Array(0);
    const long = mode.block;
    const n = blocks[long];
    const prev = long ? bits.read(1) : 0;
    const next = long ? bits.read(1) : 0;
    const sum = spectrum(bits, setup, mappings[mode.mapping], n / 2);
    const block = imdct(sum, new Float64Array(n), n);
    const w = windowOf(windows, blocks, n, long, prev, next);
    for (let i = 0; i < n; i++) block[i] *= w[i];
    const last = previous;
    previous = block;
    return last ? overlap(last, block) : new Float32Array(0);
  };
}

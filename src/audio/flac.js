import { AudioFormatError, concat, encodePcm16 } from './pcm.js';

const RATES = [0, 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000];
const SIZES = [0, 8, 12, 0, 16, 20, 24, 32];
const FIXED = [[], [1], [2, -1], [3, -3, 1], [4, -6, 4, -1]];
const BLOCK = 4096;

const crcTable = (poly, bits) =>
  new Uint16Array(256).map((_, i) => {
    let c = i << (bits - 8);
    for (let k = 0; k < 8; k++) c = c & (1 << (bits - 1)) ? (c << 1) ^ poly : c << 1;
    return c & ((1 << bits) - 1);
  });
const CRC8 = crcTable(0x07, 8);
const CRC16 = crcTable(0x8005, 16);

function crc8(bytes) {
  let crc = 0;
  for (const b of bytes) crc = CRC8[crc ^ b];
  return crc;
}

function crc16(bytes) {
  let crc = 0;
  for (const b of bytes) crc = ((crc << 8) ^ CRC16[(crc >> 8) ^ b]) & 0xffff;
  return crc;
}

function corrupt() {
  return new AudioFormatError('the FLAC file is corrupt');
}

export function isFlac(bytes) {
  return (
    bytes.length >= 4 &&
    bytes[0] === 0x66 &&
    bytes[1] === 0x4c &&
    bytes[2] === 0x61 &&
    bytes[3] === 0x43
  );
}

class Reader {
  constructor(bytes, at) {
    this.bytes = bytes;
    this.pos = at * 8;
  }

  byte() {
    const b = this.bytes[this.pos >>> 3];
    if (b === undefined) throw corrupt();
    return b;
  }

  read(n) {
    let value = 0;
    while (n > 0) {
      const free = 8 - (this.pos & 7);
      const take = Math.min(free, n);
      value = value * 2 ** take + ((this.byte() >> (free - take)) & ((1 << take) - 1));
      this.pos += take;
      n -= take;
    }
    return value;
  }

  signed(n) {
    const value = this.read(n);
    return n && value >= 2 ** (n - 1) ? value - 2 ** n : value;
  }

  unary() {
    let count = 0;
    for (;;) {
      const rest = (this.byte() << (this.pos & 7)) & 0xff;
      if (rest) {
        const zeros = Math.clz32(rest) - 24;
        this.pos += zeros + 1;
        return count + zeros;
      }
      const free = 8 - (this.pos & 7);
      count += free;
      this.pos += free;
    }
  }

  rice(k) {
    const u = this.unary() * 2 ** k + this.read(k);
    return u % 2 ? -(u + 1) / 2 : u / 2;
  }

  align() {
    this.pos = Math.ceil(this.pos / 8) * 8;
  }
}

function residual(r, out, order) {
  const method = r.read(2);
  if (method > 1) throw corrupt();
  const bits = method ? 5 : 4;
  const escape = method ? 31 : 15;
  const partitions = 2 ** r.read(4);
  const per = out.length / partitions;
  if (!Number.isInteger(per) || per < order) throw corrupt();
  let i = order;
  for (let p = 0; p < partitions; p++) {
    const end = (p + 1) * per;
    const k = r.read(bits);
    if (k === escape) {
      const width = r.read(5);
      while (i < end) out[i++] = r.signed(width);
    } else while (i < end) out[i++] = r.rice(k);
  }
}

function predict(out, coefs, shift) {
  const order = coefs.length;
  const scale = 2 ** shift;
  for (let i = order; i < out.length; i++) {
    let sum = 0;
    for (let j = 0; j < order; j++) sum += coefs[j] * out[i - 1 - j];
    out[i] += Math.floor(sum / scale);
  }
}

function subframe(r, n, bits) {
  r.read(1);
  const type = r.read(6);
  const wasted = r.read(1) ? r.unary() + 1 : 0;
  bits -= wasted;
  const out = new Float64Array(n);
  if (type === 0) out.fill(r.signed(bits));
  else if (type === 1) for (let i = 0; i < n; i++) out[i] = r.signed(bits);
  else if ((type >= 8 && type <= 12) || type >= 32) {
    const order = type >= 32 ? type - 31 : type - 8;
    if (order > n) throw corrupt();
    for (let i = 0; i < order; i++) out[i] = r.signed(bits);
    let coefs = FIXED[order];
    let shift = 0;
    if (type >= 32) {
      const precision = r.read(4) + 1;
      shift = r.signed(5);
      if (precision === 16 || shift < 0) throw corrupt();
      coefs = Array.from({ length: order }, () => r.signed(precision));
    }
    residual(r, out, order);
    predict(out, coefs, shift);
  } else throw corrupt();
  if (wasted) for (let i = 0; i < n; i++) out[i] *= 2 ** wasted;
  return out;
}

function blockSize(r, code) {
  if (code === 1) return 192;
  if (code >= 2 && code <= 5) return 576 * 2 ** (code - 2);
  if (code === 6) return r.read(8) + 1;
  if (code === 7) return r.read(16) + 1;
  if (code >= 8) return 256 * 2 ** (code - 8);
  throw corrupt();
}

function frameRate(r, code, info) {
  if (code === 0) return info.rate;
  if (code < 12) return RATES[code];
  if (code === 12) return r.read(8) * 1000;
  if (code === 13) return r.read(16);
  if (code === 14) return r.read(16) * 10;
  throw corrupt();
}

function decorrelate(layout, a, b) {
  for (let i = 0; i < a.length; i++) {
    if (layout === 8) b[i] = a[i] - b[i];
    else if (layout === 9) a[i] += b[i];
    else {
      const mid = a[i] * 2 + (b[i] % 2 ? 1 : 0);
      a[i] = (mid + b[i]) / 2;
      b[i] = (mid - b[i]) / 2;
    }
  }
}

function frame(r, info) {
  r.read(16);
  const sizeCode = r.read(4);
  const rateCode = r.read(4);
  const layout = r.read(4);
  const bitsCode = r.read(3);
  r.read(1);
  const lead = r.read(8);
  for (let i = 1; i < Math.clz32(~(lead << 24)); i++) r.read(8);
  const n = blockSize(r, sizeCode);
  const rate = frameRate(r, rateCode, info);
  r.read(8);
  const bits = bitsCode ? SIZES[bitsCode] : info.bits;
  if (!bits || layout > 10) throw corrupt();
  const count = layout < 8 ? layout + 1 : 2;
  const side = layout === 9 ? 0 : layout > 7 ? 1 : -1;
  const channels = Array.from({ length: count }, (_, c) => subframe(r, n, bits + (c === side)));
  r.align();
  r.read(16);
  if (layout > 7) decorrelate(layout, ...channels);
  const scale = count * 2 ** (bits - 1);
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (const ch of channels) sum += ch[i];
    samples[i] = sum / scale;
  }
  return { samples, rate };
}

export function* decodeFlac(bytes) {
  if (!isFlac(bytes)) throw new AudioFormatError('the file is not a FLAC file');
  let info = null;
  let at = 4;
  for (let last = false; !last; ) {
    if (at + 4 > bytes.length) throw corrupt();
    last = Boolean(bytes[at] & 0x80);
    const size = (bytes[at + 1] << 16) | (bytes[at + 2] << 8) | bytes[at + 3];
    if ((bytes[at] & 0x7f) === 0) {
      const r = new Reader(bytes, at + 14);
      info = { rate: r.read(20), bits: (r.read(8) & 0x1f) + 1 };
    }
    at += 4 + size;
  }
  if (!info) throw new AudioFormatError('the FLAC file has no STREAMINFO');
  const r = new Reader(bytes, at);
  for (let i = at; bytes[i] === 0xff && (bytes[i + 1] & 0xfe) === 0xf8; i = r.pos / 8)
    yield frame(r, info);
}

class Writer {
  constructor(size = 1024) {
    this.bytes = new Uint8Array(size);
    this.pos = 0;
  }

  room(bits) {
    const need = Math.ceil((this.pos + bits) / 8) + 1;
    if (need <= this.bytes.length) return;
    const grown = new Uint8Array(Math.max(need, this.bytes.length * 2));
    grown.set(this.bytes);
    this.bytes = grown;
  }

  write(value, n) {
    this.room(n);
    if (value < 0) value += 2 ** n;
    while (n > 0) {
      const free = 8 - (this.pos & 7);
      const take = Math.min(free, n);
      const part = Math.floor(value / 2 ** (n - take)) & ((1 << take) - 1);
      this.bytes[this.pos >>> 3] |= part << (free - take);
      this.pos += take;
      n -= take;
    }
  }

  zeros(n) {
    this.room(n);
    this.pos += n;
  }

  align() {
    this.pos = Math.ceil(this.pos / 8) * 8;
  }

  done() {
    return this.bytes.subarray(0, Math.ceil(this.pos / 8));
  }
}

function utf8(value) {
  if (value < 0x80) return [value];
  let n = 2;
  while (value >= 2 ** (5 * n + 1)) n++;
  const out = [((0xff00 >> n) & 0xff) | Math.floor(value / 2 ** (6 * (n - 1)))];
  for (let i = n - 2; i >= 0; i--) out.push(0x80 | (Math.floor(value / 2 ** (6 * i)) & 0x3f));
  return out;
}

function fixedResidual(x, order) {
  const coefs = FIXED[order];
  const out = new Int32Array(x.length - order);
  for (let i = order; i < x.length; i++) {
    let p = 0;
    for (let j = 0; j < order; j++) p += coefs[j] * x[i - 1 - j];
    out[i - order] = x[i] - p;
  }
  return out;
}

function riceCost(u, from, to) {
  let sum = 0;
  for (let i = from; i < to; i++) sum += u[i];
  const mean = sum / Math.max(1, to - from);
  let best = { k: 0, bits: Infinity };
  const guess = mean > 1 ? Math.floor(Math.log2(mean)) : 0;
  for (let k = Math.max(0, guess - 1); k <= Math.min(14, guess + 1); k++) {
    let bits = (to - from) * (k + 1);
    for (let i = from; i < to; i++) bits += Math.floor(u[i] / 2 ** k);
    if (bits < best.bits) best = { k, bits };
  }
  return best;
}

function partitionPlan(u, n, order) {
  let best = null;
  for (let p = 0; p <= 8 && n % 2 ** p === 0 && n / 2 ** p > order; p++) {
    const per = n / 2 ** p;
    const ks = [];
    let bits = 0;
    for (let q = 0; q < 2 ** p; q++) {
      const cost = riceCost(u, Math.max(0, q * per - order), (q + 1) * per - order);
      ks.push(cost.k);
      bits += cost.bits + 4;
    }
    if (!best || bits < best.bits) best = { p, ks, bits };
  }
  return best;
}

function writeSubframe(w, x) {
  const n = x.length;
  if (x.every((v) => v === x[0])) {
    w.write(0, 8);
    w.write(x[0], 16);
    return;
  }
  let order = 0;
  let res = null;
  let least = Infinity;
  for (let o = 0; o <= Math.min(4, n - 1); o++) {
    const r = fixedResidual(x, o);
    let sum = 0;
    for (const v of r) sum += Math.abs(v);
    if (sum < least) [least, order, res] = [sum, o, r];
  }
  const u = res.map((v) => (v < 0 ? -2 * v - 1 : 2 * v));
  const plan = partitionPlan(u, n, order);
  if (plan.bits + order * 16 >= n * 16) {
    w.write(2, 8);
    for (const v of x) w.write(v, 16);
    return;
  }
  w.write(16 + order * 2, 8);
  for (let i = 0; i < order; i++) w.write(x[i], 16);
  w.write(0, 2);
  w.write(plan.p, 4);
  const per = n / 2 ** plan.p;
  plan.ks.forEach((k, q) => {
    w.write(k, 4);
    for (let i = Math.max(0, q * per - order); i < (q + 1) * per - order; i++) {
      w.zeros(Math.floor(u[i] / 2 ** k));
      w.write(1, 1);
      if (k) w.write(u[i] % 2 ** k, k);
    }
  });
}

function encodeFrame(x, number) {
  const w = new Writer(x.length * 2 + 64);
  const n = x.length;
  const power = Math.log2(n / 256);
  const code = Number.isInteger(power) && power >= 0 && power <= 7 ? 8 + power : n <= 256 ? 6 : 7;
  w.write(0xfff8, 16);
  w.write(code, 4);
  w.write(0, 4);
  w.write(0, 4);
  w.write(4, 3);
  w.write(0, 1);
  for (const b of utf8(number)) w.write(b, 8);
  if (code === 6) w.write(n - 1, 8);
  if (code === 7) w.write(n - 1, 16);
  w.write(crc8(w.done()), 8);
  writeSubframe(w, x);
  w.align();
  w.write(crc16(w.done()), 16);
  return w.done();
}

function streamInfo(rate, block) {
  const w = new Writer(42);
  for (const c of 'fLaC') w.write(c.charCodeAt(0), 8);
  w.write(0x80, 8);
  w.write(34, 24);
  w.write(block, 16);
  w.write(block, 16);
  w.write(0, 48);
  w.write(rate, 20);
  w.write(0, 3);
  w.write(15, 5);
  w.zeros(36 + 128);
  return w.done();
}

export function createFlacEncoder(rate, { block = BLOCK } = {}) {
  let pending = new Int32Array(0);
  let number = 0;
  let header = streamInfo(rate, block);
  const flush = (all) => {
    const parts = header ? [header] : [];
    header = null;
    let at = 0;
    for (; at + block <= pending.length || (all && at < pending.length); at += block)
      parts.push(encodeFrame(pending.subarray(at, Math.min(at + block, pending.length)), number++));
    pending = pending.slice(at);
    return concat(parts);
  };
  return {
    push(samples) {
      pending = concat([pending, new Int16Array(encodePcm16(samples).buffer)], Int32Array);
      return flush(false);
    },
    end() {
      return flush(true);
    },
  };
}

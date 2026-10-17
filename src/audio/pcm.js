export class AudioFormatError extends Error {
  constructor(message) {
    super(message);
    this.name = 'AudioFormatError';
  }
}

const tag = (view, at) =>
  String.fromCharCode(
    view.getUint8(at),
    view.getUint8(at + 1),
    view.getUint8(at + 2),
    view.getUint8(at + 3)
  );

export function isWav(bytes) {
  if (bytes.length < 12) return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return tag(view, 0) === 'RIFF' && tag(view, 8) === 'WAVE';
}

function sampleReader(view, format, bits) {
  if (format === 3 && bits === 32) return (at) => view.getFloat32(at, true);
  if (format === 3 && bits === 64) return (at) => view.getFloat64(at, true);
  if (format !== 1) return null;
  if (bits === 8) return (at) => (view.getUint8(at) - 128) / 128;
  if (bits === 16) return (at) => view.getInt16(at, true) / 32768;
  if (bits === 24)
    return (at) => ((view.getInt8(at + 2) << 16) | view.getUint16(at, true)) / 8388608;
  if (bits === 32) return (at) => view.getInt32(at, true) / 2147483648;
  return null;
}

export function wavHeader(rate, dataBytes = 0xffffffff, channels = 1) {
  const out = new Uint8Array(44);
  const view = new DataView(out.buffer);
  const put = (at, text) => {
    for (let i = 0; i < 4; i++) out[at + i] = text.charCodeAt(i);
  };
  put(0, 'RIFF');
  view.setUint32(4, Math.min(0xffffffff, dataBytes + 36), true);
  put(8, 'WAVE');
  put(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  put(36, 'data');
  view.setUint32(40, dataBytes, true);
  return out;
}

export function decodeWav(bytes) {
  if (!isWav(bytes)) throw new AudioFormatError('the file is not a WAV file');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let fmt = null;
  let data = null;
  for (let at = 12; at + 8 <= bytes.length; ) {
    const id = tag(view, at);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (id === 'fmt ' && size >= 16) {
      let format = view.getUint16(body, true);
      if (format === 0xfffe && size >= 26) format = view.getUint16(body + 24, true);
      fmt = {
        format,
        channels: view.getUint16(body + 2, true),
        rate: view.getUint32(body + 4, true),
        bits: view.getUint16(body + 14, true),
      };
    } else if (id === 'data') {
      data = { at: body, size: Math.min(size, bytes.length - body) };
    }
    at = body + size + (size % 2);
  }
  if (!fmt || !data) throw new AudioFormatError('the WAV file has no fmt or data chunk');
  const read = sampleReader(view, fmt.format, fmt.bits);
  if (!read || fmt.channels < 1 || fmt.rate < 1)
    throw new AudioFormatError(
      `WAV encoding ${fmt.format} with ${fmt.bits} bits is not supported; send PCM or float WAV`
    );
  const width = fmt.bits / 8;
  const frame = width * fmt.channels;
  const count = Math.floor(data.size / frame);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    for (let c = 0; c < fmt.channels; c++) sum += read(data.at + i * frame + c * width);
    samples[i] = sum / fmt.channels;
  }
  return { samples, rate: fmt.rate };
}

export function encodePcm16(samples) {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] || 0));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out;
}

export function decodePcm16(bytes, rate = 24000, channels = 1) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = Math.floor(bytes.length / (2 * channels));
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += view.getInt16((i * channels + c) * 2, true);
    samples[i] = sum / channels / 32768;
  }
  return { samples, rate };
}

function sinc(x) {
  if (x === 0) return 1;
  const y = Math.PI * x;
  return Math.sin(y) / y;
}

function filterOf(from, to, zeros) {
  const ratio = to / from;
  const cutoff = Math.min(1, ratio) * 0.97;
  const half = Math.ceil(zeros / cutoff);
  return {
    ratio,
    half,
    at(get, j) {
      const t = j / ratio;
      const center = Math.floor(t);
      let sum = 0;
      let weight = 0;
      for (let k = center - half + 1; k <= center + half; k++) {
        const d = t - k;
        const h = cutoff * sinc(cutoff * d) * (0.5 + 0.5 * Math.cos((Math.PI * d) / half));
        weight += h;
        sum += get(k) * h;
      }
      return sum / weight;
    },
  };
}

export function resample(samples, from, to, zeros = 16) {
  if (from === to) return samples;
  const filter = filterOf(from, to, zeros);
  const get = (k) => (k >= 0 && k < samples.length ? samples[k] : 0);
  const out = new Float32Array(Math.floor((samples.length * to) / from));
  for (let j = 0; j < out.length; j++) out[j] = filter.at(get, j);
  return out;
}

export function concat(parts, Type = Uint8Array) {
  const out = new Type(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

export function createResampler(from, to, zeros = 16) {
  if (from === to) return Object.assign((samples) => samples, { end: () => new Float32Array(0) });
  const filter = filterOf(from, to, zeros);
  let buffer = new Float32Array(0);
  let base = 0;
  let next = 0;
  let taken = 0;
  const get = (k) => (k >= base && k < base + buffer.length ? buffer[k - base] : 0);
  const push = (samples) => {
    taken += samples.length;
    const joined = new Float32Array(buffer.length + samples.length);
    joined.set(buffer);
    joined.set(samples, buffer.length);
    buffer = joined;
    const end = base + buffer.length;
    const out = [];
    while (next / filter.ratio + filter.half < end) out.push(filter.at(get, next++));
    const keep = Math.max(base, Math.floor(next / filter.ratio) - filter.half);
    buffer = buffer.slice(keep - base);
    base = keep;
    return Float32Array.from(out);
  };
  push.end = () => {
    const want = Math.floor((taken * to) / from) - next;
    return push(new Float32Array(filter.half + 1)).subarray(0, Math.max(0, want));
  };
  return push;
}

export function audioParams(contentType = '') {
  const [type, ...params] = contentType.split(';').map((s) => s.trim());
  const values = {};
  for (const p of params) {
    const eq = p.indexOf('=');
    if (eq > 0)
      values[p.slice(0, eq).trim().toLowerCase()] = p
        .slice(eq + 1)
        .trim()
        .replace(/^"|"$/g, '');
  }
  return { type: type.toLowerCase(), params: values };
}

const WAV_TYPES = new Set(['audio/wav', 'audio/wave', 'audio/x-wav', 'audio/vnd.wave']);
export const PCM_TYPES = new Set(['audio/pcm', 'audio/l16', 'audio/raw']);

export function decodeAudio(bytes, contentType = '') {
  const { type, params } = audioParams(contentType);
  if (isWav(bytes)) return decodeWav(bytes);
  if (WAV_TYPES.has(type)) throw new AudioFormatError('the file is not a valid WAV file');
  if (PCM_TYPES.has(type)) {
    const rate = params.rate === undefined ? 24000 : Number(params.rate);
    const channels = params.channels === undefined ? 1 : Number(params.channels);
    if (!Number.isInteger(rate) || rate < 1000 || rate > 384000)
      throw new AudioFormatError(`audio rate ${params.rate} is not supported`);
    if (!Number.isInteger(channels) || channels < 1 || channels > 8)
      throw new AudioFormatError(`audio channels ${params.channels} is not supported`);
    return decodePcm16(bytes, rate, channels);
  }
  throw new AudioFormatError(
    `audio type ${type || 'unknown'} is not supported yet; send WAV or 16-bit PCM (audio/pcm;rate=N)`
  );
}

export function toMono16k(bytes, contentType) {
  const { samples, rate } = decodeAudio(bytes, contentType);
  return resample(samples, rate, 16000);
}

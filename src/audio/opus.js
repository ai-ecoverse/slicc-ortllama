import { oggPage } from './ogg.js';
import { AudioFormatError, concat, createResampler } from './pcm.js';

const RATES = [8000, 12000, 16000, 24000, 48000];
const FRAME_MS = 20;
const BITRATE = 32000;
const VENDOR = 'slicc-ortllama';

const library = () => import('libopus-wasm');

const text = (bytes, at, length) => String.fromCharCode(...bytes.subarray(at, at + length));

export function opusHead(bytes) {
  if (!bytes || bytes.length < 19 || text(bytes, 0, 8) !== 'OpusHead')
    throw new AudioFormatError('the Opus stream has no OpusHead');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const head = { channels: bytes[9], skip: view.getUint16(10, true), family: bytes[18] };
  if (head.family !== 0 || head.channels < 1 || head.channels > 2)
    throw new AudioFormatError('multichannel Opus is not supported; send mono or stereo');
  return head;
}

const FRAMES = [480, 960, 1920, 2880];

export function packetSamples(packet) {
  if (packet.length === 0) return 0;
  const config = packet[0] >> 3;
  const size =
    config < 12 ? FRAMES[config % 4] : config < 16 ? FRAMES[config % 2] : 120 * 2 ** (config % 4);
  const code = packet[0] & 3;
  const count = code === 0 ? 1 : code < 3 ? 2 : (packet[1] ?? 0) & 0x3f;
  return size * count;
}

export function opusRate(rate) {
  return RATES.find((r) => r >= rate) ?? 48000;
}

export async function* decodeOpus({ head, packets, granule = -1, rate = 48000 }) {
  const { skip } = opusHead(head);
  const { createDecoder } = await library();
  const decoder = await createDecoder({ sampleRate: rate, channels: 1 });
  const scale = rate / 48000;
  let drop = Math.round(skip * scale);
  let left = granule >= 0 ? Math.max(0, Math.round((granule - skip) * scale)) : Infinity;
  try {
    for (const packet of packets) {
      if (packet.length === 0) continue;
      let samples;
      try {
        samples = decoder.decodeFloat(packet);
      } catch (error) {
        throw new AudioFormatError(`the Opus stream is corrupt: ${error.message}`);
      }
      const cut = Math.min(drop, samples.length);
      drop -= cut;
      samples = samples.subarray(cut, cut + Math.min(left, samples.length - cut));
      left -= samples.length;
      if (samples.length) yield { samples, rate };
    }
  } finally {
    decoder.free();
  }
}

function headers(rate, skip) {
  const head = new Uint8Array(19);
  const view = new DataView(head.buffer);
  head.set([...'OpusHead'].map((c) => c.charCodeAt(0)));
  head[8] = 1;
  head[9] = 1;
  view.setUint16(10, skip, true);
  view.setUint32(12, rate, true);
  const vendor = new TextEncoder().encode(VENDOR);
  const tags = new Uint8Array(16 + vendor.length);
  const tagView = new DataView(tags.buffer);
  tags.set([...'OpusTags'].map((c) => c.charCodeAt(0)));
  tagView.setUint32(8, vendor.length, true);
  tags.set(vendor, 12);
  return [head, tags];
}

export async function opusEncoder() {
  const { createEncoder } = await library();
  return async (input) => {
    const rate = opusRate(input);
    const encoder = await createEncoder({ sampleRate: rate, channels: 1, bitrate: BITRATE });
    const resampler = createResampler(input, rate);
    const frame = (rate * FRAME_MS) / 1000;
    const scale = 48000 / rate;
    const lookahead = encoder.getLookahead();
    const skip = lookahead * scale;
    const serial = (Math.random() * 0x100000000) >>> 0;
    let sequence = 0;
    let pending = new Float32Array(0);
    let encoded = 0;
    let total = 0;
    const page = (packets, granule, flags = 0) =>
      oggPage({ serial, sequence: sequence++, granule, flags, packets });
    const pages = (packets, granule, last) => {
      const out = [];
      for (let i = 0; i < packets.length; i += 255) {
        const end = i + 255 >= packets.length;
        out.push(page(packets.slice(i, i + 255), end ? granule : -1, end && last ? 4 : 0));
      }
      return out;
    };
    const encode = (samples) => {
      total += samples.length;
      const joined = concat([pending, samples], Float32Array);
      const packets = [];
      let at = 0;
      for (; at + frame <= joined.length; at += frame) {
        packets.push(encoder.encodeFloat(joined.subarray(at, at + frame)));
        encoded += frame;
      }
      pending = joined.slice(at);
      return packets;
    };
    const [head, tags] = headers(input, skip);
    const start = [page([head], 0, 2), page([tags], 0)];
    return {
      close: () => encoder.free(),
      push(samples) {
        const packets = encode(resampler(samples));
        return concat([...start.splice(0), ...pages(packets, encoded * scale, false)]);
      },
      end() {
        const packets = encode(resampler.end());
        const length = total;
        const want = total + lookahead - encoded;
        packets.push(...encode(new Float32Array(Math.ceil(want / frame) * frame - pending.length)));
        encoder.free();
        return concat([...start.splice(0), ...pages(packets, skip + length * scale, true)]);
      },
    };
  };
}

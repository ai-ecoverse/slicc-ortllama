const MAGIC = 0x46554747;
const FIRST = 1 << 16;
const STRING = 8;
const ARRAY = 9;
const READERS = {
  0: [1, (v, i) => v.getUint8(i)],
  1: [1, (v, i) => v.getInt8(i)],
  2: [2, (v, i) => v.getUint16(i, true)],
  3: [2, (v, i) => v.getInt16(i, true)],
  4: [4, (v, i) => v.getUint32(i, true)],
  5: [4, (v, i) => v.getInt32(i, true)],
  6: [4, (v, i) => v.getFloat32(i, true)],
  7: [1, (v, i) => v.getUint8(i) !== 0],
  10: [8, (v, i) => Number(v.getBigUint64(i, true))],
  11: [8, (v, i) => Number(v.getBigInt64(i, true))],
  12: [8, (v, i) => v.getFloat64(i, true)],
};
const utf8 = new TextDecoder();
const read = new Map();

export const HEADER_LIMIT = 16 * 1024 * 1024;

class Truncated extends Error {}

function cursor(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let at = 0;
  const take = (n) => {
    if (at + n > bytes.length) throw new Truncated();
    at += n;
    return at - n;
  };
  const known = (type) => {
    if (!READERS[type]) throw new RangeError(`unknown GGUF value type ${type}`);
    return READERS[type];
  };
  const c = {
    u32: () => view.getUint32(take(4), true),
    u64: () => Number(view.getBigUint64(take(8), true)),
    string() {
      const n = c.u64();
      const from = take(n);
      return utf8.decode(bytes.subarray(from, from + n));
    },
    value(type) {
      if (type === STRING) return c.string();
      if (type !== ARRAY) {
        const [size, get] = known(type);
        return get(view, take(size));
      }
      const inner = c.u32();
      const count = c.u64();
      if (inner === STRING) for (let i = 0; i < count; i++) take(c.u64());
      else take(known(inner)[0] * count);
      return { array: inner, count };
    },
  };
  return c;
}

export function parseHeader(bytes, { complete = true } = {}) {
  const c = cursor(bytes);
  try {
    if (c.u32() !== MAGIC) return null;
    const version = c.u32();
    const tensors = c.u64();
    const count = c.u64();
    const meta = {};
    for (let i = 0; i < count; i++) {
      const key = c.string();
      meta[key] = c.value(c.u32());
    }
    return { version, tensors, meta };
  } catch (error) {
    return error instanceof Truncated && !complete ? undefined : null;
  }
}

async function headBytes(handle, length) {
  const out = new Uint8Array(length);
  const { bytesRead } = await handle.read(out, 0, length, 0);
  return out.subarray(0, bytesRead);
}

export async function readHeader(files, rel, { limit = HEADER_LIMIT } = {}) {
  const { size, mtimeMs } = await files.stat(rel);
  const key = files.path(rel);
  const stamp = `${size}:${mtimeMs}`;
  if (read.get(key)?.stamp === stamp) return read.get(key).header;
  const handle = await files.open(rel);
  let header;
  try {
    for (let length = Math.min(FIRST, size); ; length = Math.min(length * 4, size, limit)) {
      const bytes = await headBytes(handle, length);
      const last = length >= size || length >= limit;
      header = parseHeader(bytes, { complete: last });
      if (header !== undefined || last) break;
    }
  } finally {
    await handle.close();
  }
  header ??= null;
  read.set(key, { stamp, header });
  return header;
}

export function describe(header) {
  const meta = header?.meta ?? {};
  const architecture =
    typeof meta['general.architecture'] === 'string' ? meta['general.architecture'] : null;
  if (!architecture) return null;
  const projector = architecture === 'clip' || meta['general.type'] === 'mmproj';
  const decision = !projector && typeof meta[`${architecture}.decision.type`] === 'string';
  const context = meta[`${architecture}.context_length`];
  return {
    architecture,
    projector,
    decision,
    fileType: Number.isInteger(meta['general.file_type']) ? meta['general.file_type'] : null,
    contextLength: Number.isInteger(context) && context > 0 ? context : null,
    vision: projector && meta['clip.has_vision_encoder'] !== false,
    split: Number.isInteger(meta['split.count']) ? meta['split.count'] : null,
  };
}

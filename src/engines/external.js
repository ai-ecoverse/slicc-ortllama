import { posix } from 'node:path';

export const GRAPH_SCAN_LIMIT = 16 * 1024 * 1024;

const scanned = new Map();
const utf8 = new TextDecoder();

function varint(bytes, at, end) {
  let value = 0;
  for (let shift = 0; ; shift += 7) {
    if (at.i >= end || shift > 63) throw new RangeError('truncated varint');
    const b = bytes[at.i++];
    value += (b & 0x7f) * 2 ** shift;
    if (b < 0x80) return value;
  }
}

function* fields(bytes, start, end) {
  const at = { i: start };
  while (at.i < end) {
    const key = varint(bytes, at, end);
    const wire = key % 8;
    if (wire === 0) varint(bytes, at, end);
    else if (wire === 1) at.i += 8;
    else if (wire === 5) at.i += 4;
    else if (wire === 2) {
      const length = varint(bytes, at, end);
      if (at.i + length > end) throw new RangeError('truncated field');
      yield [Math.floor(key / 8), at.i, at.i + length];
      at.i += length;
    } else throw new RangeError(`unsupported wire type ${wire}`);
    if (at.i > end) throw new RangeError('truncated field');
  }
}

const TENSOR = 't';
const SPARSE = 's';
const GRAPH = 'g';
const NODE = 'n';
const ATTRIBUTE = 'a';
const FUNCTION = 'f';

const CHILDREN = {
  model: { 7: GRAPH, 25: FUNCTION },
  [GRAPH]: { 1: NODE, 5: TENSOR, 15: SPARSE },
  [FUNCTION]: { 7: NODE },
  [NODE]: { 5: ATTRIBUTE },
  [ATTRIBUTE]: { 5: TENSOR, 6: GRAPH, 10: TENSOR, 11: GRAPH, 22: SPARSE, 23: SPARSE },
  [SPARSE]: { 1: TENSOR, 2: TENSOR },
};

function location(bytes, start, end) {
  let key = null;
  let value = null;
  for (const [field, s, e] of fields(bytes, start, end)) {
    if (field === 1) key = utf8.decode(bytes.subarray(s, e));
    if (field === 2) value = utf8.decode(bytes.subarray(s, e));
  }
  return key === 'location' ? value : null;
}

function walk(bytes, kind, start, end, out) {
  for (const [field, s, e] of fields(bytes, start, end)) {
    if (kind === TENSOR) {
      if (field === 13) {
        const at = location(bytes, s, e);
        if (at) out.add(at);
      }
      continue;
    }
    const child = CHILDREN[kind][field];
    if (child) walk(bytes, child, s, e, out);
  }
}

export function externalLocations(bytes) {
  const out = new Set();
  walk(bytes, 'model', 0, bytes.length, out);
  return [...out];
}

export async function externalFiles(files, rel) {
  if (!(await files.exists(rel))) return [];
  const size = await files.size(rel);
  if (size > GRAPH_SCAN_LIMIT) return [];
  const key = `${files.path(rel)}:${size}`;
  if (!scanned.has(key)) {
    let names;
    try {
      names = externalLocations(await files.read(rel));
    } catch {
      names = [];
    }
    const dir = posix.dirname(rel);
    scanned.set(
      key,
      names.map((name) => (dir === '.' ? posix.normalize(name) : posix.join(dir, name)))
    );
  }
  return scanned.get(key);
}

export async function missingExternal(files, graphs) {
  const out = [];
  for (const graph of graphs)
    for (const rel of await externalFiles(files, graph))
      if (!out.includes(rel) && !(await files.exists(rel))) out.push(rel);
  return out;
}

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
    const field = Math.floor(key / 8);
    const wire = key % 8;
    if (wire === 0) yield [field, null, null, varint(bytes, at, end)];
    else if (wire === 1) at.i += 8;
    else if (wire === 5) at.i += 4;
    else if (wire === 2) {
      const length = varint(bytes, at, end);
      if (at.i + length > end) throw new RangeError('truncated field');
      yield [field, at.i, at.i + length];
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
  [FUNCTION]: { 7: NODE, 11: ATTRIBUTE },
  [NODE]: { 5: ATTRIBUTE },
  [ATTRIBUTE]: { 5: TENSOR, 6: GRAPH, 10: TENSOR, 11: GRAPH, 22: SPARSE, 23: SPARSE },
  [SPARSE]: { 1: TENSOR, 2: TENSOR },
};

function location(bytes, start, end) {
  let key = null;
  let value = null;
  for (const [field, s, e] of fields(bytes, start, end)) {
    if (field === 1 && s !== null) key = utf8.decode(bytes.subarray(s, e));
    if (field === 2 && s !== null) value = utf8.decode(bytes.subarray(s, e));
  }
  return key === 'location' ? value : null;
}

const EXTERNAL = 1;

function tensorLocation(bytes, start, end) {
  let at = null;
  let stored = 0;
  for (const [field, s, e, value] of fields(bytes, start, end)) {
    if (field === 13 && s !== null) at = location(bytes, s, e) ?? at;
    if (field === 14 && s === null) stored = value;
  }
  return stored === EXTERNAL ? at : null;
}

function walk(bytes, kind, start, end, out) {
  for (const [field, s, e] of fields(bytes, start, end)) {
    const child = s === null ? null : CHILDREN[kind][field];
    if (child === TENSOR) {
      const at = tensorLocation(bytes, s, e);
      if (at) out.add(at);
    } else if (child) walk(bytes, child, s, e, out);
  }
}

export function externalLocations(bytes) {
  const out = new Set();
  walk(bytes, 'model', 0, bytes.length, out);
  return [...out];
}

function inModel(dir, name) {
  if (posix.isAbsolute(name) || /^[A-Za-z]:/.test(name)) return null;
  const rel = posix.normalize(posix.join(dir, name));
  return rel === '.' || rel === '..' || rel.startsWith('../') ? null : rel;
}

export async function externalFiles(files, rel) {
  if (!(await files.exists(rel))) return [];
  const { size, mtimeMs } = await files.stat(rel);
  if (size > GRAPH_SCAN_LIMIT) return [];
  const path = files.path(rel);
  const stamp = `${size}:${mtimeMs}`;
  if (scanned.get(path)?.stamp === stamp) return scanned.get(path).names;
  let found;
  try {
    found = externalLocations(await files.read(rel));
  } catch {
    return [];
  }
  const dir = posix.dirname(rel);
  const names = found.map((name) => inModel(dir, name)).filter((name) => name !== null);
  scanned.set(path, { stamp, names });
  return names;
}

export async function missingExternal(files, graphs) {
  const out = [];
  for (const graph of graphs)
    for (const rel of await externalFiles(files, graph))
      if (!out.includes(rel) && !(await files.exists(rel))) out.push(rel);
  return out;
}

import { readFile } from 'node:fs/promises';

export async function readNpy(path) {
  const bytes = await readFile(path);
  const length = bytes.readUInt16LE(8);
  const header = bytes.subarray(10, 10 + length).toString('latin1');
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shape = /'shape':\s*\(([^)]*)\)/
    .exec(header)[1]
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const body = bytes.subarray(10 + length);
  const buffer = body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength);
  const Type = { '<f4': Float32Array, '<i8': BigInt64Array }[descr];
  const raw = new Type(buffer);
  if (!fortran || shape.length < 2) return { shape, data: raw };
  const data = new Type(raw.length);
  const strides = shape.map((_, i) => shape.slice(0, i).reduce((a, b) => a * b, 1));
  const index = new Array(shape.length).fill(0);
  for (let c = 0; c < raw.length; c++) {
    let f = 0;
    for (let d = 0; d < shape.length; d++) f += index[d] * strides[d];
    data[c] = raw[f];
    for (let d = shape.length - 1; d >= 0; d--) {
      if (++index[d] < shape[d]) break;
      index[d] = 0;
    }
  }
  return { shape, data };
}

const ELEM = { float32: 1, int64: 7, float16: 10 };

function varint(n) {
  const out = [];
  let v = BigInt(n);
  while (v > 127n) {
    out.push(Number((v & 127n) | 128n));
    v >>= 7n;
  }
  out.push(Number(v));
  return out;
}

const bytes = (parts) => parts.flat();
const tag = (field, wire) => varint((field << 3) | wire);
const int = (field, n) => [...tag(field, 0), ...varint(n)];
const msg = (field, body) => [...tag(field, 2), ...varint(body.length), ...body];
const str = (field, s) => msg(field, [...new TextEncoder().encode(s)]);

function valueInfo(name, elem, dims) {
  const shape = dims.map((d) => msg(1, typeof d === 'number' ? int(1, d) : str(2, d)));
  return bytes([str(1, name), msg(2, msg(1, bytes([int(1, elem), msg(2, bytes(shape))])))]);
}

function int64Tensor(name, values) {
  return bytes([
    int(1, values.length),
    int(2, ELEM.int64),
    msg(7, values.flatMap(varint)),
    str(8, name),
  ]);
}

export function sliceModel(names, type) {
  const elem = ELEM[type];
  const nodes = names.map((name, i) =>
    msg(
      1,
      bytes([
        str(1, name),
        str(1, 'starts'),
        str(1, 'ends'),
        str(1, 'axes'),
        str(2, `out.${i}`),
        str(4, 'Slice'),
      ])
    )
  );
  const graph = bytes([
    ...nodes,
    str(2, 'kv_slice'),
    msg(5, int64Tensor('starts', [0])),
    msg(5, int64Tensor('axes', [2])),
    ...names.map((name) => msg(11, valueInfo(name, elem, ['b', 'h', 't', 'd']))),
    msg(11, valueInfo('ends', ELEM.int64, [1])),
    ...names.map((_, i) => msg(12, valueInfo(`out.${i}`, elem, ['b', 'h', 's', 'd']))),
  ]);
  return Uint8Array.from(
    bytes([int(1, 8), str(2, 'ortllama'), msg(7, graph), msg(8, bytes([str(1, ''), int(2, 13)]))])
  );
}

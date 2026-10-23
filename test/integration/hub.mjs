import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';

export const SHA = 'b'.repeat(40);
export const REPO = 'test/bark';
const WEIGHTS = [10, 20];

const varint = (n) => {
  const out = [];
  for (; n > 127; n >>>= 7) out.push((n & 127) | 128);
  return [...out, n];
};
const tag = (field, wire) => varint((field << 3) | wire);
const int = (field, n) => [...tag(field, 0), ...varint(n)];
const msg = (field, body) => [...tag(field, 2), ...varint(body.length), ...body];
const str = (field, s) => msg(field, [...Buffer.from(s)]);
const entry = (key, value) => msg(13, [...str(1, key), ...str(2, value)]);
const vector = (name, n) => [
  ...str(1, name),
  ...msg(2, msg(1, [...int(1, 1), ...msg(2, msg(1, int(1, n)))])),
];

export function addGraph(location, n = 2) {
  const w = [
    ...int(1, n),
    ...int(2, 1),
    ...str(8, 'w'),
    ...entry('location', location),
    ...entry('offset', '0'),
    ...entry('length', String(n * 4)),
    ...int(14, 1),
  ];
  const node = msg(1, [...str(1, 'x'), ...str(1, 'w'), ...str(2, 'y'), ...str(4, 'Add')]);
  const graph = [
    ...node,
    ...str(2, 'g'),
    ...msg(5, w),
    ...msg(11, vector('x', n)),
    ...msg(12, vector('y', n)),
  ];
  return Buffer.from([...int(1, 8), ...msg(7, graph), ...msg(8, [...str(1, ''), ...int(2, 13)])]);
}

export const weights = Buffer.from(new Float32Array(WEIGHTS).buffer);
export const joint = Buffer.from(new Float32Array([30, 40]).buffer);
export const sha256 = (content) => createHash('sha256').update(content).digest('hex');
const gitSha = (content) =>
  createHash('sha1').update(`blob ${content.length}\0`).update(content).digest('hex');

export const PARAKEET = {
  'config.json': JSON.stringify({ model_type: 'parakeet_tdt' }),
  'generation_config.json': JSON.stringify({ eos_token_id: 1 }),
  'preprocessor_config.json': '{}',
  'tokenizer.json': JSON.stringify({ model: { vocab: { a: 0 } } }),
  'onnx/encoder_model.onnx': addGraph('encoder_model.onnx_data'),
  'onnx/encoder_model.onnx_data': weights,
  'onnx/decoder_joint_model.onnx': addGraph('decoder_joint_model.onnx_data'),
  'onnx/decoder_joint_model.onnx_data': joint,
};

export const KEV = {
  'kev-tiny/manifest.json': JSON.stringify({
    name: 'kev-tiny',
    special: { state: 1 },
    files: { tokenizer: 'r-1/tokenizer.json' },
    variants: { q8f32: { model: 'r-1/q8f32/model.onnx', data: ['r-1/q8f32/model.onnx.data'] } },
  }),
  'kev-tiny/r-1/tokenizer.json': '{}',
  'kev-tiny/r-1/q8f32/model.onnx': addGraph('model.onnx.data'),
  'kev-tiny/r-1/q8f32/model.onnx.data': weights,
};

const isLfs = (rel) => /\.onnx|_data|\.data/.test(rel);

export const repoOf = (cache) => join(cache, `models--${REPO.replace('/', '--')}`);
export const snapshotOf = (cache) => join(repoOf(cache), 'snapshots', SHA);

export async function hubCache(layout, files, { stray = null } = {}) {
  const cache = await mkdtemp(join(tmpdir(), `ortllama-hub${layout}-`));
  const repo = repoOf(cache);
  const snapshot = snapshotOf(cache);
  await mkdir(join(repo, 'refs'), { recursive: true });
  await writeFile(join(repo, 'refs', 'main'), SHA);
  const shared = {};
  for (const [rel, text] of Object.entries(files)) {
    const content = Buffer.from(text);
    const lfs = isLfs(rel);
    const blob = join(repo, 'blobs', lfs ? sha256(content) : gitSha(content));
    await mkdir(dirname(blob), { recursive: true });
    if (layout === 2 && lfs) {
      const xet = sha256(Buffer.concat([Buffer.from(rel), content]));
      const store = rel === stray ? join(cache, '..', `${xet}-elsewhere`) : join(cache, 'blobs');
      shared[rel] = join(store, xet.slice(0, 2), xet);
      await mkdir(dirname(shared[rel]), { recursive: true });
      await writeFile(shared[rel], content);
      await symlink(relative(dirname(blob), shared[rel]), blob);
    } else await writeFile(blob, content);
    const link = join(snapshot, rel);
    await mkdir(dirname(link), { recursive: true });
    await symlink(relative(dirname(link), blob), link);
  }
  return { cache, shared };
}

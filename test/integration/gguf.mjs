import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';

const utf8 = new TextEncoder();

function u32(n) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n);
  return b;
}

function u64(n) {
  const b = Buffer.alloc(8);
  b.writeBigUInt64LE(BigInt(n));
  return b;
}

function str(s) {
  const bytes = utf8.encode(s);
  return Buffer.concat([u64(bytes.length), bytes]);
}

function value(v) {
  if (typeof v === 'string') return [u32(8), str(v)];
  if (typeof v === 'boolean') return [u32(7), Buffer.from([v ? 1 : 0])];
  if (Array.isArray(v)) return [u32(9), u32(8), u64(v.length), ...v.map((s) => str(s))];
  if (Number.isInteger(v)) return [u32(4), u32(v)];
  const f = Buffer.alloc(4);
  f.writeFloatLE(v);
  return [u32(6), f];
}

export function ggufBytes(meta, { tail = 64 } = {}) {
  const entries = Object.entries(meta);
  return Buffer.concat([
    Buffer.from('GGUF'),
    u32(3),
    u64(0),
    u64(entries.length),
    ...entries.flatMap(([k, v]) => [str(k), ...value(v)]),
    Buffer.alloc(tail, 7),
  ]);
}

export const chatMeta = (arch = 'llama', extra = {}) => ({
  'general.architecture': arch,
  'general.type': 'model',
  [`${arch}.context_length`]: 4096,
  'tokenizer.ggml.tokens': ['<s>', '</s>', 'a'],
  'general.file_type': 2,
  ...extra,
});

export const projectorMeta = () => ({
  'general.architecture': 'clip',
  'general.type': 'mmproj',
  'clip.has_vision_encoder': true,
  'general.file_type': 7,
});

export const decisionMeta = () => chatMeta('modern-bert', { 'modern-bert.decision.type': 'mask' });

export const SHA = 'c'.repeat(40);

export async function ggufCache(repos) {
  const root = await mkdtemp(join(tmpdir(), 'ortllama-gguf-'));
  for (const [repo, files] of Object.entries(repos)) {
    const dir = join(root, `models--${repo.replace('/', '--')}`);
    const snapshot = join(dir, 'snapshots', SHA);
    await mkdir(join(dir, 'refs'), { recursive: true });
    await writeFile(join(dir, 'refs', 'main'), SHA);
    for (const [rel, content] of Object.entries(files)) {
      const blob = join(dir, 'blobs', createHash('sha256').update(content).digest('hex'));
      await mkdir(dirname(blob), { recursive: true });
      await writeFile(blob, content);
      const link = join(snapshot, rel);
      await mkdir(dirname(link), { recursive: true });
      await symlink(relative(dirname(link), blob), link);
    }
  }
  return root;
}

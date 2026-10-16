import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';

export const SHA = 'a'.repeat(40);

function manifest(name, variants) {
  return {
    name,
    special: { state: 1, q: 2, opt: 3, opt_end: 4, decide: 5 },
    files: { head: 'r-1/head.safetensors', tokenizer: 'r-1/tokenizer.json' },
    variants: Object.fromEntries(
      variants.map((v) => [
        v,
        { model: `r-1/${v}/model.onnx`, data: [`r-1/${v}/model.onnx.data`], max_positions: 4096 },
      ])
    ),
  };
}

export async function fakeCache() {
  const root = await mkdtemp(join(tmpdir(), 'ortllama-cache-'));
  const repo = join(root, 'models--test--kev');
  const snapshot = join(repo, 'snapshots', SHA);
  await mkdir(join(repo, 'refs'), { recursive: true });
  await writeFile(join(repo, 'refs', 'main'), SHA);
  const put = async (rel, content) => {
    const blob = join(repo, 'blobs', createHash('sha256').update(content).digest('hex'));
    await mkdir(dirname(blob), { recursive: true });
    await writeFile(blob, content);
    const link = join(snapshot, rel);
    await mkdir(dirname(link), { recursive: true });
    await symlink(relative(dirname(link), blob), link).catch(() => {});
  };
  for (const name of ['kev-tiny', 'kev-part']) {
    await put(`${name}/manifest.json`, JSON.stringify(manifest(name, ['q8f32', 'fp32'])));
    await put(`${name}/r-1/head.safetensors`, `head ${name}`);
    await put(`${name}/r-1/tokenizer.json`, '{}');
  }
  for (const v of ['q8f32', 'fp32']) {
    await put(`kev-tiny/r-1/${v}/model.onnx`, `onnx ${v}`);
    await put(`kev-tiny/r-1/${v}/model.onnx.data`, `data ${v}`);
  }
  await put('README.md', 'not a model');
  await mkdir(join(root, 'models--test--empty', 'refs'), { recursive: true });
  return root;
}

export async function fakeHome(aliases = {}) {
  const home = await mkdtemp(join(tmpdir(), 'ortllama-home-'));
  await writeFile(join(home, 'config.json'), JSON.stringify({ aliases }));
  return home;
}

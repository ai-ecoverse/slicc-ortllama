import { link, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

export async function stageFiles(files, rels, root = tmpdir()) {
  const dir = await mkdtemp(join(root, 'ortllama-'));
  const release = () => rm(dir, { recursive: true, force: true });
  try {
    for (const rel of rels) {
      const from = await files.realPath(rel);
      const to = join(dir, rel);
      await mkdir(dirname(to), { recursive: true });
      await link(from, to);
    }
  } catch (error) {
    await release();
    throw error;
  }
  return { path: (rel) => join(dir, rel), release };
}

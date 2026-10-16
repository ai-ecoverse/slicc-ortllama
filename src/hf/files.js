import { readFile, stat } from 'node:fs/promises';
import { join, normalize, sep } from 'node:path';
import { isDir, listDir } from './cache.js';

function inside(root, rel) {
  const path = normalize(join(root, rel));
  if (path !== root && !path.startsWith(root.endsWith(sep) ? root : root + sep)) {
    throw new RangeError(`${rel} is outside the model directory`);
  }
  return path;
}

export function modelFiles(root) {
  const files = {
    root,
    path: (rel) => inside(root, rel),
    async exists(rel) {
      try {
        return (await stat(inside(root, rel))).isFile();
      } catch {
        return false;
      }
    },
    async size(rel) {
      return (await stat(inside(root, rel))).size;
    },
    async read(rel) {
      return new Uint8Array(await readFile(inside(root, rel)));
    },
    text: (rel) => readFile(inside(root, rel), 'utf8'),
    async json(rel) {
      return JSON.parse(await files.text(rel));
    },
    async dirs() {
      const out = [];
      for (const name of await listDir(root)) {
        if (!name.startsWith('.') && (await isDir(join(root, name)))) out.push(name);
      }
      return out.sort();
    },
    sub: (rel) => modelFiles(inside(root, rel)),
  };
  return files;
}

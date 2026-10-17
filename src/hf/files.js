import { readFileSync, realpathSync } from 'node:fs';
import { readFile, realpath, stat } from 'node:fs/promises';
import { join, normalize, posix, sep } from 'node:path';
import { isDir, listDir } from './cache.js';

function within(root, path) {
  return path === root || path.startsWith(root.endsWith(sep) ? root : root + sep);
}

function inside(root, rel) {
  const path = normalize(join(root, rel));
  if (!within(root, path)) throw new RangeError(`${rel} is outside the model directory`);
  return path;
}

function missing(rel) {
  return Object.assign(new Error(`${rel} is not in the model directory`), { code: 'ENOENT' });
}

function glob(pattern) {
  const parts = pattern.split('*').map((p) => p.replace(/[.+?^${}()|[\]\\]/g, '\\$&'));
  return new RegExp(`^${parts.join('.*')}$`);
}

export function modelFiles(root, { allow = [root] } = {}) {
  let roots;
  let rootsSync;
  const allowed = async (path) => {
    roots ??= Promise.all(allow.map((dir) => realpath(dir).catch(() => dir)));
    const real = await realpath(path);
    return (await roots).some((dir) => within(dir, real)) ? real : null;
  };
  const real = async (rel) => {
    const path = await allowed(inside(root, rel)).catch(() => null);
    if (!path) throw missing(rel);
    return path;
  };
  const realSync = (rel) => {
    rootsSync ??= allow.map((dir) => {
      try {
        return realpathSync(dir);
      } catch {
        return dir;
      }
    });
    let path;
    try {
      path = realpathSync(inside(root, rel));
    } catch {
      throw missing(rel);
    }
    if (!rootsSync.some((dir) => within(dir, path))) throw missing(rel);
    return path;
  };
  const files = {
    root,
    path: (rel) => inside(root, rel),
    async exists(rel) {
      try {
        return (await stat(await real(rel))).isFile();
      } catch {
        return false;
      }
    },
    async size(rel) {
      return (await stat(await real(rel))).size;
    },
    async read(rel) {
      return new Uint8Array(await readFile(await real(rel)));
    },
    readSync(rel) {
      const data = readFileSync(realSync(rel));
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    },
    text: async (rel) => readFile(await real(rel), 'utf8'),
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
    async bytes(patterns) {
      const seen = new Set();
      let total = 0;
      for (const pattern of patterns) {
        const dir = posix.dirname(pattern);
        const name = posix.basename(pattern);
        const names = name.includes('*')
          ? (await listDir(inside(root, dir))).filter((n) => glob(name).test(n))
          : [name];
        for (const n of names) {
          const rel = dir === '.' ? n : `${dir}/${n}`;
          if (seen.has(rel) || !(await files.exists(rel))) continue;
          seen.add(rel);
          total += await files.size(rel);
        }
      }
      return total;
    },
    sub: (rel) => modelFiles(inside(root, rel), { allow }),
  };
  return files;
}

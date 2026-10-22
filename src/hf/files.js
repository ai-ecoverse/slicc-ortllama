import { readFileSync, realpathSync } from 'node:fs';
import { open, readFile, readlink, realpath, stat } from 'node:fs/promises';
import { basename, dirname, join, normalize, posix, resolve, sep } from 'node:path';
import { isDir, listDir, repoDir } from './cache.js';

const ETAG = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/;
const STORED = /^[0-9a-f]{64}$/;
const READ_MAX = 2 ** 31 - 1;
const CHUNK = 2 ** 30;

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

export function glob(pattern) {
  const parts = pattern.split('*').map((p) => p.replace(/[.+?^${}()|[\]\\]/g, '\\$&'));
  return new RegExp(`^${parts.join('.*')}$`);
}

function stored(store, path) {
  const name = basename(path);
  const shard = dirname(path);
  return (
    store !== null &&
    dirname(shard) === store &&
    STORED.test(name) &&
    basename(shard) === name.slice(0, 2)
  );
}

function permitted(dirs, path) {
  return dirs.slice(1).some((dir) => within(dir, path)) || stored(dirs[0], path);
}

async function readAll(path, max, chunk) {
  const { size } = await stat(path);
  if (size <= max) return new Uint8Array(await readFile(path));
  const handle = await open(path);
  try {
    const out = new Uint8Array(size);
    let at = 0;
    for (let got = 1; at < size && got > 0; at += got)
      got = (await handle.read(out, at, Math.min(chunk, size - at), at)).bytesRead;
    return out.subarray(0, at);
  } finally {
    await handle.close();
  }
}

export function hubFiles(cache, repo, snapshot, type = 'model') {
  return modelFiles(snapshot, {
    allow: [snapshot, join(repoDir(cache, repo, type), 'blobs')],
    shared: join(cache, 'blobs'),
  });
}

export function modelFiles(
  root,
  { allow = [root], shared = null, readMax = READ_MAX, chunk = CHUNK } = {}
) {
  let roots;
  let rootsSync;
  const dirs = [shared, ...allow];
  const allowed = async (path) => {
    roots ??= Promise.all(dirs.map((dir) => dir && realpath(dir).catch(() => dir)));
    const real = await realpath(path);
    return permitted(await roots, real) ? real : null;
  };
  const real = async (rel) => {
    const path = await allowed(inside(root, rel)).catch(() => null);
    if (!path) throw missing(rel);
    return path;
  };
  const realSync = (rel) => {
    rootsSync ??= dirs.map((dir) => {
      try {
        return dir && realpathSync(dir);
      } catch {
        return dir;
      }
    });
    const lexical = inside(root, rel);
    let path;
    try {
      path = realpathSync(lexical);
    } catch {
      throw missing(rel);
    }
    if (!permitted(rootsSync, path)) throw missing(rel);
    return path;
  };
  const files = {
    root,
    path: (rel) => inside(root, rel),
    realPath: (rel) => real(rel),
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
    async stat(rel) {
      const { size, mtimeMs } = await stat(await real(rel));
      return { size, mtimeMs };
    },
    async read(rel) {
      return readAll(await real(rel), readMax, chunk);
    },
    open: async (rel) => open(await real(rel)),
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
    async list(rel) {
      return listDir(await real(rel).catch(() => ''));
    },
    async matches(patterns) {
      const seen = new Set();
      for (const pattern of patterns) {
        const dir = posix.dirname(pattern);
        const name = posix.basename(pattern);
        const names = name.includes('*')
          ? (await listDir(inside(root, dir))).filter((n) => glob(name).test(n))
          : [name];
        for (const n of names) {
          const rel = dir === '.' ? n : `${dir}/${n}`;
          if (!seen.has(rel) && (await files.exists(rel))) seen.add(rel);
        }
      }
      return [...seen];
    },
    async bytes(patterns) {
      let total = 0;
      for (const rel of await files.matches(patterns)) total += await files.size(rel);
      return total;
    },
    async colocated(rels) {
      const places = new Set();
      for (const rel of rels) {
        const path = await real(rel).catch(() => null);
        if (path) places.add(dirname(path));
      }
      return places.size <= 1;
    },
    async blob(rel) {
      const path = await real(rel);
      const link = inside(root, rel);
      const target = await readlink(link).then(
        (to) => resolve(dirname(link), to),
        () => link
      );
      const name = basename(target);
      const dir = await realpath(dirname(target));
      if (
        !ETAG.test(name) ||
        !(await roots).slice(1).some((d) => d === dir && basename(d) === 'blobs')
      )
        return null;
      const { size, mtimeMs } = await stat(path);
      return { path, etag: name, size, mtimeMs };
    },
    sub: (rel) => modelFiles(inside(root, rel), { allow, shared, readMax, chunk }),
  };
  return files;
}

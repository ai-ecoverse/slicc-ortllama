import { readFile, readlink, realpath } from 'node:fs/promises';
import { delimiter, isAbsolute, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { isDir, listDir } from '../hf/cache.js';
import { HttpError } from '../http/errors.js';

const PNPM_LINK = /^[0-9a-f]{64}$/;
const loaded = new Map();

export function pnpmHomes(env = process.env, platform = process.platform) {
  const home = env.HOME;
  const local = home && join(home, '.local', 'share', 'pnpm');
  const mac = home && join(home, 'Library', 'pnpm');
  const defaults = platform === 'darwin' ? [mac, local] : [local, mac];
  return [...new Set([env.PNPM_HOME, '/usr/local/share/pnpm', ...defaults].filter(Boolean))];
}

async function liveLink(base, id) {
  try {
    const target = await readlink(join(base, id));
    return isAbsolute(target) ? target : resolve(base, target);
  } catch {
    return null;
  }
}

export async function moduleRoots(env = process.env, platform = process.platform) {
  const roots = (env.ORTLLAMA_MODULE_PATH ?? '').split(delimiter).filter(Boolean);
  for (const home of pnpmHomes(env, platform)) {
    for (const layout of await listDir(join(home, 'global'))) {
      const base = join(home, 'global', layout);
      if (await isDir(join(base, 'node_modules'))) roots.push(base);
      for (const id of await listDir(base)) {
        const target = PNPM_LINK.test(id) ? await liveLink(base, id) : null;
        if (target && !roots.includes(target) && (await isDir(join(target, 'node_modules'))))
          roots.push(target);
      }
    }
  }
  return roots;
}

async function readJson(path) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch {
    return null;
  }
}

export function pickExport(target) {
  if (typeof target === 'string') return target;
  if (Array.isArray(target)) {
    for (const item of target) {
      const found = pickExport(item);
      if (found) return found;
    }
    return null;
  }
  if (!target || typeof target !== 'object') return null;
  for (const condition of ['import', 'node', 'default', 'require']) {
    const found = pickExport(target[condition]);
    if (found) return found;
  }
  return null;
}

export function entryOf(pkg) {
  const { exports } = pkg;
  if (exports !== undefined) {
    const root =
      typeof exports === 'object' && !Array.isArray(exports) && '.' in exports
        ? exports['.']
        : exports;
    return pickExport(root);
  }
  return pkg.module ?? pkg.main ?? 'index.js';
}

export function compareVersions(a = '0', b = '0') {
  const pa = String(a)
    .split(/[.+-]/)
    .map((x) => Number.parseInt(x, 10) || 0);
  const pb = String(b)
    .split(/[.+-]/)
    .map((x) => Number.parseInt(x, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if ((pa[i] ?? 0) !== (pb[i] ?? 0)) return (pa[i] ?? 0) - (pb[i] ?? 0);
  }
  return 0;
}

async function candidate(root, name) {
  const pkg = await readJson(join(root, 'node_modules', name, 'package.json'));
  if (!pkg) return null;
  const entry = entryOf(pkg);
  if (!entry) return null;
  const owner = await readJson(join(root, 'package.json'));
  return { root, entry, version: pkg.version, direct: Boolean(owner?.dependencies?.[name]) };
}

export async function locate(name, env = process.env, platform = process.platform) {
  const explicit = (env.ORTLLAMA_MODULE_PATH ?? '').split(delimiter).filter(Boolean);
  const found = [];
  for (const root of await moduleRoots(env, platform)) {
    const hit = await candidate(root, name);
    if (hit) found.push({ ...hit, explicit: explicit.includes(root) });
  }
  if (found.length === 0) return null;
  const first = found.find((c) => c.explicit);
  const best =
    first ??
    found.sort(
      (a, b) => Number(b.direct) - Number(a.direct) || compareVersions(b.version, a.version)
    )[0];
  return realpath(join(best.root, 'node_modules', name, best.entry));
}

export function missingModule(error, name) {
  const message = String(error?.message ?? '');
  return (
    (error?.code === 'ERR_MODULE_NOT_FOUND' || error?.code === 'MODULE_NOT_FOUND') &&
    (message.includes(`Cannot find package '${name}'`) ||
      message.includes(`Cannot find module '${name}'`))
  );
}

async function load(name, env, importer, platform) {
  try {
    return await importer(name);
  } catch (error) {
    if (!missingModule(error, name)) throw error;
  }
  const file = await locate(name, env, platform);
  if (file) return importer(pathToFileURL(file).href);
  throw new HttpError(
    501,
    `this model needs the npm package ${name}: install it next to ortllama (npm install ${name}; pnpm add -g ${name} in slicc-kernel), or add its project folder to ORTLLAMA_MODULE_PATH`,
    { code: 'engine_unavailable' }
  );
}

export function optional(
  name,
  {
    env = process.env,
    importer = (specifier) => import(specifier),
    platform = process.platform,
  } = {}
) {
  if (!loaded.has(name)) {
    loaded.set(
      name,
      load(name, env, importer, platform).catch((error) => {
        loaded.delete(name);
        throw error;
      })
    );
  }
  return loaded.get(name);
}

import { createRequire } from 'node:module';
import { delimiter, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { isDir, listDir } from '../hf/cache.js';
import { HttpError } from '../http/errors.js';

const PNPM_LINK = /^[0-9a-f]{64}$/;
const loaded = new Map();

export async function moduleRoots(env = process.env) {
  const roots = (env.ORTLLAMA_MODULE_PATH ?? '').split(delimiter).filter(Boolean);
  const homes = new Set(
    [
      env.PNPM_HOME,
      '/usr/local/share/pnpm',
      env.HOME && join(env.HOME, '.local', 'share', 'pnpm'),
      env.HOME && join(env.HOME, 'Library', 'pnpm'),
    ].filter(Boolean)
  );
  for (const home of homes) {
    for (const layout of await listDir(join(home, 'global'))) {
      const base = join(home, 'global', layout);
      if (await isDir(join(base, 'node_modules'))) roots.push(base);
      for (const id of await listDir(base)) {
        if (id === 'node_modules' || PNPM_LINK.test(id)) continue;
        if (await isDir(join(base, id, 'node_modules'))) roots.push(join(base, id));
      }
    }
  }
  return roots;
}

function missingModule(error, name) {
  return (
    (error?.code === 'ERR_MODULE_NOT_FOUND' || error?.code === 'MODULE_NOT_FOUND') &&
    String(error.message).includes(name)
  );
}

async function load(name, env, importer) {
  try {
    return await importer(name);
  } catch (error) {
    if (!missingModule(error, name)) throw error;
  }
  for (const root of await moduleRoots(env)) {
    let resolved;
    try {
      resolved = createRequire(join(root, 'package.json')).resolve(name);
    } catch {
      continue;
    }
    return importer(pathToFileURL(resolved).href);
  }
  throw new HttpError(
    501,
    `this model needs the npm package ${name}: install it next to ortllama (npm install ${name}; pnpm add -g ${name} in slicc-kernel), or add its project folder to ORTLLAMA_MODULE_PATH`,
    { code: 'engine_unavailable' }
  );
}

export function optional(
  name,
  { env = process.env, importer = (specifier) => import(specifier) } = {}
) {
  if (!loaded.has(name)) {
    loaded.set(
      name,
      load(name, env, importer).catch((error) => {
        loaded.delete(name);
        throw error;
      })
    );
  }
  return loaded.get(name);
}

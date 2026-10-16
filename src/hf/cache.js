import { readdir, readFile, stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

const SHA = /^[0-9a-f]{40}$/;

export function hubCache(env = process.env, home = homedir()) {
  if (env.HF_HUB_CACHE) return env.HF_HUB_CACHE;
  if (env.HF_HOME) return join(env.HF_HOME, 'hub');
  if (env.XDG_CACHE_HOME) return join(env.XDG_CACHE_HOME, 'huggingface', 'hub');
  return join(home, '.cache', 'huggingface', 'hub');
}

export function parseModelId(id) {
  if (typeof id !== 'string' || id.trim() === '')
    throw new TypeError('model id must be a non-empty string');
  let rest = id.trim();
  let variant = null;
  const colon = rest.lastIndexOf(':');
  if (colon > 0) {
    variant = rest.slice(colon + 1) || null;
    rest = rest.slice(0, colon);
  }
  let revision = null;
  const at = rest.lastIndexOf('@');
  if (at > 0) {
    revision = rest.slice(at + 1) || null;
    rest = rest.slice(0, at);
  }
  const parts = rest.split('/').filter(Boolean);
  if (parts.length < 2)
    throw new TypeError(`model id ${id} must be org/repo[/subfolder][@revision][:variant]`);
  return {
    repo: `${parts[0]}/${parts[1]}`,
    subfolder: parts.slice(2).join('/'),
    revision,
    variant,
  };
}

export function formatModelId({ repo, subfolder = '', revision = null, variant = null }) {
  return `${repo}${subfolder ? `/${subfolder}` : ''}${revision ? `@${revision}` : ''}${variant ? `:${variant}` : ''}`;
}

export function repoDir(cache, repo) {
  return join(cache, `models--${repo.replace('/', '--')}`);
}

export async function resolveRevision(cache, repo, revision = 'main') {
  const dir = repoDir(cache, repo);
  const sha = SHA.test(revision)
    ? revision
    : (await readText(join(dir, 'refs', ...revision.split('/')))).trim();
  if (!sha) return null;
  const snapshot = join(dir, 'snapshots', sha);
  return (await isDir(snapshot)) ? { sha, snapshot } : null;
}

export async function cachedRepos(cache) {
  const out = [];
  for (const name of await listDir(cache)) {
    if (!name.startsWith('models--')) continue;
    const [org, ...rest] = name.slice('models--'.length).split('--');
    if (org && rest.length > 0) out.push(`${org}/${rest.join('--')}`);
  }
  return out.sort();
}

export async function isDir(path) {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}

export async function listDir(path) {
  try {
    return await readdir(path);
  } catch {
    return [];
  }
}

async function readText(path) {
  try {
    return await readFile(path, 'utf8');
  } catch {
    return '';
  }
}

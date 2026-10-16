import { readFileSync } from 'node:fs';
import { loadBackend } from '#ort';
import { createCatalog, localRoots } from './catalog.js';
import { listenPort, ortllamaHome, parseDuration, readConfig } from './config.js';
import { hubCache } from './hf/cache.js';
import { createServer } from './http/server.js';
import { attachRealtime } from './realtime/socket.js';
import { createRoutes } from './routes.js';
import { Runtime } from './runtime.js';

export const VERSION = JSON.parse(
  readFileSync(new URL('../package.json', import.meta.url), 'utf8')
).version;

export { createCatalog, downloadHint } from './catalog.js';
export { DEFAULT_ALIASES, DEFAULT_PORT, ROLES } from './config.js';
export { hubCache, parseModelId } from './hf/cache.js';
export { PRIORITY, Runtime } from './runtime.js';

export async function openCatalog({
  env = process.env,
  home = ortllamaHome(env),
  cache = hubCache(env),
  modelPath = localRoots(env),
} = {}) {
  const config = await readConfig(home);
  return createCatalog({ cache, home, config, modelPath });
}

function positive(value, fallback) {
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 ? n : fallback;
}

export function hostsFor(host, env = process.env) {
  const extra = (env.ORTLLAMA_HOSTS ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (extra.length > 0) return [...extra, host];
  return host === '0.0.0.0' || host === '::' ? ['*'] : [host];
}

export function createRuntime({
  env = process.env,
  backend = () => loadBackend({ env }),
  keepAlive,
  maxLoaded,
} = {}) {
  return new Runtime({
    loadBackend: backend,
    keepAliveMs: parseDuration(keepAlive ?? env.ORTLLAMA_KEEP_ALIVE, 300_000),
    maxLoaded: positive(maxLoaded ?? env.ORTLLAMA_MAX_LOADED, 1),
  });
}

export async function startServer({
  env = process.env,
  port,
  host = '127.0.0.1',
  apiKey,
  catalog,
  runtime,
  ...rest
} = {}) {
  const cat = catalog ?? (await openCatalog({ env, ...rest }));
  const rt = runtime ?? createRuntime({ env, ...rest });
  const routes = createRoutes({ catalog: cat, runtime: rt, version: VERSION });
  const origins = (env.ORTLLAMA_ORIGINS ?? '').split(',').filter(Boolean);
  const options = { apiKey: apiKey ?? env.ORTLLAMA_API_KEY, origins, hosts: hostsFor(host, env) };
  const server = createServer(routes, options);
  const realtime = attachRealtime(server, { catalog: cat, runtime: rt, ...options });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(listenPort(port, env), host, resolve);
  });
  const bound = server.address().port;
  return {
    server,
    catalog: cat,
    runtime: rt,
    port: bound,
    url: `http://${host.includes(':') ? `[${host}]` : host}:${bound}`,
    async close() {
      await realtime.close();
      await new Promise((resolve) => server.close(() => resolve()));
      await rt.close();
    },
  };
}

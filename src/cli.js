import { spawn as nodeSpawn } from 'node:child_process';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { DEFAULT_ALIASES, ortllamaHome, ROLES, readConfig, writeConfig } from './config.js';
import { createRuntime, openCatalog, startServer, VERSION } from './index.js';

const USAGE = `ortllama ${VERSION}

Usage:
  ortllama serve [--port N] [--host H] [--detach]
  ortllama ls [--json]
  ortllama ps [--json]
  ortllama alias [role [model]] [--unset role]
  ortllama check <model> [--fixtures file] [--json]

Models come from the Hugging Face cache; fill it with hf download.`;

const runFile = (home) => join(home, 'run', 'ortllama.json');

export const FIXTURES = fileURLToPath(new URL('../fixtures/', import.meta.url));

export async function readFixtures(path, { optional = false } = {}) {
  let text;
  try {
    text = await readFile(path, 'utf8');
  } catch (error) {
    if (optional && error.code === 'ENOENT') return null;
    throw error;
  }
  const { fixtures, ...about } = JSON.parse(text);
  for (const f of fixtures)
    if (f.file) f.audio = (await readFile(join(dirname(path), f.file))).toString('base64');
  return { path, about, fixtures };
}

function table(rows) {
  const widths = rows[0].map((_, i) => Math.max(...rows.map((r) => String(r[i]).length)));
  return rows
    .map((r) =>
      r
        .map((c, i) => String(c).padEnd(widths[i]))
        .join('  ')
        .trimEnd()
    )
    .join('\n');
}

function megabytes(bytes) {
  return typeof bytes === 'number' ? `${Math.round(bytes / 1e6)} MB` : '-';
}

async function readRun(home) {
  try {
    return JSON.parse(await readFile(runFile(home), 'utf8'));
  } catch {
    return null;
  }
}

async function serve(values, io) {
  const home = ortllamaHome(io.env);
  if (values.detach) return detach(values, io, home);
  const port = values.port === undefined ? undefined : Number(values.port);
  const server = await (io.startServer ?? startServer)({ env: io.env, port, host: values.host });
  await mkdir(join(home, 'run'), { recursive: true });
  await writeFile(
    runFile(home),
    JSON.stringify({ pid: process.pid, url: server.url, port: server.port, version: VERSION })
  );
  io.out.write(`ortllama ${VERSION} listening on ${server.url}\n`);
  const stop = async () => {
    await server.close();
    await rm(runFile(home), { force: true });
  };
  for (const signal of ['SIGINT', 'SIGTERM']) io.signals.once(signal, stop);
  return null;
}

async function answers(io, url) {
  try {
    return (await io.fetch(`${url}/`)).ok;
  } catch {
    return false;
  }
}

async function detach(values, io, home) {
  const args = [
    'serve',
    ...(values.port ? ['--port', values.port] : []),
    ...(values.host ? ['--host', values.host] : []),
  ];
  const running = await readRun(home);
  if (running && (await answers(io, running.url))) {
    io.out.write(`ortllama is already running on ${running.url} (pid ${running.pid})\n`);
    return 0;
  }
  await rm(runFile(home), { force: true });
  const child = io.spawn(process.execPath, [io.binPath, ...args], {
    detached: true,
    stdio: 'ignore',
    env: io.env,
  });
  child.unref();
  for (let i = 0; i < io.waitSteps; i++) {
    const run = await readRun(home);
    if (run) {
      io.out.write(
        `ortllama ${VERSION} started in the background on ${run.url} (pid ${run.pid})\n`
      );
      return 0;
    }
    await io.sleep(100);
  }
  io.err.write('ortllama did not start; run ortllama serve in the foreground to see why\n');
  return 1;
}

async function ls(values, io) {
  const catalog = await openCatalog({ env: io.env });
  const models = await catalog.list();
  if (values.json) {
    io.out.write(`${JSON.stringify({ models, aliases: catalog.aliases }, null, 2)}\n`);
    return 0;
  }
  if (models.length === 0) {
    io.out.write(`no models ortllama can run in ${catalog.cache}\n`);
    return 0;
  }
  const rows = [['MODEL', 'ENGINE', 'VARIANTS', 'REVISION']];
  for (const m of models) {
    rows.push([
      m.id,
      m.engine,
      m.variants.map((v) => (v === m.defaultVariant ? `${v}*` : v)).join(','),
      m.revision ? m.revision.slice(0, 7) : 'local',
    ]);
  }
  io.out.write(`${table(rows)}\n`);
  return 0;
}

async function ps(values, io) {
  const home = ortllamaHome(io.env);
  const run = await readRun(home);
  if (!run) {
    io.err.write('ortllama is not running\n');
    return 1;
  }
  let models;
  let memory;
  try {
    const headers = io.env.ORTLLAMA_API_KEY
      ? { authorization: `Bearer ${io.env.ORTLLAMA_API_KEY}` }
      : {};
    const response = await io.fetch(`${run.url}/ortllama/ps`, { headers });
    const body = await response.json();
    if (!response.ok) {
      io.err.write(
        `ortllama on ${run.url} answered ${response.status}: ${body.error?.message ?? 'no message'}\n`
      );
      return 1;
    }
    models = body.models;
    memory = body.memory ?? null;
  } catch {
    io.err.write(`ortllama is not answering on ${run.url}\n`);
    return 1;
  }
  if (values.json)
    io.out.write(`${JSON.stringify({ url: run.url, pid: run.pid, models, memory }, null, 2)}\n`);
  else {
    if (models.length === 0) io.out.write(`ortllama on ${run.url} has no models loaded\n`);
    else {
      const rows = [['MODEL', 'VARIANT', 'ENGINE', 'PROVIDERS', 'SIZE', 'EXPIRES']];
      for (const m of models)
        rows.push([
          m.id,
          m.variant,
          m.engine,
          m.executionProviders.join(','),
          megabytes(m.modelBytes),
          m.expiresAt ?? 'never',
        ]);
      io.out.write(`${table(rows)}\n`);
    }
    if (memory) io.out.write(`process rss ${megabytes(memory.rss)}\n`);
  }
  return 0;
}

async function alias(values, positionals, io) {
  const home = ortllamaHome(io.env);
  const config = await readConfig(home);
  const [role, model] = positionals;
  if (values.unset) {
    delete config.aliases[values.unset];
    await writeConfig(home, config);
    io.out.write(`${values.unset} → ${DEFAULT_ALIASES[values.unset] ?? '(none)'}\n`);
    return 0;
  }
  if (role && model) {
    config.aliases[role] = model;
    await writeConfig(home, config);
    io.out.write(`${role} → ${model}\n`);
    return 0;
  }
  const merged = { ...DEFAULT_ALIASES, ...config.aliases };
  const names = role ? [role] : [...new Set([...ROLES, ...Object.keys(merged)])];
  const rows = [['ROLE', 'MODEL']];
  for (const name of names)
    rows.push([
      name,
      merged[name] ? `${merged[name]}${config.aliases[name] ? '' : ' (default)'}` : '-',
    ]);
  io.out.write(`${table(rows)}\n`);
  return 0;
}

async function check(values, positionals, io) {
  const [id] = positionals;
  if (!id) {
    io.err.write('ortllama check needs a model\n');
    return 2;
  }
  const catalog = await openCatalog({ env: io.env });
  const resolved = await catalog.resolve(id);
  const found = values.fixtures
    ? await readFixtures(values.fixtures)
    : await readFixtures(join(io.fixtures, `${resolved.id}.json`), { optional: true });
  const fixtures = found?.fixtures ?? null;
  const engine = resolved.engine;
  const runtime = io.runtime ?? createRuntime({ env: io.env, keepAlive: '0' });
  const overrides = engine.checkOverrides(fixtures);
  const entry = await runtime.acquire({
    ...resolved,
    engine: { ...engine, load: (o) => engine.load({ ...o, overrides }) },
  });
  const report = {
    id: resolved.id,
    variant: resolved.variant,
    executionProviders: entry.executionProviders,
    loadMs: entry.loadMs,
    ...(found ? { fixtures: found.path, reference: found.about } : {}),
    ...(await engine.check(entry.instance, { fixtures })),
  };
  await runtime.close();
  if (values.json) io.out.write(`${JSON.stringify(report, null, 2)}\n`);
  else {
    io.out.write(
      `${report.id}:${report.variant} on ${report.executionProviders.join(',')}: loaded in ${Math.round(report.loadMs)} ms, request ${Math.round(report.requestMs)} ms\n`
    );
    if (report.summary) io.out.write(`${report.summary}\n`);
    const made = report.reference?.variant;
    if (made && made !== report.variant)
      io.out.write(`the fixtures were made with ${made}; other variants can fall outside them\n`);
    if (!report.ok) io.out.write('check failed\n');
  }
  return report.ok ? 0 : 1;
}

export async function main(argv, overrides = {}) {
  const io = {
    out: process.stdout,
    err: process.stderr,
    env: process.env,
    fetch: globalThis.fetch,
    spawn: nodeSpawn,
    sleep,
    signals: process,
    binPath: new URL('./bin.js', import.meta.url).pathname,
    waitSteps: 300,
    fixtures: FIXTURES,
    ...overrides,
  };
  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      allowPositionals: true,
      options: {
        port: { type: 'string' },
        host: { type: 'string' },
        detach: { type: 'boolean' },
        json: { type: 'boolean' },
        unset: { type: 'string' },
        fixtures: { type: 'string' },
        help: { type: 'boolean', short: 'h' },
        version: { type: 'boolean', short: 'v' },
      },
    });
  } catch (error) {
    io.err.write(`${error.message}\n\n${USAGE}\n`);
    return 2;
  }
  const { values, positionals } = parsed;
  const [command, ...rest] = positionals;
  if (values.version) {
    io.out.write(`${VERSION}\n`);
    return 0;
  }
  if (values.help || !command) {
    io.out.write(`${USAGE}\n`);
    return command || values.help ? 0 : 2;
  }
  try {
    if (command === 'serve') return await serve(values, io);
    if (command === 'ls') return await ls(values, io);
    if (command === 'ps') return await ps(values, io);
    if (command === 'alias') return await alias(values, rest, io);
    if (command === 'check') return await check(values, rest, io);
  } catch (error) {
    io.err.write(`ortllama ${command}: ${error.message}\n`);
    return 1;
  }
  io.err.write(`unknown command ${command}\n\n${USAGE}\n`);
  return 2;
}

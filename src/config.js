import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

export const ROLES = [
  'chat',
  'chat-reasoning',
  'decisions',
  'transcribe',
  'realtime-transcribe',
  'speech',
];

export const DEFAULT_ALIASES = {
  decisions: 'ai-ecoverse/kev.js/kev-0.8b:q8f32',
  transcribe: 'onnx-community/whisper-tiny',
};

export const DEFAULT_PORT = 11435;

export function listenPort(port, env = process.env) {
  return port ?? Number(env.ORTLLAMA_PORT ?? DEFAULT_PORT);
}

export function ortllamaHome(env = process.env, home = homedir()) {
  return env.ORTLLAMA_HOME || join(home, '.ortllama');
}

export function parseDuration(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  if (typeof value === 'number') return value < 0 ? Number.POSITIVE_INFINITY : value * 1000;
  const match = /^(-?\d+(?:\.\d+)?)(ms|s|m|h)?$/.exec(String(value).trim());
  if (!match) throw new RangeError(`cannot read duration ${value}`);
  const n = Number(match[1]);
  if (n < 0) return Number.POSITIVE_INFINITY;
  const unit = { ms: 1, s: 1000, m: 60_000, h: 3_600_000 }[match[2] ?? 's'];
  return n * unit;
}

export async function readConfig(home) {
  try {
    const parsed = JSON.parse(await readFile(join(home, 'config.json'), 'utf8'));
    return { aliases: {}, hints: {}, ...parsed };
  } catch (error) {
    if (error.code === 'ENOENT') return { aliases: {}, hints: {} };
    throw error;
  }
}

export async function writeConfig(home, config) {
  const path = join(home, 'config.json');
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(config, null, 2)}\n`);
}

export function aliasesOf(config) {
  return { ...DEFAULT_ALIASES, ...config.aliases };
}

import { join } from 'node:path';
import { listDir, repoDir, resolveRevision } from '../hf/cache.js';
import { hubFiles } from '../hf/files.js';

const REPO = 'gradium/phonon-voices';
const LOCALE = /^([a-z]{2,3})-([a-z]{2})$/i;

export function voiceLanguage(locale, languages) {
  const match = LOCALE.exec(locale ?? '');
  if (match) return `${match[1].toLowerCase()}-${match[2].toUpperCase()}`;
  return languages?.[0] ?? 'en';
}

function parsed(line) {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

async function snapshots(cache) {
  const main = await resolveRevision(cache, REPO);
  const dir = join(repoDir(cache, REPO), 'snapshots');
  const shas = main ? [main.sha] : await listDir(dir);
  return shas.map((sha) => hubFiles(cache, REPO, join(dir, sha)));
}

export async function gradiumVoices(cache) {
  const out = new Map();
  if (!cache) return out;
  for (const files of await snapshots(cache)) {
    if (!(await files.exists('voices.jsonl'))) continue;
    for (const line of (await files.text('voices.jsonl')).split('\n')) {
      const voice = parsed(line);
      if (voice?.voice_id) out.set(voice.voice_id, voice);
    }
    break;
  }
  return out;
}

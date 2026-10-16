import { silero } from './silero.js';
import { systemone } from './systemone.js';
import { whisper } from './whisper.js';

export const ENGINES = [systemone, whisper, silero];

export async function detectEngine(files, engines = ENGINES) {
  for (const engine of engines) {
    const info = await engine.detect(files);
    if (info) return { engine, info };
  }
  return null;
}

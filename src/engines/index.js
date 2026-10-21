import { decoder } from './decoder/index.js';
import { jev } from './jev.js';
import { kokoro } from './kokoro.js';
import { parakeet } from './parakeet.js';
import { phonon } from './phonon.js';
import { silero } from './silero.js';
import { systemone } from './systemone.js';
import { whisper } from './whisper.js';

export const ENGINES = [systemone, jev, whisper, parakeet, silero, kokoro, phonon, decoder];

export async function detectEngine(files, engines = ENGINES) {
  for (const engine of engines) {
    const info = await engine.detect(files);
    if (info) return { engine, info };
  }
  return null;
}

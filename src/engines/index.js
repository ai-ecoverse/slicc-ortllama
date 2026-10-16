import { systemone } from './systemone.js';

export const ENGINES = [systemone];

export async function detectEngine(files, engines = ENGINES) {
  for (const engine of engines) {
    const info = await engine.detect(files);
    if (info) return { engine, info };
  }
  return null;
}

export function createRng(seed) {
  let state = Number.isInteger(seed) ? seed >>> 0 : (Math.random() * 2 ** 32) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
}

export function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

const FLOOR = 1e-7;

export function pick(logits, { temperature = 1, topP = 1, topK = 0, rng, banned = [] }) {
  for (const id of banned) logits[id] = Number.NEGATIVE_INFINITY;
  if (temperature <= 1e-5) return argmax(logits);
  const max = logits[argmax(logits)];
  const weights = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    const w = Math.exp((logits[i] - max) / temperature);
    weights[i] = w;
    total += w;
  }
  if (topP >= 1 && !(topK > 0)) {
    let r = rng() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return argmax(logits);
  }
  const candidates = [];
  for (let i = 0; i < weights.length; i++) if (weights[i] >= total * FLOOR) candidates.push(i);
  candidates.sort((a, b) => weights[b] - weights[a]);
  const top = topK > 0 ? candidates.slice(0, topK) : candidates;
  const mass = top.reduce((s, i) => s + weights[i], 0);
  const kept = [];
  let sum = 0;
  for (const i of top) {
    kept.push(i);
    sum += weights[i];
    if (sum >= topP * mass) break;
  }
  let r = rng() * sum;
  for (const i of kept) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return kept.at(-1);
}

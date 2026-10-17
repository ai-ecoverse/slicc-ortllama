import { logSumExp } from '../../whisper/model.js';
import { argmax } from './sample.js';

export function rank(logits, token) {
  const norm = logSumExp(logits);
  const top = argmax(logits);
  const best = logits[top];
  logits[top] = Number.NEGATIVE_INFINITY;
  const second = logits[argmax(logits)];
  logits[top] = best;
  return { token, logprob: logits[token] - norm, top, margin: best - second };
}

export function createScore({
  model,
  render,
  constraints,
  prefillAll,
  advance,
  eosIds,
  cache,
  vision,
}) {
  return async function score(request, tokens = [], length = tokens.length) {
    const ids = render(request);
    const prompt = { ids, keys: ids, spans: [] };
    const { banned } = constraints(request);
    const run = { past: model.emptyPast(), seq: [], vision: { images: 0, ms: 0 }, logits: null };
    const steps = [];
    try {
      await prefillAll(run, prompt, null, async () => {});
      for (let i = 0; i < length; i++) {
        for (const id of banned) run.logits[id] = Number.NEGATIVE_INFINITY;
        const token = tokens[i] ?? argmax(run.logits);
        if (i >= tokens.length && eosIds.has(token)) break;
        steps.push(rank(run.logits, token));
        if (i === length - 1) break;
        run.logits = (await advance(run, [token], true)).logits;
        run.seq.push(token);
      }
    } finally {
      cache.put(run.seq, run.past);
    }
    return steps;
  };
}

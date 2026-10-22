import { badRequest, HttpError } from '../http/errors.js';
import { roundProb, validate } from './decisions.js';
import { greedy, noImages, streamChat } from './gguf/chat.js';
import { byPreference, pickProjector, scanGguf, variantsOf } from './gguf/layout.js';
import { openWllama } from './gguf/load.js';
import { compareFixtures } from './systemone.js';

export const DEFAULT_CONTEXT = 8192;
export const CHECK_TOKENS = 24;
export const CHECK_TOP = 5;
export const CLOSE_MARGIN = 1;
export const MEAN_LOGPROB = 0.15;

function contextFor(train, hint) {
  const asked =
    Number.isInteger(hint?.contextLength) && hint.contextLength > 0
      ? hint.contextLength
      : DEFAULT_CONTEXT;
  return train ? Math.min(train, asked) : asked;
}

async function detectWith(files, decision) {
  const { models, projectors } = await scanGguf(files);
  const own = models.filter((m) => m.head.decision === decision);
  if (own.length === 0) return null;
  const parts = variantsOf(own);
  const variants = byPreference(Object.keys(parts));
  const mmproj = decision ? null : pickProjector(projectors);
  return {
    gguf: true,
    parts,
    variants,
    defaultVariant: variants[0],
    mmproj,
    inputModalities: mmproj ? ['text', 'image'] : ['text'],
    outputModalities: [decision ? 'decisions' : 'text'],
    contextLength: contextFor(parts[variants[0]].contextLength, null),
  };
}

async function missingParts(files, info, variant) {
  const out = [];
  for (const rel of info.parts[variant].files) if (!(await files.exists(rel))) out.push(rel);
  return out;
}

function specialize(_files, info, variant) {
  const contextLength = contextFor(info.parts[variant]?.contextLength, info.hint);
  const details = {
    architecture: info.parts[variant]?.architecture,
    ...(info.mmproj ? { mmproj: info.mmproj } : {}),
  };
  return { info: { ...info, contextLength }, key: `ctx=${contextLength}`, details };
}

function rawProbs(answer) {
  if (answer.type === 'noul') return [1 - answer.noul, answer.noul];
  return Object.values(answer.probabilities);
}

export function roundAnswers(answers) {
  const out = {};
  for (const [id, a] of Object.entries(answers)) {
    const probabilities = a.probabilities
      ? Object.fromEntries(Object.entries(a.probabilities).map(([k, v]) => [k, roundProb(v)]))
      : undefined;
    out[id] = {
      ...a,
      ...(a.type === 'noul' ? { noul: roundProb(a.noul) } : {}),
      ...(a.type === 'score' ? { score: roundProb(a.score) } : {}),
      ...(probabilities ? { probabilities } : {}),
      ...(typeof a.confidence === 'number' ? { confidence: roundProb(a.confidence) } : {}),
    };
  }
  return out;
}

export function compareGreedy(results, fixtures, { close = CLOSE_MARGIN } = {}) {
  let tokens = 0;
  let sum = 0;
  let max = 0;
  let worstAt = '';
  const flips = [];
  results.forEach((steps, i) => {
    const f = fixtures[i];
    for (let j = 0; j < f.tokens.length; j++) {
      const at = `${f.name}#${j}`;
      const step = steps[j];
      if (!step || step.token !== f.tokens[j]) {
        if (f.margins[j] === null || f.margins[j] > close) flips.push({ at, margin: f.margins[j] });
        break;
      }
      const d = Math.abs(step.logprob - f.logprobs[j]);
      tokens++;
      sum += d;
      if (d > max) {
        max = d;
        worstAt = at;
      }
      if (f.margins[j] !== null && f.margins[j] <= close) break;
    }
  });
  const mean = tokens ? sum / tokens : 0;
  return { tokens, mean, max, worstAt, flips, ok: mean <= MEAN_LOGPROB && flips.length === 0 };
}

const SMOKE_CHAT = {
  messages: [{ role: 'user', content: 'Say hello in five words.' }],
  tools: [],
  stop: [],
  maxTokens: 24,
  temperature: 0,
};

const SMOKE_DECISION = {
  state: 'Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.',
  questions: {
    department: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: {
        returns: 'Exchanges and refunds',
        shipping: 'Delivery and delays',
        billing: 'Charges and payments',
      },
    },
    escalate: { type: 'noul', instructions: 'Does this need urgent human attention?' },
  },
};

async function open({ backend, files, variant, info, overrides = {}, library, gpus }) {
  const part = info.parts[variant];
  const rels = [...part.files, ...(info.mmproj ? [info.mmproj] : [])];
  return openWllama({
    backend,
    files,
    rels,
    contextLength: overrides.contextLength ?? info.contextLength,
    options: overrides.options,
    library,
    gpus,
  });
}

export const gguf = {
  name: 'gguf',
  capability: 'chat',
  detect: (files) => detectWith(files, false),
  missing: missingParts,
  include(info, variant) {
    return [...info.parts[variant].files, info.mmproj ?? '*mmproj*.gguf'];
  },
  specialize,
  async load(options) {
    const { wllama, release } = await open(options);
    const vision = Boolean(options.info.mmproj);
    return {
      exclusive: true,
      async chat(request, context) {
        if (!vision) noImages(request);
        return streamChat(wllama, request, context);
      },
      greedy: (request, tokens = CHECK_TOKENS, top = CHECK_TOP) =>
        greedy(wllama, request, { tokens, top }),
      release,
    };
  },
  checkOverrides: () => ({}),
  async check(instance, { fixtures, now = () => performance.now() } = {}) {
    const started = now();
    let answer = '';
    const result = await instance.chat(SMOKE_CHAT, {
      emit(event) {
        if (event.type === 'content' || event.type === 'reasoning') answer += event.text;
      },
    });
    const smoke = { ok: answer.length > 0, requestMs: now() - started, answer, ...result };
    if (!fixtures) return smoke;
    const results = [];
    for (const f of fixtures)
      results.push(await instance.greedy(fixtureChat(f), f.tokens.length, CHECK_TOP));
    const compared = compareGreedy(results, fixtures);
    return {
      ...smoke,
      ...compared,
      summary: `fixtures: ${compared.tokens} greedy tokens up to the first close margin, mean |dlogp| ${compared.mean.toFixed(3)}, max ${compared.max.toFixed(3)}${compared.worstAt ? ` (${compared.worstAt})` : ''}, ${compared.flips.length} changed`,
      ok: smoke.ok && compared.ok,
    };
  },
};

export function fixtureChat(f) {
  return { messages: f.messages, tools: [], toolChoice: 'auto', stop: [] };
}

export const ggufDecisions = {
  name: 'gguf-decisions',
  capability: 'systemone',
  detect: (files) => detectWith(files, true),
  missing: missingParts,
  include: (info, variant) => [...info.parts[variant].files],
  specialize,
  async load(options) {
    const { wllama, release } = await open(options);
    const ask = async (request) => {
      try {
        return await wllama.createSystemOne({ state: request.state, questions: request.questions });
      } catch (error) {
        if (error?.type === 'inference_error')
          throw new HttpError(400, error.message, { param: 'questions' });
        throw error;
      }
    };
    return {
      async systemOne(request) {
        validate(request);
        if (request.image !== undefined)
          throw badRequest('this System One model takes no images', 'image');
        const started = performance.now();
        const result = await ask(request);
        return {
          answers: roundAnswers(result.answers),
          usage: result.usage,
          latency_ms: Math.round((performance.now() - started) * 10) / 10,
        };
      },
      async probs(request) {
        const { answers } = await ask(request);
        return Object.keys(request.questions).map((id) => rawProbs(answers[id]));
      },
      release,
    };
  },
  checkOverrides: () => ({}),
  async check(instance, { fixtures, now = () => performance.now() } = {}) {
    const started = now();
    const answer = await instance.systemOne(SMOKE_DECISION);
    const requestMs = now() - started;
    if (!fixtures) return { ok: true, requestMs, answer };
    const results = [];
    for (const f of fixtures) results.push(await instance.probs(f.request));
    const compared = compareFixtures(results, fixtures);
    return {
      requestMs,
      answer,
      ...compared,
      summary: `fixtures: ${compared.questions} questions, mean |dp| ${compared.mean.toFixed(4)}, max ${compared.max.toFixed(4)} (${compared.worstAt}), ${compared.flips.length} changed`,
    };
  },
};

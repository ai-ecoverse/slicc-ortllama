import { badRequest } from '../http/errors.js';

const isObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v);

function optionCount(q) {
  if (q.type === 'noul') return 2;
  return q.type === 'choice' ? Object.keys(q.criteria).length : q.criteria.length;
}

const CRITERIA = {
  noul: (c) => c == null || isObject(c),
  choice: isObject,
  score: Array.isArray,
};

function validateQuestion(param, q, min, max) {
  if (!isObject(q)) throw badRequest(`${param} must be an object`, param);
  const fits = CRITERIA[q.type];
  if (!fits) throw badRequest(`${param}.type must be noul, choice or score`, param);
  if (!fits(q.criteria))
    throw badRequest(
      `${param}.criteria must be ${q.type === 'score' ? 'a list' : 'an object'}`,
      param
    );
  const n = optionCount(q);
  if (n < min || n > max)
    throw badRequest(`${param} must have ${min} to ${max} options for this model`, param);
}

export function validate(request, { min = 1, max = 255 } = {}) {
  if (!isObject(request)) throw badRequest('request must be an object');
  if (!('state' in request)) throw badRequest('state is required', 'state');
  if (!isObject(request.questions) || Object.keys(request.questions).length < 1)
    throw badRequest('questions must have at least 1 entry', 'questions');
  for (const [id, q] of Object.entries(request.questions))
    validateQuestion(`questions.${id}`, q, min, max);
  return request;
}

export function render(v, indent = 0) {
  const pad = '  '.repeat(indent);
  if (v === null || v === undefined) return '';
  if (typeof v !== 'object') return String(v);
  if (Array.isArray(v))
    return v.map((x) => `${pad}- ${render(x, indent + 1).trimStart()}`).join('\n');
  return Object.entries(v)
    .map(([k, x]) =>
      typeof x === 'object' && x !== null
        ? `${pad}${k}:\n${render(x, indent + 1)}`
        : `${pad}${k}: ${render(x)}`
    )
    .join('\n');
}

const optionText = (name, desc) =>
  desc === null || desc === undefined || desc === '' ? name : `${name}: ${render(desc)}`;

export function toRecord(request) {
  const questions = [];
  const meta = [];
  for (const [id, q] of Object.entries(request.questions)) {
    let options;
    if (q.type === 'noul') {
      const c = q.criteria ?? {};
      options = [optionText('no', c.false), optionText('yes', c.true)];
      meta.push({ id, type: 'noul' });
    } else if (q.type === 'choice') {
      options = Object.entries(q.criteria).map(([k, v]) => optionText(k, v));
      meta.push({ id, type: 'choice', keys: Object.keys(q.criteria) });
    } else {
      options = q.criteria.map((x) => render(x));
      meta.push({ id, type: 'score', legend: Object.fromEntries(options.map((o, i) => [i, o])) });
    }
    questions.push({ instr: render(q.instructions), options });
  }
  return { record: { state: render(request.state), questions }, meta };
}

const argmax = (p) => p.reduce((best, v, i) => (v > p[best] ? i : best), 0);

function normalized(p) {
  const total = p.reduce((s, x) => s + x, 0);
  return total === 0 ? p.map(() => 1 / p.length) : p.map((x) => x / total);
}

export function choiceConfidence(p) {
  const k = p.length;
  return k === 1 ? 1 : (Math.max(...normalized(p)) - 1 / k) / (1 - 1 / k);
}

export function scoreConfidence(p) {
  const levels = p.length;
  if (levels === 1) return 1;
  const n = normalized(p);
  const mode = argmax(n);
  const spread =
    Array.from({ length: levels }, (_, i) => Math.abs(i - (levels - 1) / 2)).reduce(
      (s, x) => s + x,
      0
    ) / levels;
  return Math.max(0, 1 - n.reduce((s, pi, i) => s + pi * Math.abs(i - mode), 0) / spread);
}

export const roundProb = (x) => Number(x.toFixed(4));

export function toAnswers(probs, meta) {
  const out = {};
  probs.forEach((p, i) => {
    const m = meta[i];
    if (m.type === 'noul') out[m.id] = { type: 'noul', noul: roundProb(p[1]) };
    else if (m.type === 'choice')
      out[m.id] = {
        type: 'choice',
        choice: m.keys[argmax(p)],
        confidence: roundProb(choiceConfidence(p)),
        probabilities: Object.fromEntries(m.keys.map((k, j) => [k, roundProb(p[j])])),
      };
    else
      out[m.id] = {
        type: 'score',
        score: roundProb(p.reduce((s, pi, j) => s + j * pi, 0)),
        legend: m.legend,
        probabilities: Object.fromEntries(p.map((v, j) => [String(j), roundProb(v)])),
        confidence: roundProb(scoreConfidence(p)),
      };
  });
  return out;
}

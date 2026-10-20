import { ENCODERS } from './audio/encode.js';
import { badRequest } from './http/errors.js';
import { splitLong, splitSentences } from './kokoro/text.js';
import { PRIORITY } from './runtime.js';

const MAX_INPUT = 4096;
const FLUSH_MS = 250;

function optionalString(body, key) {
  if (body[key] !== undefined && typeof body[key] !== 'string')
    throw badRequest(`${key} must be a string`, key);
  return body[key];
}

export function speechRequest(body) {
  if (typeof body !== 'object' || body === null || Array.isArray(body))
    throw badRequest('request body must be a JSON object');
  const { input, response_format: format = 'wav', speed = 1 } = body;
  if (typeof input !== 'string' || input.trim() === '')
    throw badRequest('input must be a non-empty string', 'input');
  if (input.length > MAX_INPUT)
    throw badRequest(`input must be at most ${MAX_INPUT} characters`, 'input');
  if (typeof format !== 'string' || !Object.hasOwn(ENCODERS, format))
    throw badRequest(
      `response_format ${format} is not supported; use one of ${Object.keys(ENCODERS).join(', ')}`,
      'response_format'
    );
  if (typeof speed !== 'number' || !(speed >= 0.25 && speed <= 4))
    throw badRequest('speed must be a number from 0.25 to 4', 'speed');
  optionalString(body, 'instructions');
  return {
    model: optionalString(body, 'model') ?? 'speech',
    input,
    voice: optionalString(body, 'voice'),
    format,
    speed,
  };
}

function ignored(promise) {
  promise.catch(() => {});
  return promise;
}

function flushGate(ms) {
  let passed = 0;
  let wake = null;
  return {
    pass(n = passed + 1) {
      passed = n;
      wake?.();
    },
    async wait(k) {
      if (passed < k) {
        let timer;
        await new Promise((resolve) => {
          wake = resolve;
          timer = setTimeout(resolve, ms);
        });
        wake = null;
        clearTimeout(timer);
      }
      await new Promise((resolve) => setImmediate(resolve));
    },
  };
}

function schedule({ resolved, runtime, request, signal, gate }) {
  const job = (fn) => ignored(runtime.use(resolved, PRIORITY.speech, fn, signal));
  const { voice, speed } = request;
  const sentences = splitSentences(request.input).flatMap((s) => splitLong(s));
  let count = 0;
  const synthesis = (ids) => {
    const k = count++;
    return job(async (model) => {
      await gate.wait(k);
      return model.synthesize(ids, { voice, speed });
    });
  };
  const plan = (i) =>
    job(async (tts) => {
      const chunks = await tts.tokens(sentences[i], { voice });
      const audio = chunks.map(synthesis);
      return { rate: tts.sampleRate, audio, next: i + 1 < sentences.length ? plan(i + 1) : null };
    });
  return plan(0);
}

async function* synthesize(context) {
  const gate = flushGate(FLUSH_MS);
  let planned = schedule({ ...context, gate });
  let encoder = null;
  try {
    while (planned) {
      const sentence = await planned;
      encoder ??= await context.open(sentence.rate);
      for (const pending of sentence.audio) {
        const bytes = encoder.push(await pending);
        if (bytes.length) yield bytes;
        gate.pass();
      }
      planned = sentence.next;
    }
    const tail = encoder.end();
    if (tail.length) yield tail;
  } finally {
    gate.pass(Number.POSITIVE_INFINITY);
    encoder?.close?.();
  }
}

export function speechRoute({ catalog, runtime }) {
  return {
    async handle({ body, signal }) {
      const request = speechRequest(body);
      const resolved = await catalog.resolve(request.model);
      if (resolved.engine.capability !== 'speech')
        throw badRequest(`model ${resolved.id} does not synthesize speech`, 'model');
      const format = ENCODERS[request.format];
      const open = await format.open();
      return {
        headers: { 'content-type': format.type },
        stream: synthesize({ resolved, runtime, request, signal, open }),
      };
    },
  };
}

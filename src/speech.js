import { encodePcm16, wavHeader } from './audio/pcm.js';
import { badRequest } from './http/errors.js';
import { splitLong, splitSentences } from './kokoro/text.js';
import { PRIORITY } from './runtime.js';

const FORMATS = { pcm: 'audio/pcm', wav: 'audio/wav' };
const MAX_INPUT = 4096;

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
  if (typeof format !== 'string' || !Object.hasOwn(FORMATS, format))
    throw badRequest(
      `response_format ${format} is not supported; use one of ${Object.keys(FORMATS).join(', ')}`,
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

function schedule({ resolved, runtime, request, signal }) {
  const job = (fn) => ignored(runtime.use(resolved, PRIORITY.speech, fn, signal));
  const { voice, speed } = request;
  const sentences = splitSentences(request.input).flatMap((s) => splitLong(s));
  const plan = (i) =>
    job(async (tts) => {
      const chunks = await tts.tokens(sentences[i], { voice });
      const audio = chunks.map((ids) =>
        job(async (model) => encodePcm16(await model.synthesize(ids, { voice, speed })))
      );
      return { rate: tts.sampleRate, audio, next: i + 1 < sentences.length ? plan(i + 1) : null };
    });
  return plan(0);
}

async function* synthesize(context) {
  let header = context.request.format === 'wav';
  let planned = schedule(context);
  let rate;
  while (planned) {
    const sentence = await planned;
    rate = sentence.rate;
    for (const pending of sentence.audio) {
      const pcm = await pending;
      if (!header) yield pcm;
      else {
        header = false;
        const out = new Uint8Array(44 + pcm.length);
        out.set(wavHeader(rate));
        out.set(pcm, 44);
        yield out;
      }
    }
    planned = sentence.next;
  }
  if (header) yield wavHeader(rate);
}

export function speechRoute({ catalog, runtime }) {
  return {
    async handle({ body, signal }) {
      const request = speechRequest(body);
      const resolved = await catalog.resolve(request.model);
      if (resolved.engine.capability !== 'speech')
        throw badRequest(`model ${resolved.id} does not synthesize speech`, 'model');
      return {
        headers: { 'content-type': FORMATS[request.format] },
        stream: synthesize({ resolved, runtime, request, signal }),
      };
    },
  };
}

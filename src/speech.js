import { ENCODERS } from './audio/encode.js';
import { AudioFormatError } from './audio/pcm.js';
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

function flushGate() {
  const opens = [];
  const waits = [];
  let passed = 0;
  return {
    passed(k) {
      if (passed >= k) return null;
      waits[k] ??= new Promise((resolve) => {
        opens[k] = resolve;
      });
      return waits[k];
    },
    pass(n = passed + 1) {
      passed = n;
      for (const [k, open] of opens.entries()) if (open && k <= n) open();
    },
  };
}

async function within(promise, ms) {
  let timer;
  await Promise.race([promise, new Promise((resolve) => (timer = setTimeout(resolve, ms)))]);
  clearTimeout(timer);
}

const immediate = () => new Promise((resolve) => setImmediate(resolve));

function channel() {
  const items = [];
  let wake = null;
  let ended = false;
  let failure = null;
  const notify = () => {
    wake?.();
    wake = null;
  };
  return {
    push(item) {
      items.push(item);
      notify();
    },
    end(error = null) {
      ended = true;
      failure = error;
      notify();
    },
    async *[Symbol.asyncIterator]() {
      for (;;) {
        if (items.length) yield items.shift();
        else if (failure) throw failure;
        else if (ended) return;
        else await new Promise((resolve) => (wake = resolve));
      }
    },
  };
}

function schedule({ resolved, runtime, request, signal, gate }) {
  const job = (fn, ready) => ignored(runtime.use(resolved, PRIORITY.speech, fn, signal, ready));
  const { voice, speed } = request;
  const sentences = resolved.engine.splits
    ? [request.input]
    : splitSentences(request.input).flatMap((s) => splitLong(s));
  let produced = 0;
  let previous = Promise.resolve();
  const flushed = () => within(gate.passed(produced), FLUSH_MS).then(immediate);
  const pieces = (model, ids) =>
    model.stream
      ? model.stream(ids, { voice, speed, signal })
      : [model.synthesize(ids, { voice, speed })];
  const synthesis = (ids) => {
    const out = channel();
    const done = job(async (model, _entry, pause) => {
      for await (const audio of pieces(model, ids)) {
        out.push(audio);
        produced++;
        await pause(flushed());
      }
    }, previous.then(flushed));
    done.then(
      () => out.end(),
      (error) => out.end(error)
    );
    previous = done.catch(() => {});
    return out;
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
  const gate = flushGate();
  let planned = schedule({ ...context, gate });
  let encoder = null;
  try {
    while (planned) {
      const sentence = await planned;
      encoder ??= await context.open(sentence.rate);
      for (const pieces of sentence.audio)
        for await (const audio of pieces) {
          const bytes = await encoder.push(audio);
          if (bytes.length) yield bytes;
          gate.pass();
        }
      planned = sentence.next;
    }
    const tail = await encoder.end();
    if (tail.length) yield tail;
  } finally {
    gate.pass(Number.POSITIVE_INFINITY);
    encoder?.close?.();
  }
}

async function opener(format, aac) {
  try {
    return await format.open({ aac });
  } catch (error) {
    if (!(error instanceof AudioFormatError)) throw error;
    throw badRequest(error.message, 'response_format');
  }
}

export function speechRoute({ catalog, runtime, aac }) {
  return {
    async handle({ body, signal }) {
      const request = speechRequest(body);
      const resolved = await catalog.resolve(request.model);
      if (resolved.engine.capability !== 'speech')
        throw badRequest(`model ${resolved.id} does not synthesize speech`, 'model');
      const format = ENCODERS[request.format];
      const open = await opener(format, aac);
      return {
        headers: { 'content-type': format.type },
        stream: synthesize({ resolved, runtime, request, signal, open }),
      };
    },
  };
}

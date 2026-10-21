import { badRequest } from '../http/errors.js';
import { contextError } from './decoder/chat.js';
import { createRng } from './decoder/sample.js';
import { stopFilter } from './decoder/stream.js';

export const BASE_MODE = 'base';

const LABELS = { system: 'System', developer: 'System', user: 'User', assistant: 'Assistant' };
const TURNS = ['\n\nUser:', '\nUser:', '\nSystem:', '\nAssistant:'];
const TOP_K = 40;

class Stopped extends Error {}

export function gatedOrt(ort) {
  const gate = { before: null };
  const wrap = (session) =>
    Object.assign(Object.create(session), {
      async run(...args) {
        await gate.before?.();
        return session.run(...args);
      },
    });
  return {
    gate,
    ort: {
      ...ort,
      InferenceSession: {
        create: async (...args) => wrap(await ort.InferenceSession.create(...args)),
      },
    },
  };
}

function refuseTools(request) {
  if (request.tools.length > 0)
    throw badRequest('this base model takes no tools; use an instruct model', 'tools');
  for (const m of request.messages)
    if (m.role === 'tool' || m.tool_calls)
      throw badRequest('this base model takes no tool calls; use an instruct model', 'messages');
}

export function imageTokenCount(image, { pixtralSize, PIXTRAL }) {
  const unit = PIXTRAL.patch * PIXTRAL.merge;
  const [h, w] = pixtralSize(image.height, image.width);
  const rows = Math.ceil(h / unit);
  const cols = Math.ceil(w / unit);
  if (rows === 0 || cols === 0)
    throw badRequest(
      `an image of ${image.width}x${image.height} is too thin for this model's vision encoder`,
      'messages'
    );
  return rows * (cols + 1);
}

export function basePrompt(kev, request, kevjs) {
  refuseTools(request);
  const tokenizer = kev.tokenizer;
  const img = tokenizer.token_to_id('[IMG]');
  const ids = [tokenizer.token_to_id('<s>')];
  let pending = '';
  const flush = () => {
    ids.push(...kev.userTokens(pending));
    pending = '';
  };
  let image;
  for (const m of request.messages) {
    pending += `${LABELS[m.role]}: `;
    const parts =
      typeof m.content === 'string' ? [{ type: 'text', text: m.content }] : (m.content ?? []);
    parts.forEach((part, i) => {
      if (i > 0) pending += '\n';
      if (part.type === 'text') {
        pending += part.text;
        return;
      }
      if (image) throw badRequest('this base model takes one image per request', 'messages');
      image = part.image;
      flush();
      ids.push(img);
    });
    pending += '\n';
  }
  pending += 'Assistant:';
  flush();
  const inputTokens = ids.length + (image ? imageTokenCount(image, kevjs) - 1 : 0);
  return { ids, image, inputTokens };
}

function settingsOf(request, budget) {
  return {
    temperature: request.temperature ?? 0,
    top_p: 1,
    top_k: request.topK > 0 ? request.topK : TOP_K,
    max_tokens: budget,
  };
}

export function baseChat({ kev, gate, kevjs, contextLength, now = () => performance.now() }) {
  return async function chat(request, { signal, emit, yieldTurn = async () => {} }) {
    const { ids, image, inputTokens } = basePrompt(kev, request, kevjs);
    if (inputTokens >= contextLength) throw contextError(inputTokens, contextLength);
    const budget = Math.min(
      request.maxTokens ?? Number.POSITIVE_INFINITY,
      contextLength - inputTokens
    );
    const settings = settingsOf(request, budget);
    if (signal?.aborted) throw signal.reason;
    emit({ type: 'start', promptTokens: inputTokens });
    const state = { pending: true, stopped: false, started: false, count: 0, first: null };
    const stops = stopFilter([...TURNS, ...request.stop], (text) => {
      const lead = state.started ? text : text.trimStart();
      if (lead === '') return;
      state.started = true;
      emit({ type: 'content', text: lead });
    });
    const push = (text) => {
      if (text === '' || state.stopped) return;
      state.stopped = stops.push(text) || state.stopped;
    };
    let shown = '';
    const started = now();
    gate.before = async () => {
      if (signal?.aborted) throw signal.reason;
      if (!state.pending) return;
      state.pending = false;
      if (state.stopped) throw new Stopped();
      await yieldTurn();
      if (signal?.aborted) throw signal.reason;
    };
    let finish;
    try {
      const out = await kev.generate(ids, {
        maxTokens: budget,
        image,
        temperature: settings.temperature,
        topK: settings.top_k,
        random: createRng(request.seed),
        onToken(text, count) {
          state.first ??= now();
          state.count = count;
          state.pending = true;
          const ready = text.endsWith('\uFFFD') ? text.slice(0, text.lastIndexOf('\uFFFD')) : text;
          if (!ready.startsWith(shown) || ready.length <= shown.length) return;
          push(ready.slice(shown.length));
          shown = ready;
        },
      });
      if (out.text.startsWith(shown)) push(out.text.slice(shown.length));
      finish = state.stopped ? 'stop' : out.finish;
    } catch (error) {
      if (image && error instanceof RangeError)
        throw badRequest(
          `the image does not fit this model's vision encoder: ${error.message}`,
          'messages'
        );
      if (!(error instanceof Stopped)) throw error;
      finish = 'stop';
    } finally {
      gate.before = null;
    }
    stops.end();
    const end = now();
    const generated = state.count;
    const first = state.first ?? end;
    return {
      finishReason: finish,
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: generated,
        total_tokens: inputTokens + generated,
      },
      timings: {
        prompt_n: inputTokens,
        prompt_ms: first - started,
        prompt_per_second: (inputTokens * 1000) / (first - started),
        predicted_n: generated,
        predicted_ms: end - first,
        predicted_per_second: (generated * 1000) / (end - first),
      },
      settings,
    };
  };
}

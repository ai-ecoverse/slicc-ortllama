import { Template } from '@huggingface/jinja';
import { badRequest, HttpError } from '../../http/errors.js';
import { createPrefixCache } from './prefix.js';
import { checkTools, DIRECT_ANSWER, escaper } from './profiles.js';
import { createRng, pick } from './sample.js';
import { createParser, stopFilter } from './stream.js';
import { expandImages, requestImages, splicer } from './vision.js';

export const PREFILL_CHUNK = 256;

export const CACHE_BYTES = 2 ** 30;

export function contextError(tokens, limit) {
  return new HttpError(
    400,
    `the prompt has ${tokens} tokens, which exceeds the context window of ${limit} tokens; shorten the conversation`,
    { code: 'context_length_exceeded', param: 'messages' }
  );
}

function summary(ids, run, times, vision) {
  const prefillMs = times.decode - times.start;
  const decodeMs = times.end - times.decode;
  const fresh = ids.length - run.cached;
  return {
    usage: {
      prompt_tokens: ids.length,
      completion_tokens: run.generated,
      total_tokens: ids.length + run.generated,
      prompt_tokens_details: { cached_tokens: run.cached },
      completion_tokens_details: { reasoning_tokens: run.reasoning },
    },
    timings: {
      cache_n: run.cached,
      prompt_n: fresh,
      prompt_ms: prefillMs,
      prompt_per_second: (fresh * 1000) / prefillMs,
      predicted_n: run.generated,
      predicted_ms: decodeMs,
      predicted_per_second: (run.generated * 1000) / decodeMs,
      ...(vision.images > 0 ? { image_n: vision.images, image_ms: vision.ms } : {}),
    },
  };
}

const INSTRUCTIONS = new Set(['system', 'developer']);

export function directMessages(request, thinks) {
  if (request.reasoningEffort !== 'none' || !thinks || INSTRUCTIONS.has(request.messages[0]?.role))
    return request.messages;
  return [{ role: 'system', content: DIRECT_ANSWER }, ...request.messages];
}

function samplingFor(request, defaults, banned) {
  return {
    temperature: request.temperature ?? defaults.temperature ?? 1,
    topP: request.topP ?? defaults.topP ?? 1,
    topK: request.topK ?? defaults.topK ?? 0,
    rng: createRng(request.seed),
    banned,
  };
}

function settingsOf(sampling, budget) {
  return {
    temperature: sampling.temperature,
    top_p: sampling.topP,
    top_k: sampling.topK,
    max_tokens: budget,
  };
}

export function createChat({
  model,
  tokenizer,
  tokenizerJson,
  bosToken = '',
  eosToken = '',
  template,
  profile,
  contextLength,
  eos = [],
  defaults = {},
  prefill = PREFILL_CHUNK,
  vision = null,
  now = () => performance.now(),
}) {
  const added = tokenizerJson.added_tokens;
  const specials = added.filter((t) => t.special);
  const special = new Set(specials.map((t) => t.id));
  const esc = escaper(specials.map((t) => t.content));
  const idOf = (name) => tokenizer.token_to_id(name) ?? null;
  const markers = Object.fromEntries(Object.entries(profile.markers).map(([k, v]) => [k, idOf(v)]));
  const eosIds = new Set([...eos, idOf(eosToken)].filter((x) => x !== null));
  const compiled = new Template(profile.template(template));
  const thinks = profile.thinksByDefault(template);
  const cache = createPrefixCache({
    model,
    maxEntries: 2,
    maxTokens: Math.floor(CACHE_BYTES / model.bytesPerToken),
  });
  const decode = (ids) => tokenizer.decode(ids, { skip_special_tokens: false });

  function render(request) {
    const messages = profile.messages(directMessages(request, thinks), esc);
    const tools = request.tools.length > 0 ? esc.deep(request.tools) : undefined;
    let text;
    try {
      text = compiled.render({
        messages,
        tools,
        bos_token: bosToken,
        eos_token: eosToken,
        add_generation_prompt: true,
      });
    } catch (error) {
      throw badRequest(`the chat template rejected the messages: ${error.message}`, 'messages');
    }
    return tokenizer.encode(text, { add_special_tokens: false }).ids;
  }

  function constraints(request) {
    const allowed = new Set(eosIds);
    const toolsOn = request.tools.length > 0 && request.toolChoice !== 'none';
    if (toolsOn) for (const m of [markers.toolCall, markers.args]) allowed.add(m);
    if (request.reasoningEffort !== 'none')
      for (const m of [markers.think, markers.thinkEnd]) allowed.add(m);
    const banned = [...special].filter((id) => !allowed.has(id));
    let forced = [];
    if (toolsOn && request.toolChoice === 'required') forced = [markers.toolCall];
    if (toolsOn && typeof request.toolChoice === 'object') {
      const name = request.toolChoice.function.name;
      if (!request.tools.some((t) => t.function.name === name))
        throw badRequest(`tool_choice names ${name}, which is not in tools`, 'tool_choice');
      forced = [
        markers.toolCall,
        ...tokenizer.encode(name, { add_special_tokens: false }).ids,
        markers.args,
      ];
    }
    return { banned, forced };
  }

  function output(request, emit) {
    const state = { toolCalls: 0, stopped: false };
    const stops = stopFilter(request.stop, (text) => emit({ type: 'content', text }));
    const parser = createParser({
      decode,
      markers,
      special,
      emit(event) {
        if (event.type === 'content') state.stopped = stops.push(event.text) || state.stopped;
        else emit(event);
        if (event.type === 'tool') state.toolCalls++;
      },
    });
    state.parser = parser;
    state.end = () => {
      parser.end();
      stops.end();
    };
    return state;
  }

  async function advance(run, ids, wantLogits, splice = null) {
    const past = run.past;
    run.past = null;
    const out = await model.forward(ids, past, run.seq.length, wantLogits, splice);
    run.past = out.past;
    return out;
  }

  async function prefillAll(run, prompt, signal, yieldTurn) {
    const { ids, keys, spans } = prompt;
    while (run.seq.length < ids.length) {
      if (signal?.aborted) throw signal.reason;
      const from = run.seq.length;
      const chunk = ids.slice(from, from + prefill);
      const to = from + chunk.length;
      const splice = await splicer(spans, from, to, vision, run.vision);
      const out = await advance(run, chunk, to === ids.length, splice);
      run.logits = out.logits;
      run.seq.push(...keys.slice(from, to));
      await yieldTurn();
    }
  }

  async function decodeAll(run, { budget, forced, sampling, out, signal, yieldTurn }) {
    for (let n = 0; n < budget; n++) {
      const token = n < forced.length ? forced[n] : pick(run.logits, sampling);
      run.generated++;
      if (eosIds.has(token)) return 'stop';
      if (out.parser.state === 'think') run.reasoning++;
      out.parser.push(token);
      if (out.stopped) return 'stop';
      if (n === budget - 1) break;
      await yieldTurn();
      if (signal?.aborted) throw signal.reason;
      const step = await advance(run, [token], true);
      run.logits = step.logits;
      run.seq.push(token);
    }
    return 'length';
  }

  async function chat(request, { signal, emit, yieldTurn = async () => {} }) {
    checkTools(profile, request.tools);
    const images = requestImages(request.messages);
    if (images.length > 0 && !vision)
      throw badRequest(
        'this model takes no images: its vision encoder is not in the cache',
        'messages'
      );
    const prompt = expandImages(render(request), images, vision);
    const { ids } = prompt;
    if (ids.length >= contextLength) throw contextError(ids.length, contextLength);
    const { banned, forced } = constraints(request);
    const budget = Math.min(
      request.maxTokens ?? Number.POSITIVE_INFINITY,
      contextLength - ids.length
    );
    const sampling = samplingFor(request, defaults, banned);
    if (signal?.aborted) throw signal.reason;
    emit({ type: 'start', promptTokens: ids.length });
    const out = output(request, emit);
    const times = { start: now() };
    const taken = await cache.take(prompt.keys);
    const run = {
      past: taken.past,
      cached: taken.length,
      seq: prompt.keys.slice(0, taken.length),
      vision: { images: 0, ms: 0 },
      logits: null,
      generated: 0,
      reasoning: 0,
    };
    let finish;
    try {
      await prefillAll(run, prompt, signal, yieldTurn);
      times.decode = now();
      finish = await decodeAll(run, { budget, forced, sampling, out, signal, yieldTurn });
      out.end();
    } finally {
      if (run.past) cache.put(run.seq, run.past);
    }
    times.end = now();
    if (out.toolCalls > 0 && finish === 'stop' && !out.stopped) finish = 'tool_calls';
    return {
      finishReason: finish,
      ...summary(ids, run, times, run.vision),
      settings: settingsOf(sampling, budget),
    };
  }

  return { chat, render, cache };
}

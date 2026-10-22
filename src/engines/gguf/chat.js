import { badRequest, HttpError } from '../../http/errors.js';
import { toolCallId } from '../decoder/stream.js';

const CONTEXT = /context (size|length|window)|exceeds the available context/i;

async function pngOf(image) {
  const { encode } = await import('fast-png');
  const n = image.width * image.height;
  const rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) rgb[i * 3 + c] = image.data[i * 4 + c];
  const png = encode({
    width: image.width,
    height: image.height,
    data: rgb,
    channels: 3,
    depth: 8,
  });
  return png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength);
}

const textOf = (content) =>
  Array.isArray(content) ? content.map((p) => p.text ?? '').join('') : content;

async function userContent(content) {
  if (!Array.isArray(content)) return content;
  if (!content.some((p) => p.type === 'image')) return textOf(content);
  const out = [];
  for (const part of content)
    out.push(
      part.type === 'image'
        ? { type: 'image', data: await pngOf(part.image) }
        : { type: 'text', text: part.text }
    );
  return out;
}

export async function toMessages(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === 'user') out.push({ role: 'user', content: await userContent(m.content) });
    else if (m.role === 'assistant')
      out.push({
        role: 'assistant',
        content: m.content === null ? null : textOf(m.content),
        ...(m.reasoning_content ? { reasoning_content: m.reasoning_content } : {}),
        ...(m.tool_calls ? { tool_calls: m.tool_calls } : {}),
      });
    else if (m.role === 'tool')
      out.push({ role: 'tool', tool_call_id: m.tool_call_id, content: textOf(m.content) });
    else out.push({ role: 'system', content: textOf(m.content) });
  }
  return out;
}

function defined(object) {
  return Object.fromEntries(Object.entries(object).filter(([, v]) => v !== undefined));
}

export async function completionParams(request, { signal, extra = {} } = {}) {
  const tools = request.tools.length > 0;
  return defined({
    messages: await toMessages(request.messages),
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: request.maxTokens,
    temperature: request.temperature,
    top_p: request.topP,
    top_k: request.topK,
    seed: request.seed,
    stop: request.stop.length > 0 ? request.stop : undefined,
    tools: tools ? request.tools : undefined,
    tool_choice: tools ? request.toolChoice : undefined,
    chat_template_kwargs:
      request.reasoningEffort === 'none' ? { enable_thinking: false } : undefined,
    abortSignal: signal,
    ...extra,
  });
}

export function failure(error, signal) {
  if (signal?.aborted) return signal.reason;
  if (error?.name === 'AbortError') return error;
  if (CONTEXT.test(error?.message ?? ''))
    return new HttpError(400, error.message, {
      code: 'context_length_exceeded',
      param: 'messages',
    });
  return error;
}

function usageOf(usage, timings) {
  const prompt = usage?.prompt_tokens ?? (timings?.cache_n ?? 0) + (timings?.prompt_n ?? 0);
  const completion = usage?.completion_tokens ?? timings?.predicted_n ?? 0;
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
    prompt_tokens_details: {
      cached_tokens: usage?.prompt_tokens_details?.cached_tokens ?? timings?.cache_n ?? 0,
    },
    completion_tokens_details: {
      reasoning_tokens: usage?.completion_tokens_details?.reasoning_tokens ?? 0,
    },
  };
}

function timingsOf(t) {
  if (!t) return {};
  return {
    cache_n: t.cache_n,
    prompt_n: t.prompt_n,
    prompt_ms: t.prompt_ms,
    prompt_per_second: t.prompt_per_second,
    predicted_n: t.predicted_n,
    predicted_ms: t.predicted_ms,
    predicted_per_second: t.predicted_per_second,
  };
}

function relay(delta, emit, calls) {
  if (delta.reasoning_content) emit({ type: 'reasoning', text: delta.reasoning_content });
  if (delta.content) emit({ type: 'content', text: delta.content });
  for (const call of delta.tool_calls ?? []) {
    const index = call.index ?? 0;
    if (!calls.has(index)) {
      calls.set(index, calls.size);
      emit({
        type: 'tool',
        index: calls.get(index),
        id: call.id || toolCallId(),
        name: call.function?.name ?? '',
      });
    }
    if (call.function?.arguments)
      emit({ type: 'args', index: calls.get(index), text: call.function.arguments });
  }
}

export async function streamChat(wllama, request, { signal, emit, yieldTurn = async () => {} }) {
  if (signal?.aborted) throw signal.reason;
  const params = await completionParams(request, { signal });
  const calls = new Map();
  let started = false;
  let finish = 'stop';
  let usage = null;
  let timings = null;
  try {
    const stream = await wllama.createChatCompletion(params);
    for await (const chunk of stream) {
      if (chunk.usage) usage = chunk.usage;
      if (chunk.timings) timings = chunk.timings;
      const choice = chunk.choices?.[0];
      if (!choice) continue;
      if (!started) {
        started = true;
        emit({ type: 'start', promptTokens: null });
      }
      relay(choice.delta ?? {}, emit, calls);
      if (choice.finish_reason) finish = choice.finish_reason;
      await yieldTurn();
    }
  } catch (error) {
    throw failure(error, signal);
  }
  if (!started) emit({ type: 'start', promptTokens: null });
  return {
    finishReason: finish,
    usage: usageOf(usage, timings),
    timings: timingsOf(timings),
    settings: defined({
      temperature: params.temperature,
      top_p: params.top_p,
      top_k: params.top_k,
      max_tokens: params.max_tokens,
    }),
  };
}

export async function greedy(wllama, request, { tokens, top = 5 }) {
  const params = await completionParams(
    { ...request, maxTokens: tokens, temperature: 0 },
    { extra: { logprobs: true, top_logprobs: top } }
  );
  const steps = [];
  try {
    const stream = await wllama.createChatCompletion(params);
    for await (const chunk of stream)
      for (const step of chunk.choices?.[0]?.logprobs?.content ?? []) {
        const alternatives = (step.top_logprobs ?? []).filter((t) => t.token !== step.token);
        steps.push({
          token: step.token,
          logprob: step.logprob,
          margin: alternatives.length > 0 ? step.logprob - alternatives[0].logprob : null,
          top: (step.top_logprobs ?? []).map((t) => t.token),
        });
      }
  } catch (error) {
    throw failure(error);
  }
  return steps;
}

export function noImages(request) {
  if (
    request.messages.some(
      (m) => Array.isArray(m.content) && m.content.some((p) => p.type === 'image')
    )
  )
    throw badRequest(
      'this model takes no images: no vision projector (mmproj) for it is in the cache',
      'messages'
    );
}

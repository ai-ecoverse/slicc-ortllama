import { badRequest } from './http/errors.js';
import { PRIORITY } from './runtime.js';

const ROLES = new Set(['system', 'developer', 'user', 'assistant', 'tool']);
const EFFORTS = new Set(['none', 'minimal', 'low', 'medium', 'high', 'xhigh']);
const TOOL_NAME = /^[A-Za-z0-9_.-]{1,64}$/;

const isObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v);

function content(value, at, { nullable = false } = {}) {
  if (typeof value === 'string') return value;
  if ((value === null || value === undefined) && nullable) return null;
  if (!Array.isArray(value))
    throw badRequest(`${at}.content must be a string or an array of parts`, 'messages');
  return value.map((part, j) => {
    if (part?.type !== 'text')
      throw badRequest(
        `${at}.content[${j}] is ${part?.type ?? 'not a part'}; this model accepts text only`,
        'messages'
      );
    if (typeof part.text !== 'string')
      throw badRequest(`${at}.content[${j}].text must be a string`, 'messages');
    return { type: 'text', text: part.text };
  });
}

function toolCalls(value, at) {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw badRequest(`${at}.tool_calls must be an array`, 'messages');
  return value.map((call, j) => {
    const fn = call?.function;
    if (!isObject(fn) || typeof fn.name !== 'string')
      throw badRequest(`${at}.tool_calls[${j}] needs function.name`, 'messages');
    const args =
      typeof fn.arguments === 'string' ? fn.arguments : JSON.stringify(fn.arguments ?? {});
    return {
      id: String(call.id ?? ''),
      type: 'function',
      function: { name: fn.name, arguments: args },
    };
  });
}

function messages(list) {
  if (!Array.isArray(list) || list.length === 0)
    throw badRequest('messages must be a non-empty array', 'messages');
  return list.map((m, i) => {
    const at = `messages[${i}]`;
    if (!isObject(m) || !ROLES.has(m.role))
      throw badRequest(`${at}.role must be one of ${[...ROLES].join(', ')}`, 'messages');
    if (m.role === 'assistant') {
      const calls = toolCalls(m.tool_calls, at);
      return {
        role: 'assistant',
        content: content(m.content, at, { nullable: true }),
        ...(calls ? { tool_calls: calls } : {}),
      };
    }
    if (m.role === 'tool')
      return {
        role: 'tool',
        tool_call_id: String(m.tool_call_id ?? ''),
        content: content(m.content, at),
      };
    return { role: m.role, content: content(m.content, at) };
  });
}

function tools(list) {
  if (list === undefined || list === null) return [];
  if (!Array.isArray(list)) throw badRequest('tools must be an array', 'tools');
  return list.map((t, i) => {
    const fn = t?.function;
    if (t?.type !== 'function' || !isObject(fn) || !TOOL_NAME.test(fn.name ?? ''))
      throw badRequest(
        `tools[${i}] must be a function with a name of letters, digits, _, . or -`,
        'tools'
      );
    return {
      type: 'function',
      function: {
        name: fn.name,
        ...(typeof fn.description === 'string' ? { description: fn.description } : {}),
        ...(isObject(fn.parameters) ? { parameters: fn.parameters } : {}),
      },
    };
  });
}

function toolChoice(value) {
  if (value === undefined || value === null) return 'auto';
  if (['auto', 'none', 'required'].includes(value)) return value;
  if (isObject(value) && value.type === 'function' && typeof value.function?.name === 'string')
    return { type: 'function', function: { name: value.function.name } };
  throw badRequest('tool_choice must be auto, none, required or a function', 'tool_choice');
}

function number(body, key, { min, max, integer = false, above = false }) {
  const v = body[key];
  if (v === undefined || v === null) return undefined;
  const ok =
    typeof v === 'number' &&
    Number.isFinite(v) &&
    (!integer || Number.isInteger(v)) &&
    (above ? v > min : v >= min) &&
    v <= max;
  if (!ok)
    throw badRequest(
      `${key} must be a ${integer ? 'whole ' : ''}number from ${min} to ${max}`,
      key
    );
  return v;
}

function stops(value) {
  if (value === undefined || value === null) return [];
  const list = typeof value === 'string' ? [value] : value;
  if (!Array.isArray(list) || list.length > 4 || list.some((s) => typeof s !== 'string'))
    throw badRequest('stop must be a string or up to 4 strings', 'stop');
  return list;
}

export function parseChatRequest(body) {
  if (!isObject(body)) throw badRequest('request body must be a JSON object');
  if (body.n !== undefined && body.n !== null && body.n !== 1) throw badRequest('n must be 1', 'n');
  if (
    body.reasoning_effort !== undefined &&
    body.reasoning_effort !== null &&
    !EFFORTS.has(body.reasoning_effort)
  )
    throw badRequest(
      `reasoning_effort must be one of ${[...EFFORTS].join(', ')}`,
      'reasoning_effort'
    );
  const big = Number.MAX_SAFE_INTEGER;
  return {
    model: body.model ?? 'chat',
    stream: body.stream === true,
    includeUsage: body.stream_options?.include_usage === true,
    request: {
      messages: messages(body.messages),
      tools: tools(body.tools),
      toolChoice: toolChoice(body.tool_choice),
      maxTokens:
        number(body, 'max_completion_tokens', { min: 1, max: big, integer: true }) ??
        number(body, 'max_tokens', { min: 1, max: big, integer: true }),
      temperature: number(body, 'temperature', { min: 0, max: 2 }),
      topP: number(body, 'top_p', { min: 0, max: 1, above: true }),
      topK: number(body, 'top_k', { min: 0, max: big, integer: true }),
      seed: number(body, 'seed', { min: -big, max: big, integer: true }),
      stop: stops(body.stop),
      reasoningEffort: body.reasoning_effort ?? undefined,
    },
  };
}

function channel() {
  const items = [];
  let wake = null;
  let done = false;
  let error = null;
  const poke = () => {
    wake?.();
    wake = null;
  };
  return {
    push(item) {
      items.push(item);
      poke();
    },
    end() {
      done = true;
      poke();
    },
    fail(e) {
      error = e;
      done = true;
      poke();
    },
    async *[Symbol.asyncIterator]() {
      for (;;) {
        if (items.length > 0) yield items.shift();
        else if (error) throw error;
        else if (done) return;
        else await new Promise((resolve) => (wake = resolve));
      }
    },
  };
}

function delta(event) {
  if (event.type === 'start') return { role: 'assistant', content: '' };
  if (event.type === 'content') return { content: event.text };
  if (event.type === 'reasoning') return { reasoning_content: event.text };
  if (event.type === 'tool')
    return {
      tool_calls: [
        {
          index: event.index,
          id: event.id,
          type: 'function',
          function: { name: event.name, arguments: '' },
        },
      ],
    };
  return { tool_calls: [{ index: event.index, function: { arguments: event.text } }] };
}

function message(events) {
  let text = '';
  let reasoning = '';
  const calls = [];
  for (const e of events) {
    if (e.type === 'content') text += e.text;
    else if (e.type === 'reasoning') reasoning += e.text;
    else if (e.type === 'tool')
      calls.push({ id: e.id, type: 'function', function: { name: e.name, arguments: '' } });
    else if (e.type === 'args') calls[e.index].function.arguments += e.text;
  }
  return {
    role: 'assistant',
    content: text || (calls.length > 0 ? null : ''),
    ...(reasoning ? { reasoning_content: reasoning } : {}),
    ...(calls.length > 0 ? { tool_calls: calls } : {}),
  };
}

export function chatRoute({ catalog, runtime, now = Date.now, newId = () => crypto.randomUUID() }) {
  return {
    async handle({ body, signal }) {
      const parsed = parseChatRequest(body);
      const resolved = await catalog.resolve(parsed.model);
      if (resolved.engine.capability !== 'chat')
        throw badRequest(`model ${resolved.id} does not answer chat completions`, 'model');
      const id = `chatcmpl-${newId().replaceAll('-', '')}`;
      const created = Math.floor(now() / 1000);
      const base = { id, created, model: parsed.model };
      const events = channel();
      let onStart;
      const started = new Promise((resolve) => (onStart = resolve));
      const job = runtime.use(
        resolved,
        PRIORITY.chat,
        (instance, _entry, yieldTurn) =>
          instance.chat(parsed.request, {
            signal,
            yieldTurn,
            emit(event) {
              if (event.type === 'start') onStart();
              events.push(event);
            },
          }),
        signal
      );
      job.then(
        (result) => {
          events.push({ type: 'done', result });
          events.end();
        },
        (error) => events.fail(error)
      );
      await Promise.race([started, job]);
      if (!parsed.stream) {
        const all = [];
        for await (const e of events) all.push(e);
        const { result } = all.at(-1);
        return {
          ...base,
          object: 'chat.completion',
          choices: [
            { index: 0, message: message(all), finish_reason: result.finishReason, logprobs: null },
          ],
          usage: result.usage,
          timings: result.timings,
        };
      }
      const chunk = (choices, extra = {}) => ({
        ...base,
        object: 'chat.completion.chunk',
        choices,
        ...extra,
      });
      async function* stream() {
        for await (const e of events) {
          if (e.type !== 'done') {
            yield chunk([{ index: 0, delta: delta(e), finish_reason: null }]);
            continue;
          }
          yield chunk([{ index: 0, delta: {}, finish_reason: e.result.finishReason }], {
            timings: e.result.timings,
          });
          if (parsed.includeUsage) yield chunk([], { usage: e.result.usage });
        }
      }
      return { events: stream() };
    },
  };
}

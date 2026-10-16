import { badRequest } from '../../http/errors.js';

const JOINER = '⁠';

export const MISTRAL_SPLIT =
  '[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+';

export function escaper(names) {
  const sorted = [...new Set(names)]
    .filter((n) => n.length > 1)
    .sort((a, b) => b.length - a.length);
  const escape = (text) => {
    let out = text;
    for (const name of sorted)
      if (out.includes(name)) out = out.split(name).join(name[0] + JOINER + name.slice(1));
    return out;
  };
  const deep = (value) => {
    if (typeof value === 'string') return escape(value);
    if (Array.isArray(value)) return value.map(deep);
    if (value && typeof value === 'object')
      return Object.fromEntries(Object.entries(value).map(([k, v]) => [escape(k), deep(v)]));
    return value;
  };
  return { escape, deep };
}

export function textOf(content) {
  if (content === null || content === undefined) return '';
  if (typeof content === 'string') return content;
  return content.map((part) => part.text ?? '').join('');
}

function merged(messages) {
  const out = [];
  for (const m of messages) {
    const last = out.at(-1);
    if (m.role === 'user' && last?.role === 'user') {
      last.content = `${last.content}\n\n${m.content}`;
      continue;
    }
    out.push({ ...m });
  }
  return out;
}

function chatMessages(messages, escape) {
  const system = [];
  let i = 0;
  for (; i < messages.length && ['system', 'developer'].includes(messages[i].role); i++)
    system.push(escape.escape(textOf(messages[i].content)));
  const rest = messages.slice(i).map((m) => {
    if (m.role === 'assistant')
      return {
        role: 'assistant',
        content: escape.escape(textOf(m.content)),
        ...(m.tool_calls?.length
          ? {
              tool_calls: m.tool_calls.map((c) => ({
                id: c.id,
                type: 'function',
                function: {
                  name: escape.escape(c.function.name),
                  arguments: escape.escape(c.function.arguments ?? ''),
                },
              })),
            }
          : {}),
      };
    if (m.role === 'tool')
      return {
        role: 'tool',
        tool_call_id: m.tool_call_id,
        content: escape.escape(textOf(m.content)),
      };
    return { role: 'user', content: escape.escape(textOf(m.content)) };
  });
  const head = system.length > 0 ? [{ role: 'system', content: system.join('\n\n') }] : [];
  return [...head, ...merged(rest)];
}

const ALTERNATION =
  /\{\{-?\s*raise_exception\(\s*'After the optional system message[^']*'\s*\)\s*-?\}\}/;

export const mistral = {
  name: 'mistral',
  markers: { toolCall: '[TOOL_CALLS]', args: '[ARGS]', think: '[THINK]', thinkEnd: '[/THINK]' },
  tools: true,
  matches: (template) => template.includes('[TOOL_CALLS]') && template.includes('[INST]'),
  template: (source) => source.replace(ALTERNATION, ''),
  tokenizer(json) {
    const first =
      json.pre_tokenizer?.type === 'Sequence' ? json.pre_tokenizer.pretokenizers[0] : null;
    if (first?.type === 'Split' && first.pattern?.Regex && first.pattern.Regex !== MISTRAL_SPLIT)
      first.pattern = { Regex: MISTRAL_SPLIT };
    return json;
  },
  messages: chatMessages,
};

export const plain = {
  name: 'plain',
  markers: {},
  tools: false,
  matches: () => true,
  template: (source) => source,
  tokenizer: (json) => json,
  messages: chatMessages,
};

export const PROFILES = [mistral, plain];

export function pickProfile(template, hint = null) {
  if (hint) {
    const named = PROFILES.find((p) => p.name === hint);
    if (!named) throw new Error(`no chat profile named ${hint}`);
    return named;
  }
  return PROFILES.find((p) => p.matches(template));
}

export function checkTools(profile, tools) {
  if (tools.length > 0 && !profile.tools)
    throw badRequest(`this model's ${profile.name} chat profile cannot parse tool calls`, 'tools');
}

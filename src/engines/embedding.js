import { Tokenizer } from '@huggingface/tokenizers';
import { badRequest, HttpError } from '../http/errors.js';
import { DTYPES, part } from './decoder/layout.js';
import { sessionFor, toFloat32 } from './decoder/session.js';
import { missingExternal } from './external.js';

export const MAX_BATCH = 32;
export const BATCH_TOKENS = 2048;
export const REQUEST_TOKENS = 300_000;

const META = ['config.json', 'tokenizer.json', 'tokenizer_config.json'];
const TEXT_INPUTS = new Set(['input_ids', 'attention_mask']);
const EMPTY = { float32: Float32Array, float16: Uint16Array, int64: BigInt64Array };
const INPUT_TYPES = new Set(['query', 'document']);
const MISSING_KERNEL =
  /(?:Could not find an implementation for|Failed to find kernel for) (?:[\w.]+\.)?(\w+)\(/;

const GEMMA_PREFIXED = /^\s*(?:task:[^|\n]*\|\s*query:|title:[^|\n]*\|\s*text:)/;

function gemmaPrompt(text, { task, inputType, title, symmetric }) {
  if (GEMMA_PREFIXED.test(text)) return text;
  if (inputType === 'document' && !symmetric) return `title: ${title ?? 'none'} | text: ${text}`;
  return `task: ${task ?? 'search result'} | query: ${text}`;
}

export const FAMILIES = {
  embedding_gemma2: {
    architectures: ['EmbeddingGemma2Model'],
    contextLength: 8192,
    dimensions: [768, 512, 256, 128],
    output: 'sentence_embedding',
    variants: ['q8', 'q4', 'fp16', 'q4f16', 'fp32'],
    cosine: { fp32: 0.9999, fp16: 0.999, q8: 0.999, q4: 0.97, q4f16: 0.97 },
    tasks: {
      'search result': false,
      'question answering': false,
      'fact checking': false,
      'code retrieval': false,
      classification: true,
      clustering: true,
      'sentence similarity': true,
    },
    prompt: gemmaPrompt,
  },
};

async function configOf(files) {
  if (!(await files.exists('config.json'))) return null;
  try {
    return await files.json('config.json');
  } catch {
    return null;
  }
}

function familyOf(config) {
  const architectures = config?.architectures ?? [];
  return (
    Object.entries(FAMILIES).find(
      ([type, f]) =>
        config?.model_type === type || f.architectures.some((a) => architectures.includes(a))
    ) ?? null
  );
}

export function truncate(vector, dimensions = vector.length) {
  const out = new Float32Array(vector.subarray(0, dimensions));
  let sum = 0;
  for (const x of out) sum += x * x;
  const norm = Math.sqrt(sum);
  if (norm > 0) for (let i = 0; i < out.length; i++) out[i] /= norm;
  return out;
}

export function base64(vector) {
  const bytes = new Uint8Array(vector.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < vector.length; i++) view.setFloat32(i * 4, vector[i], true);
  return Buffer.from(bytes).toString('base64');
}

export function cosine(a, b) {
  let dot = 0;
  let x = 0;
  let y = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    x += a[i] * a[i];
    y += b[i] * b[i];
  }
  return dot / Math.sqrt(x * y);
}

export function batches(lengths, { maxInputs = MAX_BATCH, maxTokens = BATCH_TOKENS } = {}) {
  const order = lengths.map((_, i) => i).sort((a, b) => lengths[b] - lengths[a]);
  const out = [];
  let current = [];
  for (const i of order) {
    const full =
      current.length >= maxInputs || lengths[current[0]] * (current.length + 1) > maxTokens;
    if (current.length > 0 && full) {
      out.push(current);
      current = [];
    }
    current.push(i);
  }
  if (current.length > 0) out.push(current);
  return out;
}

function titlesOf(title, count) {
  if (title === undefined) return Array(count).fill(undefined);
  const list = Array.isArray(title) ? title : Array(count).fill(title);
  const ok =
    list.length === count && list.every((t) => t === null || (typeof t === 'string' && t !== ''));
  if (!ok)
    throw badRequest(
      'title must be a non-empty string, or an array with a string or null for each input',
      'title'
    );
  return list.map((t) => t ?? undefined);
}

export const promptOptions = (body) => ({
  task: body.task ?? undefined,
  inputType: body.input_type ?? undefined,
  title: body.title ?? undefined,
});

export function applyPrompts(info, texts, { task, inputType, title } = {}) {
  const family = FAMILIES[info.family];
  if (task === undefined && inputType === undefined && title === undefined) return texts;
  const tasks = Object.keys(family.tasks);
  if (task !== undefined && !tasks.includes(task))
    throw badRequest(`task must be one of ${tasks.join(', ')}`, 'task');
  if (inputType !== undefined && !INPUT_TYPES.has(inputType))
    throw badRequest('input_type must be query or document', 'input_type');
  const symmetric = family.tasks[task] === true;
  if (title !== undefined && (inputType !== 'document' || symmetric))
    throw badRequest(
      'title applies only to input_type document, and not to classification, clustering or sentence similarity',
      'title'
    );
  const titles = titlesOf(title, texts.length);
  return texts.map((text, i) =>
    family.prompt(text, { task, inputType, title: titles[i], symmetric })
  );
}

function contextError(index, tokens, limit) {
  return new HttpError(
    400,
    `input[${index}] has ${tokens} tokens, which exceeds the context window of ${limit} tokens; shorten it or split it`,
    { code: 'context_length_exceeded', param: 'input' }
  );
}

function emptyInputs(ort, session) {
  const feeds = {};
  for (const meta of session.inputMetadata ?? []) {
    if (TEXT_INPUTS.has(meta.name)) continue;
    const dims = (meta.shape ?? []).map((d) => (typeof d === 'number' ? d : 0));
    const Typed = EMPTY[meta.type] ?? Float32Array;
    feeds[meta.name] = new ort.Tensor(meta.type ?? 'float32', new Typed(0), dims);
  }
  return feeds;
}

export function createEmbedder({ ort, session, tokenizer, family, info, pad = 0 }) {
  const empties = emptyInputs(ort, session);
  const limit = info.contextLength;

  async function run(rows) {
    const width = Math.max(...rows.map((r) => r.length));
    const ids = new BigInt64Array(rows.length * width).fill(BigInt(pad));
    const mask = new BigInt64Array(rows.length * width);
    rows.forEach((row, b) => {
      row.forEach((id, i) => {
        ids[b * width + i] = BigInt(id);
        mask[b * width + i] = 1n;
      });
    });
    const dims = [rows.length, width];
    const out = await session.run(
      {
        input_ids: new ort.Tensor('int64', ids, dims),
        attention_mask: new ort.Tensor('int64', mask, dims),
        ...empties,
      },
      [family.output]
    );
    const tensor = out[family.output];
    const data = toFloat32(await tensor.getData());
    tensor.dispose?.();
    const size = tensor.dims.at(-1);
    return rows.map((_, b) => data.slice(b * size, (b + 1) * size));
  }

  return {
    family: info.family,
    contextLength: limit,
    dimensions: info.dimensions,
    prompt: (texts, options) => applyPrompts(info, texts, options),
    async embed(texts, { signal, yieldTurn } = {}) {
      const ids = texts.map((text) => tokenizer.encode(text).ids);
      const tokens = ids.map((row) => row.length);
      tokens.forEach((n, i) => {
        if (n > limit) throw contextError(i, n, limit);
      });
      const total = tokens.reduce((a, b) => a + b, 0);
      if (total > REQUEST_TOKENS)
        throw badRequest(
          `the input has ${total} tokens in all, more than the ${REQUEST_TOKENS} a request takes; send fewer inputs at a time`,
          'input'
        );
      const vectors = new Array(texts.length);
      for (const [n, batch] of batches(tokens).entries()) {
        if (n > 0) await yieldTurn?.();
        signal?.throwIfAborted();
        const rows = await run(batch.map((i) => ids[i]));
        batch.forEach((i, b) => {
          vectors[i] = rows[b];
        });
      }
      return { vectors, tokens };
    },
    release: () => session.release?.(),
  };
}

function unsupported(error, info, variant, backend) {
  const op = MISSING_KERNEL.exec(error?.message ?? '')?.[1];
  if (!op) return null;
  const others = info.variants.filter((v) => v !== variant && (v === 'fp32' || v === 'fp16'));
  const where = `${backend.executionProviders.join(',')} (onnxruntime-${backend.track})`;
  const instead = others.length > 0 ? `variant ${others.join(' or ')} here, or ` : '';
  return new HttpError(
    422,
    `variant ${variant} does not run on ${where}, which has no ${op} kernel; use ${instead}a backend with WebGPU`,
    { code: 'model_unsupported', param: 'model' }
  );
}

async function sessionOf(backend, files, info, variant) {
  try {
    return await sessionFor({
      backend,
      files,
      part: info.parts[variant],
      options: { ...backend.sessionOptions, executionProviders: backend.executionProviders },
      node: backend.track === 'node',
    });
  } catch (error) {
    throw unsupported(error, info, variant, backend) ?? error;
  }
}

export const embedding = {
  name: 'embedding',
  capability: 'embeddings',
  async detect(files) {
    const config = await configOf(files);
    const found = familyOf(config);
    if (!found) return null;
    const [type, family] = found;
    const external = config['transformers.js_config']?.use_external_data_format ?? {};
    const parts = {};
    for (const variant of family.variants) {
      const name = `model${DTYPES[variant]}.onnx`;
      if (await files.exists(`onnx/${name}`))
        parts[variant] = await part(files, external, 'onnx/', name);
    }
    const variants = Object.keys(parts);
    if (variants.length === 0) return null;
    return {
      family: type,
      variants,
      defaultVariant: variants[0],
      inputModalities: ['text'],
      outputModalities: ['embeddings'],
      contextLength: family.contextLength,
      dimensions: family.dimensions,
      pad: config.text_config?.pad_token_id ?? config.pad_token_id ?? 0,
      parts,
    };
  },
  async missing(files, info, variant) {
    const graph = info.parts[variant];
    const out = [];
    for (const rel of [...META, graph.model, ...graph.data])
      if (!(await files.exists(rel))) out.push(rel);
    for (const rel of await missingExternal(files, [graph.model]))
      if (!out.includes(rel)) out.push(rel);
    return out;
  },
  include: (info, variant) => ['*.json', `${info.parts[variant].model}*`],
  specialize: async (_files, info) => ({ info, details: { dimensions: info.dimensions } }),
  prompt: (info, texts, options) => applyPrompts(info, texts, options),
  async load({ backend, files, variant, info }) {
    const family = FAMILIES[info.family];
    const [tokenizerJson, tokenizerConfig] = await Promise.all([
      files.json('tokenizer.json'),
      files.json('tokenizer_config.json'),
    ]);
    const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    const session = await sessionOf(backend, files, info, variant);
    if (!session.outputNames.includes(family.output)) {
      await session.release?.();
      throw new Error(
        `${info.parts[variant].model} has no ${family.output} output; it has ${session.outputNames.join(', ')}`
      );
    }
    return createEmbedder({ ort: backend.ort, session, tokenizer, family, info, pad: info.pad });
  },
};

embedding.checkOverrides = () => ({});

const PLANETS = ['Venus', 'Mars', 'Jupiter', 'Saturn'];

const SMOKE = [
  'task: search result | query: Which planet is known as the Red Planet?',
  "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
  'title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.',
  'title: none | text: Jupiter, the largest planet in our solar system, has a prominent red spot.',
  'title: none | text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet.',
];

const round = (x) => Math.round(x * 1e4) / 1e4;

function ranking(query, docs) {
  return docs
    .map((doc) => ({ name: doc.name, score: cosine(query, doc.vector) }))
    .sort((a, b) => b.score - a.score);
}

export function compareEmbeddings(vectors, fixtures, { threshold, tokens = [] }) {
  const byName = new Map(fixtures.map((f, i) => [f.name, vectors[i]]));
  const cosines = [];
  const retokenized = [];
  fixtures.forEach((f, i) => {
    if (tokens[i] !== undefined && f.tokens !== undefined && tokens[i] !== f.tokens)
      retokenized.push(f.name);
    for (const [dim, ref] of Object.entries(f.embeddings))
      cosines.push({
        at: `${f.name}@${dim}`,
        cosine: cosine(truncate(vectors[i], Number(dim)), ref),
      });
  });
  const ranks = [];
  for (const f of fixtures.filter((x) => x.ranks)) {
    for (const [dim, order] of Object.entries(f.ranks)) {
      const cut = (name) => truncate(byName.get(name), Number(dim));
      const got = ranking(
        cut(f.name),
        order.map((name) => ({ name, vector: cut(name) }))
      ).map((r) => r.name);
      ranks.push({
        at: `${f.name}@${dim}`,
        top: got[0] === order[0],
        same: got.join() === order.join(),
      });
    }
  }
  const worst = cosines.reduce((a, b) => (b.cosine < a.cosine ? b : a), cosines[0]);
  const mean = cosines.reduce((s, c) => s + c.cosine, 0) / Math.max(1, cosines.length);
  const moved = ranks.filter((r) => !r.top).map((r) => r.at);
  return {
    texts: fixtures.length,
    min: worst?.cosine ?? 1,
    worstAt: worst?.at ?? '',
    mean,
    threshold,
    rankings: ranks.length,
    reordered: ranks.filter((r) => !r.same).map((r) => r.at),
    moved,
    retokenized,
    ok: (worst?.cosine ?? 1) >= threshold && moved.length === 0 && retokenized.length === 0,
  };
}

embedding.check = async (instance, { fixtures, variant, now = () => performance.now() } = {}) => {
  const started = now();
  const { vectors } = await instance.embed(SMOKE);
  const requestMs = now() - started;
  const docs = PLANETS.map((name, i) => ({ name, vector: vectors[i + 1] }));
  const ranked = ranking(vectors[0], docs);
  const answer = { query: SMOKE[0], ranking: ranked.map((r) => ({ ...r, score: round(r.score) })) };
  const smoke = { ok: ranked[0].name === 'Mars', requestMs, answer };
  if (!fixtures) return smoke;
  for (const f of fixtures) {
    const [made] = f.request ? instance.prompt([f.text], promptOptions(f.request)) : [f.text];
    if (made !== f.input)
      throw new Error(`fixture ${f.name}: the prompt is ${made}, not ${f.input}`);
  }
  const inputs = fixtures.map((f) => f.input);
  const embedded = await instance.embed(inputs);
  const threshold = FAMILIES[instance.family].cosine[variant] ?? 0.99;
  const compared = compareEmbeddings(embedded.vectors, fixtures, {
    threshold,
    tokens: embedded.tokens,
  });
  const dims = [...new Set(fixtures.flatMap((f) => Object.keys(f.embeddings)))].sort(
    (a, b) => b - a
  );
  return {
    ...smoke,
    ...compared,
    summary: `fixtures: ${compared.texts} texts at ${dims.join(', ')} dimensions, cosine min ${compared.min.toFixed(5)} (${compared.worstAt}), mean ${compared.mean.toFixed(5)}, threshold ${threshold}; ${compared.rankings - compared.moved.length} of ${compared.rankings} rankings keep their top document${compared.reordered.length ? `, reordered below it: ${compared.reordered.join(', ')}` : ''}${compared.retokenized.length ? `; tokenized differently: ${compared.retokenized.join(', ')}` : ''}`,
    ok: smoke.ok && compared.ok,
  };
};

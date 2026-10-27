import { refuse } from './chat.js';
import { base64, promptOptions, truncate } from './engines/embedding.js';
import { badRequest } from './http/errors.js';
import { PRIORITY } from './runtime.js';

export const MAX_INPUTS = 2048;

const FORMATS = new Set(['float', 'base64']);

const TOKEN_IDS =
  'input as token IDs is not supported, since they belong to the tokenizer that made them; send the text instead (LangChain: OpenAIEmbeddings(check_embedding_ctx_length=False))';

const isObject = (v) => typeof v === 'object' && v !== null && !Array.isArray(v);

function inputs(value) {
  const list = typeof value === 'string' ? [value] : value;
  if (!Array.isArray(list) || list.length === 0)
    throw badRequest('input must be a string or a non-empty array of strings', 'input');
  if (list.every((x) => typeof x === 'number' || Array.isArray(x)))
    throw badRequest(TOKEN_IDS, 'input');
  if (list.length > MAX_INPUTS)
    throw badRequest(`input takes at most ${MAX_INPUTS} strings, got ${list.length}`, 'input');
  list.forEach((text, i) => {
    if (typeof text !== 'string') throw badRequest(`input[${i}] must be a string`, 'input');
    if (text === '') throw badRequest(`input[${i}] is empty`, 'input');
  });
  return list;
}

export function parseEmbeddingRequest(body) {
  if (!isObject(body)) throw badRequest('request body must be a JSON object');
  const format = body.encoding_format ?? 'float';
  if (!FORMATS.has(format))
    throw badRequest('encoding_format must be float or base64', 'encoding_format');
  return {
    model: body.model ?? 'embed',
    texts: inputs(body.input),
    format,
    dimensions: body.dimensions ?? null,
    prompts: promptOptions(body),
  };
}

function dimensionsOf(requested, resolved) {
  const { dimensions } = resolved.info;
  if (requested === null) return dimensions[0];
  if (!dimensions.includes(requested))
    throw badRequest(
      `dimensions must be one of ${dimensions.join(', ')} for model ${resolved.id}`,
      'dimensions'
    );
  return requested;
}

export function embeddingsRoute({ catalog, runtime }) {
  return {
    async handle({ body, signal }) {
      const request = parseEmbeddingRequest(body);
      const resolved = await catalog.resolve(request.model);
      refuse(resolved, 'embeddings', 'embeddings requests');
      const dimensions = dimensionsOf(request.dimensions, resolved);
      const texts = resolved.engine.prompt(resolved.info, request.texts, request.prompts);
      const { vectors, tokens } = await runtime.use(
        resolved,
        PRIORITY.embeddings,
        (instance, _entry, yieldTurn) => instance.embed(texts, { signal, yieldTurn }),
        signal
      );
      const encode = request.format === 'base64' ? base64 : (v) => Array.from(v);
      const used = tokens.reduce((a, b) => a + b, 0);
      return {
        object: 'list',
        data: vectors.map((vector, index) => ({
          object: 'embedding',
          index,
          embedding: encode(truncate(vector, dimensions)),
        })),
        model: request.model,
        usage: { prompt_tokens: used, total_tokens: used },
      };
    },
  };
}

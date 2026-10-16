export const DTYPES = {
  q4f16: '_q4f16',
  q4: '_q4',
  fp16: '_fp16',
  q8: '_quantized',
  int8: '_int8',
  uint8: '_uint8',
  bnb4: '_bnb4',
  fp32: '',
};

export const MAX_CONTEXT = 32768;

const TOKENIZER = ['tokenizer.json', 'tokenizer_config.json'];

async function readJson(files, rel) {
  try {
    return (await files.exists(rel)) ? await files.json(rel) : null;
  } catch {
    return null;
  }
}

async function templateOf(files, dirs) {
  for (const dir of dirs) {
    const rel = `${dir}chat_template.jinja`;
    if (await files.exists(rel)) return { file: rel };
  }
  for (const dir of dirs) {
    const config = await readJson(files, `${dir}tokenizer_config.json`);
    const template = config?.chat_template;
    if (typeof template === 'string') return { config: `${dir}tokenizer_config.json` };
    if (Array.isArray(template) && template.some((t) => t?.name === 'default'))
      return { config: `${dir}tokenizer_config.json` };
  }
  return null;
}

async function tokenizerDir(files, dirs) {
  for (const dir of dirs) if (await files.exists(`${dir}tokenizer.json`)) return dir;
  return dirs.at(-1);
}

function dataFiles(external, name) {
  const count = external[name];
  if (count === true) return [`${name}_data`];
  if (typeof count !== 'number' || count < 1) return null;
  return Array.from({ length: count }, (_, i) => `${name}_data${i === 0 ? '' : `_${i}`}`);
}

async function probeData(files, dir, name) {
  const out = [];
  for (const candidate of [`${name}.data`, `${name}_data`]) {
    if (await files.exists(`${dir}${candidate}`)) out.push(candidate);
  }
  for (let i = 1; out.includes(`${name}_data`); i++) {
    const next = `${name}_data_${i}`;
    if (!(await files.exists(`${dir}${next}`))) break;
    out.push(next);
  }
  return out;
}

async function part(files, external, dir, name) {
  const data = dataFiles(external, name) ?? (await probeData(files, dir, name));
  return { model: `${dir}${name}`, data: data.map((d) => `${dir}${d}`) };
}

function contextOf(...values) {
  const n = values.find((v) => Number.isInteger(v) && v > 0);
  return n ? Math.min(n, MAX_CONTEXT) : MAX_CONTEXT;
}

function textConfig(config) {
  return config.text_config ?? config;
}

function causal(config) {
  return (config.architectures ?? []).some((a) => /ForCausalLM|ForConditionalGeneration/.test(a));
}

async function onnxVariants(has) {
  for (const stem of ['decoder_model_merged', 'model']) {
    const variants = [];
    for (const [variant, suffix] of Object.entries(DTYPES))
      if (await has(`${stem}${suffix}.onnx`)) variants.push(variant);
    if (variants.length > 0) return { stem, variants };
  }
  return null;
}

function dtypeOr(dtype) {
  return DTYPES[dtype] === undefined ? 'fp32' : dtype;
}

async function sibling(files, external, has, name, variant, dtype) {
  const own = (await has(`${name}${DTYPES[variant]}.onnx`)) ? variant : dtypeOr(dtype);
  return part(files, external, 'onnx/', `${name}${DTYPES[own]}.onnx`);
}

async function onnxParts({ files, external, has, stem, variants, dtypes }) {
  const parts = {};
  for (const variant of variants) {
    const decoder = await part(files, external, 'onnx/', `${stem}${DTYPES[variant]}.onnx`);
    if (stem !== 'decoder_model_merged') {
      parts[variant] = { decoder, embed: null };
      continue;
    }
    const embed = await sibling(
      files,
      external,
      has,
      'embed_tokens',
      variant,
      dtypes?.embed_tokens
    );
    const vision = await sibling(
      files,
      external,
      has,
      'vision_encoder',
      variant,
      dtypes?.vision_encoder
    );
    const known = await has(vision.model.slice('onnx/'.length));
    parts[variant] = { decoder, embed, ...(known ? { vision } : {}) };
  }
  return parts;
}

async function transformersLayout(files) {
  const config = await readJson(files, 'config.json');
  if (!config) return null;
  const tjs = config['transformers.js_config'] ?? {};
  const external = tjs.use_external_data_format ?? {};
  const has = async (name) => Object.hasOwn(external, name) || (await files.exists(`onnx/${name}`));
  const found = await onnxVariants(has);
  if (!found || (found.stem === 'model' && !causal(config))) return null;
  const template = await templateOf(files, ['']);
  if (!template) return null;
  const { stem, variants } = found;
  const parts = await onnxParts({
    files,
    external,
    has,
    stem,
    variants,
    dtypes: tjs.dtype,
  });
  const text = textConfig(config);
  const preferred = tjs.dtype?.[stem];
  return {
    layout: 'transformers.js',
    variants,
    defaultVariant: variants.includes(preferred) ? preferred : variants[0],
    contextLength: contextOf(text.max_position_embeddings, config.max_position_embeddings),
    parts,
    tokenizerDir: '',
    template,
    generation: await readJson(files, 'generation_config.json'),
    eos: [text.eos_token_id ?? config.eos_token_id].flat().filter(Number.isInteger),
  };
}

async function genaiVariant(files, dir) {
  const genai = await readJson(files, `${dir}genai_config.json`);
  const decoder = genai?.model?.decoder;
  if (typeof decoder?.filename !== 'string') return null;
  const embedding = genai.model.embedding?.filename;
  return {
    genai,
    decoder: await part(files, {}, dir, decoder.filename),
    embed: typeof embedding === 'string' ? await part(files, {}, dir, embedding) : null,
  };
}

async function genaiLayout(files) {
  const found = [];
  const root = await genaiVariant(files, '');
  if (root) found.push(['default', '', root]);
  else
    for (const dir of await files.dirs()) {
      const sub = await genaiVariant(files, `${dir}/`);
      if (sub) found.push([dir, `${dir}/`, sub]);
    }
  if (found.length === 0) return null;
  const [, firstDir, first] = found[0];
  const dirs = firstDir ? [firstDir, ''] : [''];
  const template = await templateOf(files, dirs);
  if (!template) return null;
  const search = first.genai.search ?? {};
  return {
    layout: 'genai',
    variants: found.map(([v]) => v),
    defaultVariant: found[0][0],
    contextLength: contextOf(first.genai.model.context_length),
    parts: Object.fromEntries(
      found.map(([v, dir, x]) => [v, { decoder: x.decoder, embed: x.embed, dir }])
    ),
    tokenizerDir: await tokenizerDir(files, dirs),
    template,
    generation: { temperature: search.temperature, top_p: search.top_p, top_k: search.top_k },
    eos: [first.genai.model.eos_token_id].flat().filter(Number.isInteger),
  };
}

export async function detectLayout(files) {
  return (await transformersLayout(files)) ?? (await genaiLayout(files));
}

export function neededFiles(info, variant) {
  const { decoder, embed } = info.parts[variant];
  const template = info.template.file ?? info.template.config;
  return [
    ...new Set([
      ...TOKENIZER.map((f) => `${info.tokenizerDir}${f}`),
      template,
      decoder.model,
      ...decoder.data,
      ...(embed ? [embed.model, ...embed.data] : []),
    ]),
  ];
}

export function includePatterns(info, variant) {
  const p = info.parts[variant];
  const meta = [...TOKENIZER, 'chat_template.jinja'].map((f) => `${info.tokenizerDir}${f}`);
  if (info.layout === 'genai') return [...new Set([`${p.dir}*`, ...meta])];
  return [
    'config.json',
    'generation_config.json',
    ...meta,
    `${p.decoder.model}*`,
    ...(p.embed ? [`${p.embed.model}*`] : []),
    ...(p.vision
      ? ['preprocessor_config.json', 'processor_config.json', `${p.vision.model}*`]
      : []),
  ];
}

export async function hasVision(files, info, variant) {
  const vision = info.parts[variant]?.vision;
  if (!vision) return false;
  for (const rel of ['preprocessor_config.json', vision.model, ...vision.data])
    if (!(await files.exists(rel))) return false;
  return true;
}

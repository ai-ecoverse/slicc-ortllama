import { Tokenizer } from '@huggingface/tokenizers';
import { missingExternal } from '../external.js';
import { createChat } from './chat.js';
import {
  detectLayout,
  hasVision,
  includePatterns,
  neededFiles,
  pickVision,
  visionTarget,
} from './layout.js';
import { pixtralConfig } from './pixtral.js';
import { pickProfile } from './profiles.js';
import { openModel } from './session.js';

export function samplingDefaults(generation, family) {
  const g = generation ?? {};
  return {
    temperature: g.do_sample === false ? 0 : (g.temperature ?? family.temperature),
    topP: g.top_p ?? family.topP,
    topK: g.top_k ?? family.topK,
  };
}

function tokenName(value) {
  return typeof value === 'string' ? value : (value?.content ?? '');
}

async function templateSource(files, info) {
  if (info.template.file) return files.text(info.template.file);
  const { chat_template: t } = await files.json(info.template.config);
  return typeof t === 'string' ? t : t.find((x) => x.name === 'default').template;
}

async function shapeOf(files, info, variant) {
  if (info.layout === 'genai') {
    const d = (await files.json(`${info.parts[variant].dir}genai_config.json`)).model.decoder;
    return { layers: d.num_hidden_layers, heads: d.num_key_value_heads, dim: d.head_size };
  }
  const config = await files.json('config.json');
  const t = config.text_config ?? config;
  return {
    layers: t.num_hidden_layers,
    heads: t.num_key_value_heads ?? t.num_attention_heads,
    dim: t.head_dim ?? t.hidden_size / t.num_attention_heads,
  };
}

async function visionOf(files, tokenizer, model) {
  const pre = await files.json('preprocessor_config.json');
  const processor = (await files.exists('processor_config.json'))
    ? await files.json('processor_config.json')
    : {};
  const config = await files.json('config.json');
  const id = (name, fallback) => tokenizer.token_to_id(processor[name] ?? fallback);
  return {
    cfg: pixtralConfig(pre, config),
    ids: {
      img: id('image_token', '[IMG]'),
      brk: id('image_break_token', '[IMG_BREAK]'),
      end: id('image_end_token', '[IMG_END]'),
    },
    encode: model.encodeImage,
  };
}

export const decoder = {
  name: 'decoder',
  capability: 'chat',
  async detect(files) {
    const layout = await detectLayout(files);
    if (!layout) return null;
    const image = await hasVision(files, layout, layout.defaultVariant);
    return {
      ...layout,
      inputModalities: image ? ['text', 'image'] : ['text'],
      outputModalities: ['text'],
    };
  },
  async missing(files, info, variant) {
    const out = [];
    for (const rel of neededFiles(info, variant)) if (!(await files.exists(rel))) out.push(rel);
    const { decoder: graph, embed } = info.parts[variant];
    for (const rel of await missingExternal(files, [graph.model, ...(embed ? [embed.model] : [])]))
      if (!out.includes(rel)) out.push(rel);
    return out;
  },
  include: (info, variant) => includePatterns(info, variant),
  async specialize(files, info, variant) {
    const picked = await pickVision(files, info, variant);
    if (!picked) return { info: { ...info, vision: null } };
    const target = visionTarget(info, variant);
    const preferred = target.dtype;
    const visionExtra = (await missingExternal(files, [target.model])).filter(
      (rel) => !rel.startsWith(target.model)
    );
    const dtype = picked.cached ? picked.part.dtype : 'missing';
    const report = {
      vision_encoder: dtype,
      ...(dtype === preferred ? {} : { vision_encoder_preferred: preferred }),
    };
    return {
      info: {
        ...info,
        vision: picked.cached ? picked.part : null,
        inputModalities: picked.cached ? ['text', 'image'] : ['text'],
        report,
        visionExtra,
      },
      key: `vision=${dtype}`,
      details: report,
    };
  },
  async load({ backend, files, variant, info, overrides = {} }) {
    const dir = info.tokenizerDir;
    const tokenizerJson = await files.json(`${dir}tokenizer.json`);
    const tokenizerConfig = await files.json(`${dir}tokenizer_config.json`);
    const template = await templateSource(files, info);
    const profile = pickProfile(template, overrides.profile);
    const tokenizer = new Tokenizer(profile.tokenizer(tokenizerJson), tokenizerConfig);
    const chosen =
      info.vision === undefined ? (await decoder.specialize(files, info, variant)).info : info;
    const image = Boolean(chosen.vision);
    const model = await openModel({
      backend,
      files,
      part: { ...info.parts[variant], vision: chosen.vision },
      shape: await shapeOf(files, info, variant),
    });
    const chat = createChat({
      model,
      tokenizer,
      tokenizerJson,
      bosToken: tokenName(tokenizerConfig.bos_token),
      eosToken: tokenName(tokenizerConfig.eos_token),
      template,
      profile,
      contextLength: overrides.contextLength ?? info.contextLength,
      eos: info.eos,
      defaults: samplingDefaults(info.generation, profile.sampling(template)),
      prefill: overrides.prefill,
      vision: image
        ? { ...(await visionOf(files, tokenizer, model)), report: chosen.report }
        : null,
    });
    return {
      profile: profile.name,
      chat: chat.chat,
      render: chat.render,
      score: chat.score,
      decode: chat.decode,
      cache: chat.cache,
      async release() {
        chat.cache.clear();
        await model.release();
      },
    };
  },
  checkOverrides: () => ({}),
  async check(instance, { fixtures, now = () => performance.now() } = {}) {
    const started = now();
    let text = '';
    const result = await instance.chat(
      {
        messages: [{ role: 'user', content: 'Say hello in five words.' }],
        tools: [],
        stop: [],
        maxTokens: 24,
        temperature: 0,
      },
      {
        emit(event) {
          if (event.type === 'content' || event.type === 'reasoning') text += event.text;
        },
      }
    );
    const smoke = { ok: text.length > 0, requestMs: now() - started, answer: text, ...result };
    if (!fixtures) return smoke;
    const scored = [];
    for (const f of fixtures) scored.push(await instance.score(fixtureRequest(f), f.tokens));
    const compared = compareLogprobs(scored, fixtures);
    return {
      ...smoke,
      ...compared,
      summary: `fixtures: ${compared.tokens} tokens, mean |dlogp| ${compared.mean.toFixed(3)}, max ${compared.max.toFixed(3)} (${compared.worstAt}), ${compared.flips.length} changed`,
      ok: smoke.ok && compared.ok,
    };
  },
};

export function fixtureRequest(f) {
  return {
    messages: f.messages,
    tools: [],
    stop: [],
    ...(f.reasoning_effort ? { reasoningEffort: f.reasoning_effort } : {}),
  };
}

const CLEAR = 1;

export function compareLogprobs(scored, fixtures) {
  let tokens = 0;
  let sum = 0;
  let max = 0;
  let worstAt = '';
  const flips = [];
  scored.forEach((steps, i) => {
    const f = fixtures[i];
    steps.forEach((step, j) => {
      const d = Math.abs(step.logprob - f.logprobs[j]);
      tokens++;
      sum += d;
      if (d > max) {
        max = d;
        worstAt = `${f.name}#${j}`;
      }
      if (step.top !== f.tokens[j]) flips.push({ at: `${f.name}#${j}`, margin: f.margins[j] });
    });
  });
  const mean = tokens ? sum / tokens : 0;
  const clear = flips.filter((x) => x.margin > CLEAR);
  return { tokens, mean, max, worstAt, flips, ok: mean <= 0.15 && clear.length === 0 };
}

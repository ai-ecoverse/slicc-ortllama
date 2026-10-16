import { Tokenizer } from '@huggingface/tokenizers';
import { createChat } from './chat.js';
import { detectLayout, includePatterns, neededFiles } from './layout.js';
import { pickProfile } from './profiles.js';
import { openModel } from './session.js';

const DEFAULT_TEMPERATURE = { mistral: 0.15 };

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

export const decoder = {
  name: 'decoder',
  capability: 'chat',
  async detect(files) {
    const layout = await detectLayout(files);
    if (!layout) return null;
    return { ...layout, inputModalities: ['text'], outputModalities: ['text'] };
  },
  async missing(files, info, variant) {
    const out = [];
    for (const rel of neededFiles(info, variant)) if (!(await files.exists(rel))) out.push(rel);
    return out;
  },
  include: (info, variant) => includePatterns(info, variant),
  async load({ backend, files, variant, info, overrides = {} }) {
    const dir = info.tokenizerDir;
    const tokenizerJson = await files.json(`${dir}tokenizer.json`);
    const tokenizerConfig = await files.json(`${dir}tokenizer_config.json`);
    const template = await templateSource(files, info);
    const profile = pickProfile(template, overrides.profile);
    const tokenizer = new Tokenizer(profile.tokenizer(tokenizerJson), tokenizerConfig);
    const model = await openModel({
      backend,
      files,
      part: info.parts[variant],
      shape: await shapeOf(files, info, variant),
    });
    const g = info.generation ?? {};
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
      defaults: {
        temperature: g.temperature ?? DEFAULT_TEMPERATURE[profile.name],
        topP: g.top_p,
        topK: g.top_k,
      },
      prefill: overrides.prefill,
    });
    return {
      profile: profile.name,
      chat: chat.chat,
      render: chat.render,
      cache: chat.cache,
      async release() {
        chat.cache.clear();
        await model.release();
      },
    };
  },
  checkOverrides: () => ({}),
  async check(instance, { now = () => performance.now() } = {}) {
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
          if (event.type === 'content') text += event.text;
        },
      }
    );
    return { ok: text.length > 0, requestMs: now() - started, answer: text, ...result };
  },
};

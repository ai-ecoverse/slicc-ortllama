import { LANGUAGES, languageCode } from './languages.js';
import { createMel } from './mel.js';

export class LanguageError extends RangeError {}

const RATE = 16000;
const PRECISION = 0.02;
const STRIDE = 2;

function abortError(signal) {
  return signal.reason instanceof Error
    ? signal.reason
    : new DOMException('the request was aborted', 'AbortError');
}

export function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export function floats(tensor) {
  const data = tensor.data;
  if (data instanceof Float32Array) return data;
  if (tensor.type === 'float16' && data instanceof Uint16Array)
    return Float32Array.from(data, halfToFloat);
  return Float32Array.from(data);
}

function typeOf(session, name) {
  const meta = session.inputMetadata?.find?.((m) => m.name === name);
  return meta?.type ?? 'float32';
}

function floatTensor(ort, type, data, dims) {
  if (type !== 'float16') return new ort.Tensor('float32', data, dims);
  return new ort.Tensor('float16', new globalThis.Float16Array(data), dims);
}

export function logSumExp(values, from = 0, to = values.length) {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = from; i < to; i++) if (values[i] > max) max = values[i];
  if (max === Number.NEGATIVE_INFINITY) return max;
  let sum = 0;
  for (let i = from; i < to; i++) sum += Math.exp(values[i] - max);
  return max + Math.log(sum);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function specialTokens(tokenizer, generation) {
  const id = (name) => tokenizer.id(name);
  const eot = generation.eos_token_id ?? id('<|endoftext|>');
  const noTimestamps = generation.no_timestamps_token_id ?? id('<|notimestamps|>');
  return {
    eot: Array.isArray(eot) ? eot[0] : eot,
    sot: generation.decoder_start_token_id ?? id('<|startoftranscript|>'),
    prev: generation.prev_sot_token_id ?? id('<|startofprev|>'),
    transcribe: generation.task_to_id?.transcribe ?? id('<|transcribe|>'),
    noSpeech: id('<|nospeech|>') ?? id('<|nocaptions|>'),
    noTimestamps,
    timestampBegin: noTimestamps + 1,
    languages: Object.fromEntries(
      Object.entries(generation.lang_to_id ?? {}).map(([token, value]) => [
        token.slice(2, -2),
        value,
      ])
    ),
  };
}

export function applyTimestampRules(logits, seq, ids, maxInitial) {
  const { timestampBegin: begin, eot, noTimestamps } = ids;
  logits[noTimestamps] = Number.NEGATIVE_INFINITY;
  const last = seq.length >= 1 && seq[seq.length - 1] >= begin;
  const penultimate = seq.length < 2 || seq[seq.length - 2] >= begin;
  if (last) {
    if (penultimate) logits.fill(Number.NEGATIVE_INFINITY, begin);
    else logits.fill(Number.NEGATIVE_INFINITY, 0, eot);
  }
  let lastStamp = -1;
  for (const t of seq) if (t >= begin) lastStamp = t;
  if (lastStamp >= 0) {
    const floor = last && !penultimate ? lastStamp : lastStamp + 1;
    logits.fill(Number.NEGATIVE_INFINITY, begin, floor);
  }
  if (seq.length === 0) {
    logits.fill(Number.NEGATIVE_INFINITY, 0, begin);
    logits.fill(Number.NEGATIVE_INFINITY, begin + maxInitial + 1);
  }
  const stamp = logSumExp(logits, begin);
  let text = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < begin; i++) if (logits[i] > text) text = logits[i];
  if (stamp > text) logits.fill(Number.NEGATIVE_INFINITY, 0, begin);
}

export function splitSegments(tokens, begin, offset, windowSeconds) {
  const stamp = (t) => t >= begin;
  const single = tokens.length >= 2 && !stamp(tokens.at(-2)) && stamp(tokens.at(-1));
  const cuts = [];
  for (let i = 1; i < tokens.length; i++)
    if (stamp(tokens[i - 1]) && stamp(tokens[i])) cuts.push(i);
  const segments = [];
  if (cuts.length > 0) {
    if (single) cuts.push(tokens.length);
    let from = 0;
    for (const cut of cuts) {
      const part = tokens.slice(from, cut);
      segments.push({
        start: offset + (part[0] - begin) * PRECISION,
        end: offset + (part.at(-1) - begin) * PRECISION,
        tokens: part,
      });
      from = cut;
    }
    const advance = single ? null : (tokens[from - 1] - begin) * STRIDE;
    return { segments, advance };
  }
  let duration = windowSeconds;
  const stamps = tokens.filter(stamp);
  if (stamps.length > 0 && stamps.at(-1) !== begin) duration = (stamps.at(-1) - begin) * PRECISION;
  if (tokens.some((t) => !stamp(t)))
    segments.push({ start: offset, end: offset + duration, tokens });
  return { segments, advance: null };
}

export function checkLanguage(language, multilingual, ids) {
  if (!language) return null;
  const code = languageCode(language);
  if (!code || (multilingual && ids.languages[code] === undefined))
    throw new LanguageError(`language ${language} is not supported by this model`);
  if (!multilingual && code !== 'en')
    throw new LanguageError('this model only transcribes English');
  return code;
}

function suppressedTokens(generation, ids) {
  const suppress = new Set(
    [...(generation.suppress_tokens ?? [])].filter((t) => t >= 0 && t < ids.eot)
  );
  for (let t = ids.eot + 1; t < ids.timestampBegin; t++) suppress.add(t);
  return suppress;
}

function createDecoder(ort, decoder, config) {
  const layers = config.decoder_layers;
  const heads = config.decoder_attention_heads;
  const headDim = config.d_model / heads;
  const empties = {};
  for (const kind of ['decoder', 'encoder']) {
    const half = typeOf(decoder, `past_key_values.0.${kind}.key`) === 'float16';
    empties[kind] = new ort.Tensor(
      half ? 'float16' : 'float32',
      half ? new Uint16Array(0) : new Float32Array(0),
      [1, heads, 0, headDim]
    );
  }
  return async function forward(tokens, hidden, past) {
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, tokens.length]),
      encoder_hidden_states: hidden,
      use_cache_branch: new ort.Tensor('bool', [past !== null], [1]),
    };
    for (let i = 0; i < layers; i++) {
      for (const kind of ['decoder', 'encoder']) {
        for (const part of ['key', 'value']) {
          feeds[`past_key_values.${i}.${kind}.${part}`] = past
            ? past[`present.${i}.${kind}.${part}`]
            : empties[kind];
        }
      }
    }
    const out = await decoder.run(feeds);
    const next = {};
    for (let i = 0; i < layers; i++) {
      for (const part of ['key', 'value']) {
        const enc = `present.${i}.encoder.${part}`;
        next[`present.${i}.decoder.${part}`] = out[`present.${i}.decoder.${part}`];
        next[enc] = past ? past[enc] : out[enc];
      }
    }
    return {
      logits: floats(out.logits),
      rows: tokens.length,
      vocab: out.logits.dims[2],
      past: next,
    };
  };
}

export function createWhisper({
  ort,
  encoder,
  decoder,
  tokenizer,
  config,
  generation,
  preprocessor,
}) {
  const mel = createMel({
    ...preprocessor,
    feature_size: preprocessor.feature_size ?? config.num_mel_bins,
  });
  const ids = specialTokens(tokenizer, generation);
  const textContext = config.max_target_positions ?? 448;
  const sampleLength = Math.floor(textContext / 2);
  const maxInitial = generation.max_initial_timestamp_index ?? 50;
  const multilingual = generation.is_multilingual ?? Object.keys(ids.languages).length > 0;
  const suppress = suppressedTokens(generation, ids);
  const beginSuppress = generation.begin_suppress_tokens ?? [];
  const inputName = encoder.inputNames[0];
  const inputType = typeOf(encoder, inputName);
  const forward = createDecoder(ort, decoder, config);

  async function encode(features) {
    const out = await encoder.run({
      [inputName]: floatTensor(ort, inputType, features, [1, mel.mels, mel.frames]),
    });
    return out[encoder.outputNames[0]];
  }

  async function detectLanguage(hidden) {
    const { logits, vocab } = await forward([ids.sot], hidden, null);
    const row = logits.subarray(0, vocab);
    let best = null;
    for (const [code, id] of Object.entries(ids.languages)) {
      if (best === null || row[id] > row[ids.languages[best]]) best = code;
    }
    return best;
  }

  async function decodeWindow(hidden, prefix, { timestamps, signal }) {
    const sotIndex = prefix.indexOf(ids.sot);
    let step = await forward(prefix, hidden, null);
    const sotRow = step.logits.subarray(sotIndex * step.vocab, (sotIndex + 1) * step.vocab);
    const noSpeechProb =
      ids.noSpeech === undefined ? 0 : Math.exp(sotRow[ids.noSpeech] - logSumExp(sotRow));
    const tokens = [];
    let sum = 0;
    const limit = Math.min(sampleLength, textContext - prefix.length);
    while (tokens.length < limit) {
      if (signal?.aborted) throw abortError(signal);
      const logits = step.logits.subarray((step.rows - 1) * step.vocab, step.rows * step.vocab);
      for (const t of suppress) logits[t] = Number.NEGATIVE_INFINITY;
      if (tokens.length === 0) for (const t of beginSuppress) logits[t] = Number.NEGATIVE_INFINITY;
      if (timestamps) applyTimestampRules(logits, tokens, ids, maxInitial);
      else logits.fill(Number.NEGATIVE_INFINITY, ids.timestampBegin);
      const next = argmax(logits);
      sum += logits[next] - logSumExp(logits);
      if (next === ids.eot) break;
      tokens.push(next);
      if (tokens.length >= limit) break;
      step = await forward([next], hidden, step.past);
    }
    return { tokens, noSpeechProb, avgLogprob: sum / (tokens.length + 1) };
  }

  function prefixFor(language, previous, timestamps) {
    const prompt = previous.length > 0 ? [ids.prev, ...previous.slice(-(sampleLength - 1))] : [];
    const task = multilingual ? [ids.languages[language], ids.transcribe] : [];
    return [...prompt, ids.sot, ...task, ...(timestamps ? [] : [ids.noTimestamps])];
  }

  function windowSegments(result, seek, windowFrames, end, first) {
    const offset = (seek * mel.hop) / RATE;
    const split = splitSegments(
      result.tokens,
      ids.timestampBegin,
      offset,
      (windowFrames * mel.hop) / RATE
    );
    const segments = [];
    for (const s of split.segments) {
      const words = tokenizer.decode(s.tokens.filter((t) => t < ids.eot));
      if (words.trim() === '') continue;
      segments.push({
        id: first + segments.length,
        seek,
        start: Math.min(s.start, end),
        end: Math.min(Math.max(s.end, s.start), end),
        text: words,
        tokens: s.tokens,
        temperature: 0,
        avg_logprob: result.avgLogprob,
        no_speech_prob: result.noSpeechProb,
      });
    }
    return { segments, advance: split.advance > 0 ? split.advance : windowFrames };
  }

  return {
    multilingual,
    languages: Object.keys(ids.languages),
    async transcribe(audio, { language = null, prompt = null, timestamps = true, signal } = {}) {
      let code = checkLanguage(language, multilingual, ids);
      const contentFrames = Math.ceil(audio.length / mel.hop);
      const end = audio.length / RATE;
      const promptTokens = prompt ? tokenizer.encode(` ${String(prompt).trim()}`) : [];
      const all = [];
      const segments = [];
      let seek = 0;
      while (seek < contentFrames) {
        if (signal?.aborted) throw abortError(signal);
        const windowFrames = Math.min(mel.frames, contentFrames - seek);
        const hidden = await encode(mel.compute(audio, seek * mel.hop));
        code ??= multilingual ? await detectLanguage(hidden) : 'en';
        const prefix = prefixFor(code, [...promptTokens, ...all], timestamps);
        const result = await decodeWindow(hidden, prefix, { timestamps, signal });
        if (result.noSpeechProb > 0.6 && result.avgLogprob < -1) {
          seek += windowFrames;
          continue;
        }
        const window = windowSegments(result, seek, windowFrames, end, segments.length);
        segments.push(...window.segments);
        all.push(...result.tokens);
        seek += window.advance;
      }
      return {
        text: segments
          .map((s) => s.text)
          .join('')
          .trim(),
        language: LANGUAGES[code] ?? code,
        duration: end,
        segments,
      };
    },
    async release() {
      await encoder.release?.();
      await decoder.release?.();
    },
  };
}

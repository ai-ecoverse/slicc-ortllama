import { checkLanguage, floats, logSumExp } from '../whisper/model.js';
import { createFeatures } from './features.js';

const WORD = '\u2581';
const PAUSE_SECONDS = 0.5;
const WINDOW_SECONDS = 30;
const SEARCH_SECONDS = 10;
const QUIET_SECONDS = 0.02;

function abortError(signal) {
  return signal.reason instanceof Error
    ? signal.reason
    : new DOMException('the request was aborted', 'AbortError');
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function pieceTable(tokenizer) {
  const pieces = [];
  for (const [piece, id] of Object.entries(tokenizer.model.vocab)) pieces[id] = piece;
  for (const added of tokenizer.added_tokens ?? []) pieces[added.id] = added.content;
  return pieces;
}

const special = (piece) => piece === undefined || (piece.startsWith('<') && piece.endsWith('>'));

export function detokenize(ids, pieces) {
  return ids
    .map((id) => pieces[id])
    .filter((piece) => !special(piece))
    .join('')
    .replaceAll(WORD, ' ')
    .trim();
}

export function createGreedy({ ort, session, config, generation }) {
  const blank = config.blank_token_id;
  const durations = config.durations;
  const hidden = config.decoder_hidden_size;
  const layers = config.num_decoder_layers;
  const maxSymbols = config.max_symbols_per_step ?? 10;
  const eos = new Set([generation.eos_token_id].flat().filter((id) => id !== undefined));
  const zeros = () =>
    new ort.Tensor('float32', new Float32Array(layers * hidden), [layers, 1, hidden]);

  return async function greedy(encoded, frames, length, signal) {
    const dim = encoded.length / frames;
    const limit = maxSymbols * frames - 1;
    const steps = [];
    let h = zeros();
    let c = zeros();
    let last = blank;
    let t = 0;
    while (t < length && steps.length < limit) {
      if (signal?.aborted) throw abortError(signal);
      const out = await session.run({
        encoder_outputs: new ort.Tensor('float32', encoded.subarray(t * dim, (t + 1) * dim), [
          1,
          1,
          dim,
        ]),
        input_ids: new ort.Tensor('int64', BigInt64Array.of(BigInt(last)), [1, 1]),
        state_h: h,
        state_c: c,
      });
      const logits = floats(out.token_logits);
      const token = argmax(logits);
      const predicted = durations[argmax(floats(out.duration_logits))];
      const duration = token === blank ? Math.max(predicted, 1) : predicted;
      if (token === blank) steps.push({ token, duration, frame: t });
      else {
        steps.push({ token, duration, frame: t, logprob: logits[token] - logSumExp(logits) });
        last = token;
        h = out.state_h_out;
        c = out.state_c_out;
      }
      t += duration;
      if (eos.has(token)) break;
    }
    return steps;
  };
}

export function windows(audio, rate, { seconds = WINDOW_SECONDS, search = SEARCH_SECONDS } = {}) {
  const size = seconds * rate;
  const quiet = Math.max(2, Math.round(rate * QUIET_SECONDS));
  const stride = Math.floor(quiet / 2);
  const out = [];
  let start = 0;
  while (audio.length - start > size) {
    let best = start + size - quiet;
    let quietest = Number.POSITIVE_INFINITY;
    for (let at = start + (seconds - search) * rate; at + quiet <= start + size; at += stride) {
      let energy = 0;
      for (let i = at; i < at + quiet; i++) energy += audio[i] * audio[i];
      if (energy < quietest) {
        quietest = energy;
        best = at;
      }
    }
    const cut = best + stride;
    out.push([start, cut]);
    start = cut;
  }
  out.push([start, audio.length]);
  return out;
}

export function wordsOf(tokens, pieces) {
  const words = [];
  for (const token of tokens) {
    const piece = pieces[token.id];
    if (special(piece)) continue;
    const word = words.at(-1);
    if (!word || piece.startsWith(WORD) || word.seek !== token.seek)
      words.push({ ...token, text: piece, tokens: [token] });
    else {
      word.text += piece;
      word.end = token.end;
      word.tokens.push(token);
    }
  }
  return words.map((w) => ({ ...w, text: w.text.replaceAll(WORD, ' ').trim() }));
}

export function segmentsOf(words, { pause = PAUSE_SECONDS } = {}) {
  const groups = [];
  for (const word of words) {
    const group = groups.at(-1);
    const previous = group?.at(-1);
    if (
      !previous ||
      previous.seek !== word.seek ||
      word.start - previous.end >= pause ||
      /[.?!]$/.test(previous.text)
    )
      groups.push([word]);
    else group.push(word);
  }
  return groups.map((group, id) => {
    const tokens = group.flatMap((w) => w.tokens);
    return {
      id,
      seek: group[0].seek,
      start: group[0].start,
      end: group.at(-1).end,
      text: ` ${group.map((w) => w.text).join(' ')}`,
      tokens: tokens.map((t) => t.id),
      temperature: 0,
      avg_logprob: tokens.reduce((sum, t) => sum + t.logprob, 0) / tokens.length,
    };
  });
}

export function createParakeet({
  ort,
  encoder,
  decoder,
  tokenizer,
  config,
  generation,
  preprocessor,
}) {
  const features = createFeatures(preprocessor);
  const pieces = pieceTable(tokenizer);
  const greedy = createGreedy({ ort, session: decoder, config, generation });
  const frameSeconds =
    ((config.encoder_config?.subsampling_factor ?? 8) * features.hop) / features.rate;

  async function encode(samples) {
    const { features: input, mask, frames } = features.compute(samples);
    const out = await encoder.run({
      input_features: new ort.Tensor('float32', input, [1, frames, features.mels]),
      attention_mask: new ort.Tensor('int64', mask, [1, frames]),
    });
    return {
      data: floats(out.encoder_outputs),
      frames: out.encoder_outputs.dims[1],
      length: Number(out.encoder_lengths.data[0]),
    };
  }

  async function steps(samples, signal) {
    const { data, frames, length } = await encode(samples);
    return greedy(data, frames, length, signal);
  }

  async function decodeWindow(audio, from, to, signal) {
    const shift = (from - features.lead) / features.rate;
    const start = from / features.rate;
    const end = to / features.rate;
    const at = (frame) => Math.min(end, Math.max(start, shift + frame * frameSeconds));
    const seek = Math.round(from / features.hop);
    return (await steps(audio.subarray(from, to), signal))
      .filter((s) => s.token !== config.blank_token_id)
      .map((s) => ({
        id: s.token,
        seek,
        start: at(s.frame),
        end: at(s.frame + Math.max(1, s.duration)),
        logprob: s.logprob,
      }));
  }

  return {
    multilingual: false,
    languages: ['en'],
    encode,
    steps: (samples, { signal } = {}) => steps(samples, signal),
    async transcribe(audio, { language = null, signal } = {}) {
      checkLanguage(language, false, null);
      const tokens = [];
      const texts = [];
      for (const [from, to] of windows(audio, features.rate)) {
        if (signal?.aborted) throw abortError(signal);
        const part = await decodeWindow(audio, from, to, signal);
        tokens.push(...part);
        texts.push(
          detokenize(
            part.map((t) => t.id),
            pieces
          )
        );
      }
      return {
        text: texts.filter(Boolean).join(' '),
        language: 'english',
        duration: audio.length / features.rate,
        segments: segmentsOf(wordsOf(tokens, pieces)),
      };
    },
    async release() {
      await encoder.release?.();
      await decoder.release?.();
    },
  };
}

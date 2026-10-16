import { createResampler, decodePcm16 } from '../audio/pcm.js';
import { errorBody, HttpError } from '../http/errors.js';
import { PRIORITY } from '../runtime.js';

const RATE = 16000;
const PARTIAL_SECONDS = 0.5;
const MIN_COMMIT_SECONDS = 0.1;

export const DEFAULT_VAD = {
  type: 'server_vad',
  threshold: 0.5,
  prefix_padding_ms: 300,
  silence_duration_ms: 500,
};

const words = (text) => text.split(/\s+/).filter(Boolean);

export function stablePrefix(a, b) {
  const x = words(a);
  const y = words(b);
  let n = 0;
  while (n < x.length && n < y.length && x[n] === y[n]) n++;
  return x.slice(0, n);
}

function randomId(prefix) {
  return `${prefix}_${globalThis.crypto.randomUUID().replaceAll('-', '').slice(0, 20)}`;
}

function invalid(message, param = null, code = null) {
  return new HttpError(400, message, { param, code });
}

function vadConfig(value) {
  if (value === null) return null;
  if (typeof value !== 'object' || Array.isArray(value))
    throw invalid('turn_detection must be an object or null', 'session.audio.input.turn_detection');
  if (value.type !== 'server_vad' && value.type !== 'semantic_vad')
    throw invalid(
      `turn_detection type ${value.type} is not supported`,
      'session.audio.input.turn_detection.type'
    );
  const out = { ...DEFAULT_VAD };
  for (const key of ['threshold', 'prefix_padding_ms', 'silence_duration_ms']) {
    if (value[key] === undefined) continue;
    if (typeof value[key] !== 'number' || !Number.isFinite(value[key]) || value[key] < 0)
      throw invalid(
        `${key} must be a non-negative number`,
        `session.audio.input.turn_detection.${key}`
      );
    out[key] = value[key];
  }
  return out;
}

class Turn {
  constructor(id, previous, start) {
    this.id = id;
    this.previous = previous;
    this.start = start;
    this.emitted = [];
    this.last = '';
    this.decodedAt = start;
  }
}

export class RealtimeSession {
  constructor({ send, catalog, runtime, defaultModel = 'realtime-transcribe', vadModel = 'vad' }) {
    this.send = send;
    this.catalog = catalog;
    this.runtime = runtime;
    this.vadModel = vadModel;
    this.id = randomId('sess');
    this.config = {
      rate: 24000,
      model: defaultModel,
      language: null,
      prompt: null,
      turnDetection: { ...DEFAULT_VAD },
    };
    this.controller = new AbortController();
    this.inbox = Promise.resolve();
    this.decodes = Promise.resolve();
    this.partial = null;
    this.whisper = null;
    this.held = null;
    this.vad = null;
    this.stream = null;
    this.resampler = createResampler(this.config.rate, RATE);
    this.received = 0;
    this.reset();
    this.previousItem = null;
    this.closed = false;
  }

  reset() {
    this.audio = new Float32Array(0);
    this.used = 0;
    this.offset = this.received;
    this.boundary = this.received;
    this.pending = new Float32Array(0);
    this.turn = null;
    this.silence = 0;
  }

  emit(type, fields = {}) {
    if (!this.closed) this.send({ type, event_id: randomId('event'), ...fields });
  }

  fail(error, clientEvent = null) {
    const { body } = errorBody(error);
    this.emit('error', { error: { ...body.error, event_id: clientEvent } });
  }

  describe() {
    const { rate, model, language, prompt, turnDetection } = this.config;
    return {
      type: 'transcription',
      id: this.id,
      object: 'realtime.transcription_session',
      audio: {
        input: {
          format: { type: 'audio/pcm', rate },
          transcription: { model, language, prompt },
          turn_detection: turnDetection,
        },
      },
    };
  }

  start() {
    this.emit('session.created', { session: this.describe() });
  }

  receive(text) {
    this.inbox = this.inbox.then(() => this.dispatch(text));
    return this.inbox;
  }

  async dispatch(text) {
    let event;
    try {
      event = JSON.parse(text);
    } catch {
      this.fail(invalid('the event is not valid JSON', null, 'invalid_json'));
      return;
    }
    try {
      if (typeof event !== 'object' || event === null)
        throw invalid('the event must be a JSON object');
      const handler = {
        'session.update': () => this.update(event.session),
        'input_audio_buffer.append': () => this.append(event.audio),
        'input_audio_buffer.commit': () => this.commit(),
        'input_audio_buffer.clear': () => this.clear(),
      }[event.type];
      if (!handler)
        throw invalid(`event type ${event.type} is not supported`, 'type', 'invalid_event');
      await handler();
    } catch (error) {
      this.fail(error, event?.event_id ?? null);
    }
  }

  async update(session) {
    if (typeof session !== 'object' || session === null)
      throw invalid('session must be an object', 'session');
    if (session.type !== undefined && session.type !== 'transcription')
      throw invalid('only transcription sessions are supported', 'session.type');
    const input = session.audio?.input ?? {};
    const next = { ...this.config };
    if (input.format !== undefined) {
      const { type, rate = 24000 } = input.format ?? {};
      if (type !== 'audio/pcm')
        throw invalid(
          `audio format ${type} is not supported; send audio/pcm`,
          'session.audio.input.format.type'
        );
      if (!Number.isInteger(rate) || rate < 8000 || rate > 48000)
        throw invalid(`audio rate ${rate} is not supported`, 'session.audio.input.format.rate');
      next.rate = rate;
    }
    const transcription = input.transcription ?? {};
    if (transcription.model !== undefined) next.model = transcription.model || next.model;
    if (transcription.language !== undefined) next.language = transcription.language || null;
    if (transcription.prompt !== undefined) next.prompt = transcription.prompt || null;
    if (input.turn_detection !== undefined) next.turnDetection = vadConfig(input.turn_detection);
    await this.prepare(next);
    if (next.rate !== this.config.rate) this.resampler = createResampler(next.rate, RATE);
    this.config = next;
    this.emit('session.updated', { session: this.describe() });
  }

  async prepare(config) {
    const resolved = await this.catalog.resolve(config.model);
    if (resolved.engine.capability !== 'transcription')
      throw invalid(
        `model ${resolved.id} does not transcribe audio`,
        'session.audio.input.transcription.model'
      );
    if (this.held?.key !== resolved.key) {
      const entry = await this.runtime.hold(resolved, this.controller.signal);
      if (this.held) this.runtime.unhold(this.held);
      this.held = entry;
    }
    this.whisper = resolved;
    if (config.turnDetection && !this.vad) {
      const vad = await this.catalog.resolve(this.vadModel);
      if (vad.engine.capability !== 'vad')
        throw invalid(`model ${vad.id} is not a voice activity detector`);
      const backend = await this.runtime.backend();
      const instance = await vad.engine.load({
        backend: {
          ...backend,
          executionProviders: backend.fallbackProviders,
        },
        files: vad.files,
        variant: vad.variant,
        info: vad.info,
      });
      if (this.closed) await instance.release?.();
      else this.vad = instance;
    }
    if (config.turnDetection && this.vad && !this.stream) this.stream = this.vad.stream();
    if (!config.turnDetection) this.stream = null;
  }

  async append(audio) {
    if (typeof audio !== 'string') throw invalid('audio must be a base64 string', 'audio');
    if (!this.whisper) await this.prepare(this.config);
    const bytes = Uint8Array.from(atob(audio), (c) => c.charCodeAt(0));
    const samples = this.resampler(decodePcm16(bytes, this.config.rate).samples);
    this.received += samples.length;
    this.push(samples);
    if (this.config.turnDetection) await this.detect(samples);
  }

  push(samples) {
    if (this.used + samples.length > this.audio.length) {
      const grown = new Float32Array(Math.max(2 * this.audio.length, this.used + samples.length));
      grown.set(this.audio.subarray(0, this.used));
      this.audio = grown;
    }
    this.audio.set(samples, this.used);
    this.used += samples.length;
  }

  ms(sample) {
    return Math.round((sample * 1000) / RATE);
  }

  async detect(samples) {
    const { frame } = this.vad;
    const vad = this.config.turnDetection;
    const joined = new Float32Array(this.pending.length + samples.length);
    joined.set(this.pending);
    joined.set(samples, this.pending.length);
    let at = 0;
    const end = this.received - joined.length;
    for (; at + frame <= joined.length; at += frame) {
      const probability = await this.stream.probability(joined.subarray(at, at + frame));
      const position = end + at + frame;
      if (!this.turn && probability >= vad.threshold) {
        const start = Math.max(
          this.offset,
          this.boundary,
          position - frame - Math.round((vad.prefix_padding_ms * RATE) / 1000)
        );
        this.turn = new Turn(randomId('item'), this.previousItem, start);
        this.silence = 0;
        this.emit('input_audio_buffer.speech_started', {
          audio_start_ms: this.ms(start),
          item_id: this.turn.id,
        });
      } else if (this.turn && probability < Math.max(0, vad.threshold - 0.15)) {
        this.silence += frame;
        if ((this.silence * 1000) / RATE >= vad.silence_duration_ms) this.finish(position);
      } else if (this.turn) this.silence = 0;
      if (this.turn && position - this.turn.decodedAt >= PARTIAL_SECONDS * RATE)
        this.refresh(position);
    }
    this.pending = joined.slice(at);
    if (!this.turn) this.trim();
  }

  trim() {
    const keep =
      Math.round((this.config.turnDetection.prefix_padding_ms * RATE) / 1000) + this.vad.frame;
    if (this.used > keep) {
      this.audio.copyWithin(0, this.used - keep, this.used);
      this.offset += this.used - keep;
      this.used = keep;
    }
  }

  slice(start, end) {
    const from = Math.max(0, start - this.offset);
    return this.audio.slice(from, Math.max(from, Math.min(this.used, end - this.offset)));
  }

  options(timestamps) {
    return {
      language: this.config.language,
      prompt: this.config.prompt,
      timestamps,
      signal: this.controller.signal,
    };
  }

  transcribe(audio, timestamps) {
    return this.runtime.use(
      this.whisper,
      PRIORITY.realtime,
      (model) => model.transcribe(audio, this.options(timestamps)),
      this.controller.signal
    );
  }

  refresh(position) {
    const turn = this.turn;
    turn.decodedAt = position;
    if (this.partial) return;
    const audio = this.slice(turn.start, position);
    this.partial = this.decodes
      .then(() => (turn.done ? null : this.transcribe(audio, false)))
      .then((result) => result && this.delta(turn, result.text))
      .catch((error) => this.closed || this.fail(error))
      .finally(() => {
        this.partial = null;
      });
  }

  delta(turn, text) {
    const stable = stablePrefix(turn.last, text);
    turn.last = text;
    this.extend(turn, stable);
  }

  extend(turn, next) {
    const same = turn.emitted.every((word, i) => next[i] === word);
    if (!same || next.length <= turn.emitted.length) return;
    const added = next.slice(turn.emitted.length).join(' ');
    const delta = turn.emitted.length > 0 ? ` ${added}` : added;
    turn.emitted = next;
    this.emit('conversation.item.input_audio_transcription.delta', {
      item_id: turn.id,
      content_index: 0,
      delta,
    });
  }

  finish(position) {
    const turn = this.turn;
    turn.done = true;
    this.turn = null;
    this.boundary = position;
    this.previousItem = turn.id;
    this.emit('input_audio_buffer.speech_stopped', {
      audio_end_ms: this.ms(position),
      item_id: turn.id,
    });
    this.emit('input_audio_buffer.committed', {
      item_id: turn.id,
      previous_item_id: turn.previous,
    });
    this.final(turn, this.slice(turn.start, position));
    this.silence = 0;
  }

  final(turn, audio) {
    const waitFor = this.partial ?? Promise.resolve();
    this.decodes = this.decodes
      .then(() => waitFor)
      .then(() => this.transcribe(audio, true))
      .then((result) => {
        this.extend(turn, words(result.text));
        this.emit('conversation.item.input_audio_transcription.completed', {
          item_id: turn.id,
          content_index: 0,
          transcript: result.text,
          usage: { type: 'duration', seconds: Math.ceil(audio.length / RATE) },
        });
      })
      .catch((error) => this.closed || this.fail(error));
  }

  commit() {
    if (this.turn) {
      this.finish(this.received);
      return;
    }
    const audio = this.config.turnDetection
      ? new Float32Array(0)
      : this.slice(this.offset, this.received);
    if (audio.length < MIN_COMMIT_SECONDS * RATE)
      throw invalid(
        'the input audio buffer has less than 100 ms of audio',
        null,
        'input_audio_buffer_commit_empty'
      );
    const turn = new Turn(randomId('item'), this.previousItem, this.offset);
    turn.done = true;
    this.previousItem = turn.id;
    this.emit('input_audio_buffer.committed', {
      item_id: turn.id,
      previous_item_id: turn.previous,
    });
    this.final(turn, audio);
    this.offset = this.received;
    this.used = 0;
  }

  clear() {
    if (this.turn) this.turn.done = true;
    this.reset();
    this.resampler = createResampler(this.config.rate, RATE);
    if (this.vad && this.config.turnDetection) this.stream = this.vad.stream();
    this.emit('input_audio_buffer.cleared');
  }

  async close() {
    this.closed = true;
    this.controller.abort();
    await this.inbox.catch(() => {});
    await this.decodes.catch(() => {});
    if (this.held) this.runtime.unhold(this.held);
    this.held = null;
    await this.vad?.release?.();
    this.vad = null;
  }
}

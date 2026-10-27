import { decodeFile } from './audio/decode.js';
import { AudioFormatError } from './audio/pcm.js';
import { routeHint } from './chat.js';
import { badRequest } from './http/errors.js';
import { parseMultipart } from './http/multipart.js';
import { PRIORITY } from './runtime.js';

const FORMATS = new Set(['json', 'verbose_json', 'text', 'srt', 'vtt']);

const round = (x, places = 2) => Math.round(x * 10 ** places) / 10 ** places;

function clock(seconds, separator) {
  const ms = Math.round(seconds * 1000);
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor(ms / 60_000) % 60;
  const s = Math.floor(ms / 1000) % 60;
  return `${pad(h)}:${pad(m)}:${pad(s)}${separator}${pad(ms % 1000, 3)}`;
}

function cues(segments, separator) {
  return segments.map(
    (s, i) =>
      `${separator === ',' ? `${i + 1}\n` : ''}${clock(s.start, separator)} --> ${clock(s.end, separator)}\n${s.text.trim()}\n`
  );
}

export function formatTranscript(result, format) {
  const usage = { type: 'duration', seconds: Math.ceil(result.duration) };
  const duration = round(result.duration, 3);
  if (format === 'text') return { body: `${result.text}\n` };
  if (format === 'srt') return { body: cues(result.segments, ',').join('\n') };
  if (format === 'vtt') return { body: ['WEBVTT\n', ...cues(result.segments, '.')].join('\n') };
  if (format === 'verbose_json')
    return {
      task: 'transcribe',
      language: result.language,
      duration,
      text: result.text,
      segments: result.segments.map((s) => ({ ...s, start: round(s.start), end: round(s.end) })),
      usage,
    };
  return {
    text: result.text,
    language: result.language,
    duration,
    segments: result.segments.map((s) => ({
      id: s.id,
      start: round(s.start),
      end: round(s.end),
      text: s.text,
    })),
    usage,
  };
}

export function transcriptionRoutes({ catalog, runtime, aac, decode = decodeFile }) {
  return {
    'POST /v1/audio/transcriptions': {
      raw: true,
      async handle({ req, body, signal }) {
        const { fields, files } = parseMultipart(body, req.headers['content-type']);
        const file = files.file;
        if (!file) throw badRequest('file is required', 'file');
        const format = fields.response_format || 'json';
        if (!FORMATS.has(format))
          throw badRequest(`response_format ${format} is not supported`, 'response_format');
        const resolved = await catalog.resolve(fields.model || 'transcribe');
        if (resolved.engine.capability !== 'transcription')
          throw badRequest(
            `model ${resolved.id} does not transcribe audio${routeHint(resolved)}`,
            'model'
          );
        let audio;
        try {
          audio = await decode(file.data, { type: file.type, name: file.filename, aac });
        } catch (error) {
          if (!(error instanceof AudioFormatError)) throw error;
          throw badRequest(error.message, 'file');
        }
        if (audio.length === 0) throw badRequest('the audio file has no samples', 'file');
        const options = {
          language: fields.language || null,
          prompt: fields.prompt || null,
          signal,
        };
        const result = await runtime.use(
          resolved,
          PRIORITY.speech,
          (model) => model.transcribe(audio, options),
          signal
        );
        return formatTranscript(result, format);
      },
    },
  };
}

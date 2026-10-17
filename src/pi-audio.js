import { DEFAULT_PORT } from './config.js';
import { VOICES, voiceInfo } from './kokoro/voices.js';

export const DEFAULT_BASE_URL = `http://127.0.0.1:${DEFAULT_PORT}/v1`;

const tokenCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const pcm24 = { encoding: 'pcm16', sampleRate: 24000, channels: 1 };
const wav16 = { encoding: 'wav', sampleRate: 16000, channels: 1 };
const wav24 = { encoding: 'wav', sampleRate: 24000, channels: 1 };
const coded = (encoding, sampleRate) => ({ encoding, sampleRate, channels: 1 });

export const SPEECH_FORMATS = [
  pcm24,
  wav24,
  ...['mp3', 'opus', 'flac'].map((e) => coded(e, 24000)),
];

function base(id, name, baseUrl, api) {
  return { id, name: name ?? id, api, provider: 'ortllama', baseUrl, cost: tokenCost };
}

export function transcriptionModel({ id = 'transcribe', name, baseUrl = DEFAULT_BASE_URL } = {}) {
  return {
    ...base(id, name, baseUrl, 'openai-transcriptions'),
    type: 'transcription',
    streaming: false,
    turnDetection: 'none',
    inputFormats: [wav16, pcm24, ...['mp3', 'flac', 'opus'].map((e) => coded(e, 16000))],
    audioCost: { perInputMinute: 0 },
  };
}

export function realtimeTranscriptionModel({
  id = 'realtime-transcribe',
  name,
  baseUrl = DEFAULT_BASE_URL,
} = {}) {
  return {
    ...base(id, name, baseUrl, 'openai-realtime-transcription'),
    type: 'transcription',
    streaming: true,
    turnDetection: 'vad',
    inputFormats: [pcm24],
    audioCost: { perInputMinute: 0 },
  };
}

export function speechModel({
  id = 'speech',
  name = 'Kokoro-82M',
  baseUrl = DEFAULT_BASE_URL,
  voices = VOICES,
} = {}) {
  return {
    ...base(id, name, baseUrl, 'openai-speech'),
    type: 'speech',
    textStreaming: false,
    outputFormats: SPEECH_FORMATS,
    voices: voices.map((v) => (typeof v === 'string' ? voiceInfo(v) : v)),
    cloning: false,
    audioCost: {},
  };
}

export function voiceModels(baseUrl = DEFAULT_BASE_URL) {
  return [
    speechModel({ baseUrl }),
    transcriptionModel({ baseUrl }),
    realtimeTranscriptionModel({ baseUrl }),
  ];
}

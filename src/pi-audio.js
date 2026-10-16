import { DEFAULT_PORT } from './config.js';

export const DEFAULT_BASE_URL = `http://127.0.0.1:${DEFAULT_PORT}/v1`;

const tokenCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const pcm24 = { encoding: 'pcm16', sampleRate: 24000, channels: 1 };
const wav16 = { encoding: 'wav', sampleRate: 16000, channels: 1 };

function base(id, name, baseUrl, api) {
  return { id, name: name ?? id, api, provider: 'ortllama', baseUrl, cost: tokenCost };
}

export function transcriptionModel({ id = 'transcribe', name, baseUrl = DEFAULT_BASE_URL } = {}) {
  return {
    ...base(id, name, baseUrl, 'openai-transcriptions'),
    type: 'transcription',
    streaming: false,
    turnDetection: 'none',
    inputFormats: [wav16, pcm24],
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

import { createFft } from '../whisper/mel.js';

const f32 = Math.fround;

const SPACING = 200 / 3;
const LOG_HZ = 1000;
const LOG_MEL = LOG_HZ / SPACING;
const LOG_STEP = Math.log(6.4) / 27;

const hzToMel = (hz) => (hz >= LOG_HZ ? LOG_MEL + Math.log(hz / LOG_HZ) / LOG_STEP : hz / SPACING);
const melToHz = (mel) =>
  mel >= LOG_MEL ? LOG_HZ * Math.exp(LOG_STEP * (mel - LOG_MEL)) : SPACING * mel;

export function slaneyFilters(nFft, mels, rate, fmin, fmax) {
  const bins = nFft / 2 + 1;
  const spacing = 1 / (nFft * (1 / rate));
  const low = hzToMel(fmin);
  const high = hzToMel(fmax);
  const step = (high - low) / (mels + 1);
  const edges = Array.from({ length: mels + 2 }, (_, i) =>
    melToHz(i === mels + 1 ? high : i * step + low)
  );
  const out = new Float32Array(mels * bins);
  for (let m = 0; m < mels; m++) {
    const norm = 2 / (edges[m + 2] - edges[m]);
    for (let b = 0; b < bins; b++) {
      const f = b * spacing;
      const lower = -(edges[m] - f) / (edges[m + 1] - edges[m]);
      const upper = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
      out[m * bins + b] = f32(Math.max(0, Math.min(lower, upper))) * norm;
    }
  }
  return out;
}

function sparseFilters(dense, mels, bins) {
  const rows = [];
  for (let m = 0; m < mels; m++) {
    const row = dense.subarray(m * bins, (m + 1) * bins);
    let from = 0;
    while (from < bins && row[from] === 0) from++;
    let to = bins;
    while (to > from && row[to - 1] === 0) to--;
    rows.push({ from, weights: row.slice(from, to) });
  }
  return rows;
}

export function createFeatures({
  feature_size: mels = 128,
  sampling_rate: rate = 16000,
  hop_length: hop = 160,
  n_fft: nFft = 512,
  win_length: win = 400,
  preemphasis = 0.97,
  leading_silence_seconds: leadSeconds = 0,
  log_zero_guard_value: guard = 2 ** -24,
  normalize_epsilon: epsilon = 1e-5,
  mel_fmin: fmin = 0,
  mel_fmax: fmax = rate / 2,
} = {}) {
  const bins = nFft / 2 + 1;
  const filters = sparseFilters(slaneyFilters(nFft, mels, rate, fmin, fmax), mels, bins);
  const offset = Math.floor((nFft - win) / 2);
  const window = new Float64Array(nFft);
  for (let i = 0; i < win; i++)
    window[offset + i] = f32(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (win - 1)));
  const fft = createFft(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const frame = new Float64Array(nFft);
  const power = new Float64Array(bins);
  const half = nFft / 2;
  const lead = Math.round(leadSeconds * rate);
  const coefficient = f32(preemphasis);
  const floor = f32(guard);
  const epsilon32 = f32(epsilon);

  function melFrame(signal, t, out) {
    const base = t * hop - half;
    for (let j = 0; j < nFft; j++) {
      const k = base + j;
      frame[j] = k >= 0 && k < signal.length ? f32(signal[k] * window[j]) : 0;
    }
    fft(frame, re, im);
    for (let b = 0; b < bins; b++) power[b] = f32(re[b] * re[b] + im[b] * im[b]);
    for (let m = 0; m < mels; m++) {
      const { from, weights } = filters[m];
      let sum = 0;
      for (let b = 0; b < weights.length; b++) sum = f32(sum + weights[b] * power[from + b]);
      out[t * mels + m] = Math.log(f32(sum + floor));
    }
  }

  function normalize(features, frames, valid) {
    const count = f32(Math.max(1, valid));
    const divisor = f32(Math.max(1, valid - 1));
    for (let m = 0; m < mels; m++) {
      let total = 0;
      for (let t = 0; t < valid; t++) total = f32(total + features[t * mels + m]);
      const mean = f32(total / count);
      let square = 0;
      for (let t = 0; t < valid; t++) {
        const d = f32(features[t * mels + m] - mean);
        square = f32(square + f32(d * d));
      }
      const scale = f32(f32(Math.sqrt(f32(square / divisor))) + epsilon32);
      for (let t = 0; t < frames; t++)
        features[t * mels + m] = t < valid ? f32(features[t * mels + m] - mean) / scale : 0;
    }
  }

  return {
    mels,
    rate,
    hop,
    lead,
    compute(audio) {
      const length = lead + audio.length;
      const signal = new Float32Array(length);
      signal.set(audio, lead);
      for (let n = length - 1; n > 0; n--) signal[n] -= f32(coefficient * signal[n - 1]);
      const frames = 1 + Math.floor(length / hop);
      const valid = Math.floor(length / hop);
      const features = new Float32Array(frames * mels);
      for (let t = 0; t < frames; t++) melFrame(signal, t, features);
      normalize(features, frames, valid);
      const mask = new BigInt64Array(frames);
      mask.fill(1n, 0, valid);
      return { features, mask, frames, valid };
    },
  };
}

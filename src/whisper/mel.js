export function factorize(n) {
  const factors = [];
  let p = 4;
  let rest = n;
  while (rest > 1) {
    while (rest % p !== 0) {
      p = p === 4 ? 2 : p === 2 ? 3 : p + 2;
      if (p * p > rest) p = rest;
    }
    rest /= p;
    factors.push(p, rest);
  }
  return factors;
}

export function createFft(n) {
  const factors = factorize(n);
  const twRe = new Float64Array(n);
  const twIm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    twRe[i] = Math.cos((-2 * Math.PI * i) / n);
    twIm[i] = Math.sin((-2 * Math.PI * i) / n);
  }
  const maxP = Math.max(...factors.filter((_, i) => i % 2 === 0));
  const sRe = new Float64Array(maxP);
  const sIm = new Float64Array(maxP);

  function butterfly(outRe, outIm, o, fstride, m, p) {
    for (let u = 0; u < m; u++) {
      for (let q = 0, k = u; q < p; q++, k += m) {
        sRe[q] = outRe[o + k];
        sIm[q] = outIm[o + k];
      }
      for (let q1 = 0, k = u; q1 < p; q1++, k += m) {
        let re = sRe[0];
        let im = sIm[0];
        let t = 0;
        for (let q = 1; q < p; q++) {
          t += fstride * k;
          if (t >= n) t -= n;
          re += sRe[q] * twRe[t] - sIm[q] * twIm[t];
          im += sRe[q] * twIm[t] + sIm[q] * twRe[t];
        }
        outRe[o + k] = re;
        outIm[o + k] = im;
      }
    }
  }

  function work(outRe, outIm, o, input, i, fstride, f) {
    const p = factors[f];
    const m = factors[f + 1];
    if (m === 1) {
      for (let j = 0; j < p; j++) {
        outRe[o + j] = input[i + j * fstride];
        outIm[o + j] = 0;
      }
    } else {
      for (let j = 0; j < p; j++)
        work(outRe, outIm, o + j * m, input, i + j * fstride, fstride * p, f + 2);
    }
    butterfly(outRe, outIm, o, fstride, m, p);
  }

  return (input, outRe, outIm) => work(outRe, outIm, 0, input, 0, 1, 0);
}

const melToHz = (mel) =>
  mel < 15 ? (200 * mel) / 3 : 1000 * Math.exp((Math.log(6.4) / 27) * (mel - 15));

export function melFilters(bins, mels, rate) {
  const top = 15 + (27 * Math.log(8)) / Math.log(6.4);
  const freqs = Array.from({ length: mels + 2 }, (_, i) => melToHz((top * i) / (mels + 1)));
  const out = new Float32Array(mels * bins);
  const nyquist = Math.floor(rate / 2);
  for (let m = 0; m < mels; m++) {
    const enorm = 2 / (freqs[m + 2] - freqs[m]);
    for (let b = 0; b < bins; b++) {
      const f = (nyquist * b) / (bins - 1);
      const down = (f - freqs[m]) / (freqs[m + 1] - freqs[m]);
      const up = (freqs[m + 2] - f) / (freqs[m + 2] - freqs[m + 1]);
      out[m * bins + b] = Math.max(0, Math.min(down, up)) * enorm;
    }
  }
  return out;
}

export function createMel({
  feature_size: mels = 80,
  sampling_rate: rate = 16000,
  hop_length: hop = 160,
  n_fft: nFft = 400,
  nb_max_frames: frames = 3000,
} = {}) {
  const bins = nFft / 2 + 1;
  const filters = melFilters(bins, mels, rate);
  const window = Float64Array.from(
    { length: nFft },
    (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft)
  );
  const fft = createFft(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const frame = new Float64Array(nFft);
  const power = new Float64Array(bins);
  const samples = frames * hop;
  const half = nFft / 2;

  return {
    mels,
    frames,
    samples,
    hop,
    rate,
    compute(audio, start = 0) {
      const length = Math.max(0, Math.min(samples, audio.length - start));
      const at = (j) => {
        let k = j - half;
        if (k < 0) k = -k;
        if (k >= samples) k = 2 * (samples - 1) - k;
        return k < length ? audio[start + k] : 0;
      };
      const out = new Float32Array(mels * frames);
      let top = Number.NEGATIVE_INFINITY;
      for (let t = 0; t < frames; t++) {
        const base = t * hop;
        for (let j = 0; j < nFft; j++) frame[j] = at(base + j) * window[j];
        fft(frame, re, im);
        for (let b = 0; b < bins; b++) power[b] = re[b] * re[b] + im[b] * im[b];
        for (let m = 0; m < mels; m++) {
          let sum = 0;
          const row = m * bins;
          for (let b = 0; b < bins; b++) sum += filters[row + b] * power[b];
          const value = Math.log10(Math.max(sum, 1e-10));
          out[m * frames + t] = value;
          if (value > top) top = value;
        }
      }
      for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], top - 8) + 4) / 4;
      return out;
    },
  };
}

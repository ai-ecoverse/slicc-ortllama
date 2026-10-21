const plans = new Map();

function plan(n) {
  const m = n >> 1;
  const half = m >> 1;
  const bits = Math.log2(half);
  const reverse = new Uint32Array(half);
  for (let i = 0; i < half; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    reverse[i] = r;
  }
  const cos = new Float64Array(half >> 1);
  const sin = new Float64Array(half >> 1);
  for (let i = 0; i < half >> 1; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / half);
    sin[i] = -Math.sin((2 * Math.PI * i) / half);
  }
  const twistCos = new Float64Array(half);
  const twistSin = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    twistCos[k] = Math.cos((Math.PI * (8 * k + 1)) / (8 * m));
    twistSin[k] = -Math.sin((Math.PI * (8 * k + 1)) / (8 * m));
  }
  return {
    m,
    half,
    reverse,
    cos,
    sin,
    twistCos,
    twistSin,
    re: new Float64Array(half),
    im: new Float64Array(half),
    u: new Float64Array(m),
  };
}

function fft(p) {
  const { half, re, im, cos, sin } = p;
  for (let size = 2; size <= half; size <<= 1) {
    const step = half / size;
    const span = size >> 1;
    for (let start = 0; start < half; start += size) {
      for (let j = 0; j < span; j++) {
        const wr = cos[j * step];
        const wi = sin[j * step];
        const a = start + j;
        const b = a + span;
        const tr = re[b] * wr - im[b] * wi;
        const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

const direct = new Map();

function matrix(n) {
  let rows = direct.get(n);
  if (!rows) {
    const m = n >> 1;
    rows = new Float64Array(n * m);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < m; k++)
        rows[i * m + k] = Math.cos(((2 * Math.PI) / n) * (i + 0.5 + n / 4) * (k + 0.5));
    direct.set(n, rows);
  }
  return rows;
}

function slow(input, output, n) {
  const m = n >> 1;
  const q = m >> 1;
  const rows = matrix(n);
  for (let i = 0; i < n; i++) {
    if ((i >= q && i < m) || i >= m + q) continue;
    let sum = 0;
    for (let k = 0, at = i * m; k < m; k++) sum += input[k] * rows[at + k];
    output[i] = sum;
    if (i < q) output[m - 1 - i] = -sum;
    else output[3 * m - 1 - i] = sum;
  }
  return output;
}

export function mdct(input, output, n) {
  const m = n >> 1;
  const rows = matrix(n);
  for (let k = 0; k < m; k++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += input[i] * rows[i * m + k];
    output[k] = sum;
  }
  return output;
}

export function imdct(input, output, n) {
  if (!Number.isInteger(Math.log2(n))) return slow(input, output, n);
  let p = plans.get(n);
  if (!p) {
    p = plan(n);
    plans.set(n, p);
  }
  const { m, half, reverse, twistCos, twistSin, re, im, u } = p;
  for (let k = 0; k < half; k++) {
    const a = input[2 * k];
    const b = input[m - 1 - 2 * k];
    const at = reverse[k];
    re[at] = a * twistCos[k] - b * twistSin[k];
    im[at] = a * twistSin[k] + b * twistCos[k];
  }
  fft(p);
  for (let k = 0; k < half; k++) {
    const r = re[k] * twistCos[k] - im[k] * twistSin[k];
    const i = re[k] * twistSin[k] + im[k] * twistCos[k];
    u[2 * k] = r;
    u[m - 1 - 2 * k] = -i;
  }
  const q = m >> 1;
  for (let i = 0; i < q; i++) output[i] = u[i + q];
  for (let i = q; i < 3 * q; i++) output[i] = -u[3 * q - 1 - i];
  for (let i = 3 * q; i < 2 * m; i++) output[i] = -u[i - 3 * q];
  return output;
}

export const LOW_DETAIL_EDGE = 512;

export function pixtralConfig(pre, config = {}) {
  return {
    patch: pre.patch_size,
    merge: pre.spatial_merge_size ?? config.spatial_merge_size,
    longest: pre.size.longest_edge,
    mean: pre.image_mean,
    std: pre.image_std,
    rescale: pre.rescale_factor,
  };
}

export function targetSize(height, width, { patch, merge }, longest) {
  let h = height;
  let w = width;
  const ratio = Math.max(h / longest, w / longest);
  if (ratio > 1) {
    h = Math.floor(h / ratio);
    w = Math.floor(w / ratio);
  }
  const unit = patch * merge;
  return [Math.ceil(h / unit) * unit, Math.ceil(w / unit) * unit];
}

function cubic(t, a = -0.5) {
  const x = Math.abs(t);
  if (x <= 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}

function taps(inSize, outSize) {
  const scale = inSize / outSize;
  const support = scale > 1 ? 2 * scale : 2;
  const inv = scale > 1 ? 1 / scale : 1;
  const out = [];
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const end = Math.min(inSize, Math.trunc(center + support + 0.5));
    const w = [];
    let sum = 0;
    for (let j = start; j < end; j++) {
      const k = cubic((j - center + 0.5) * inv);
      w.push(k);
      sum += k;
    }
    out.push({ start, w: w.map((k) => k / sum) });
  }
  return out;
}

const clamp = (v) => Math.min(255, Math.max(0, Math.round(v)));

function pass(src, lines, length, outLength, stride, step) {
  const t = taps(length, outLength);
  const out = new Uint8Array(lines * outLength * 3);
  for (let line = 0; line < lines; line++)
    for (let i = 0; i < outLength; i++) {
      const { start, w } = t[i];
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < w.length; k++) s += src[stride(line, start + k) + c] * w[k];
        out[step(line, i) + c] = clamp(s);
      }
    }
  return out;
}

export function resizeRgb(src, h, w, H, W) {
  const wide =
    W === w
      ? src
      : pass(
          src,
          h,
          w,
          W,
          (y, x) => (y * w + x) * 3,
          (y, x) => (y * W + x) * 3
        );
  if (H === h) return wide;
  return pass(
    wide,
    W,
    h,
    H,
    (x, y) => (y * W + x) * 3,
    (x, y) => (y * W + x) * 3
  );
}

export function flatten(image) {
  const n = image.width * image.height;
  const rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) rgb[i * 3 + c] = image.data[i * 4 + c];
  return rgb;
}

export function preprocess(image, cfg, longest = cfg.longest) {
  const [H, W] = targetSize(image.height, image.width, cfg, longest);
  const rgb = resizeRgb(flatten(image), image.height, image.width, H, W);
  const plane = H * W;
  const data = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < plane; i++)
      data[c * plane + i] = (rgb[i * 3 + c] * cfg.rescale - cfg.mean[c]) / cfg.std[c];
  const unit = cfg.patch * cfg.merge;
  return { data, height: H, width: W, rows: H / unit, cols: W / unit };
}

export function imageTokens(rows, cols, ids) {
  const out = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) out.push(ids.img);
    out.push(r === rows - 1 ? ids.end : ids.brk);
  }
  return out;
}

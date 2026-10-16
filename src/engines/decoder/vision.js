import { badRequest } from '../../http/errors.js';
import { imageTokens, LOW_DETAIL_EDGE, preprocess, targetSize } from './pixtral.js';
import { hasImages } from './profiles.js';

export function requestImages(messages) {
  const out = [];
  for (const m of messages)
    if (m.role === 'user' && hasImages(m.content))
      for (const part of m.content) if (part.type === 'image') out.push(part.image);
  return out;
}

export function expandImages(ids, images, vision) {
  if (images.length === 0) return { ids, keys: ids, spans: [] };
  const out = [];
  const keys = [];
  const spans = [];
  const unit = vision.cfg.patch * vision.cfg.merge;
  let next = 0;
  for (const id of ids) {
    if (id !== vision.ids.img) {
      out.push(id);
      keys.push(id);
      continue;
    }
    const image = images[next++];
    if (!image)
      throw badRequest('the chat template placed more images than the request has', 'messages');
    const longest =
      image.detail === 'low' ? Math.min(LOW_DETAIL_EDGE, vision.cfg.longest) : vision.cfg.longest;
    const [H, W] = targetSize(image.height, image.width, vision.cfg, longest);
    if (H === 0 || W === 0)
      throw badRequest(
        `an image of ${image.width}x${image.height} is too thin for this model's vision encoder`,
        'messages'
      );
    const tokens = imageTokens(H / unit, W / unit, vision.ids);
    spans.push({
      start: out.length,
      end: out.length + tokens.length,
      image,
      longest,
      rows: H / unit,
      cols: W / unit,
      features: null,
    });
    tokens.forEach((t, k) => {
      out.push(t);
      keys.push(`${image.hash}:${longest}:${k}`);
    });
  }
  if (next !== images.length)
    throw badRequest(`the chat template placed ${next} of ${images.length} images`, 'messages');
  return { ids: out, keys, spans };
}

function fill(span, embeds, dim, from, to) {
  if (span.features.length !== span.rows * span.cols * dim)
    throw new Error(
      `the vision encoder returned ${span.features.length / dim} rows for ${span.rows * span.cols} image tokens`
    );
  const width = span.cols + 1;
  for (let p = Math.max(from, span.start); p < Math.min(to, span.end); p++) {
    const k = p - span.start;
    const col = k % width;
    if (col === span.cols) continue;
    const row = Math.floor(k / width) * span.cols + col;
    embeds.set(span.features.subarray(row * dim, (row + 1) * dim), (p - from) * dim);
  }
}

export async function splicer(spans, from, to, vision, timings) {
  const hit = spans.filter((s) => s.start < to && s.end > from);
  if (hit.length === 0) return null;
  for (const span of hit) {
    if (span.features) continue;
    const started = performance.now();
    span.features = await vision.encode(preprocess(span.image, vision.cfg, span.longest));
    timings.images++;
    timings.ms += performance.now() - started;
  }
  return (embeds, dim) => {
    for (const span of hit) fill(span, embeds, dim, from, to);
  };
}

import { AudioFormatError } from './pcm.js';

const STARTS = new Set(['ftyp', 'styp', 'moov']);
const TRAVERSE = new Set(['moov', 'trak', 'mdia', 'minf', 'stbl', 'edts', 'mvex', 'moof', 'traf']);

function corrupt() {
  return new AudioFormatError('the MP4 file is corrupt');
}

const text = (bytes, at) => String.fromCharCode(...bytes.subarray(at, at + 4));

export function isMp4(bytes) {
  return bytes.length >= 12 && STARTS.has(text(bytes, 4));
}

function* boxes(bytes, view, from, to) {
  for (let at = from; at + 8 <= to; ) {
    let size = view.getUint32(at);
    let body = at + 8;
    if (size === 1) {
      if (at + 16 > to) throw corrupt();
      size = Number(view.getBigUint64(at + 8));
      body = at + 16;
    } else if (size === 0) size = to - at;
    if (size < body - at) throw corrupt();
    const end = Math.min(at + size, to);
    yield { type: text(bytes, at + 4), at, body, end };
    at = end;
  }
}

function full(view, box) {
  return { version: view.getUint8(box.body), flags: view.getUint32(box.body) & 0xffffff };
}

function descriptor(bytes, at) {
  let length = 0;
  let next = at + 1;
  for (let i = 0; i < 4; i++) {
    const b = bytes[next++];
    length = length * 128 + (b & 0x7f);
    if (!(b & 0x80)) break;
  }
  return { tag: bytes[at], body: next, end: next + length };
}

export function esdsConfig(bytes, from, to) {
  const es = descriptor(bytes, from);
  if (es.tag !== 3) throw corrupt();
  const flags = bytes[es.body + 2];
  let at = es.body + 3 + (flags & 0x80 ? 2 : 0);
  if (flags & 0x40) at += 1 + bytes[at];
  at += flags & 0x20 ? 2 : 0;
  const dc = descriptor(bytes, at);
  if (dc.tag !== 4) throw corrupt();
  const object = bytes[dc.body];
  for (let d = dc.body + 13; d < Math.min(dc.end, to); ) {
    const info = descriptor(bytes, d);
    if (info.tag === 5) return { object, config: bytes.slice(info.body, info.end) };
    d = info.end;
  }
  return { object, config: null };
}

function sampleEntry(bytes, view, stsd) {
  const [entry] = boxes(bytes, view, stsd.body + 8, stsd.end);
  if (!entry) throw corrupt();
  const version = view.getUint16(entry.body + 8);
  const start = entry.body + 28 + (version === 1 ? 16 : version === 2 ? 36 : 0);
  const found = { codec: entry.type, esds: null };
  const search = (from, to) => {
    for (const child of boxes(bytes, view, from, to)) {
      if (child.type === 'esds') found.esds = esdsConfig(bytes, child.body + 4, child.end);
      else if (child.type === 'wave') search(child.body, child.end);
    }
  };
  search(Math.min(start, entry.end), entry.end);
  return found;
}

function table(view, box, width) {
  const count = view.getUint32(box.body + 4);
  if (box.body + 8 + count * width > box.end) throw corrupt();
  return { count, at: box.body + 8 };
}

function sizes(view, box) {
  const fixed = view.getUint32(box.body + 4);
  const count = view.getUint32(box.body + 8);
  if (fixed) return new Array(count).fill(fixed);
  if (box.body + 12 + count * 4 > box.end) throw corrupt();
  return Array.from({ length: count }, (_, i) => view.getUint32(box.body + 12 + i * 4));
}

function offsets(view, box) {
  const width = box.type === 'co64' ? 8 : 4;
  const { count, at } = table(view, box, width);
  return Array.from({ length: count }, (_, i) =>
    width === 8 ? Number(view.getBigUint64(at + i * 8)) : view.getUint32(at + i * 4)
  );
}

function chunked(view, stsc, chunks, lengths) {
  const { count, at } = table(view, stsc, 12);
  const runs = Array.from({ length: count }, (_, i) => ({
    first: view.getUint32(at + i * 12),
    samples: view.getUint32(at + i * 12 + 4),
  }));
  const out = [];
  let sample = 0;
  for (let c = 0, run = 0; c < chunks.length && sample < lengths.length; c++) {
    while (run + 1 < runs.length && runs[run + 1].first <= c + 1) run++;
    let at = chunks[c];
    for (let s = 0; s < (runs[run]?.samples ?? 0) && sample < lengths.length; s++) {
      out.push({ at, size: lengths[sample] });
      at += lengths[sample++];
    }
  }
  return out;
}

function edit(view, box) {
  const { version } = full(view, box);
  const count = view.getUint32(box.body + 4);
  const width = version === 1 ? 20 : 12;
  for (let i = 0; i < count; i++) {
    const at = box.body + 8 + i * width;
    if (at + width > box.end) throw corrupt();
    const duration = version === 1 ? Number(view.getBigUint64(at)) : view.getUint32(at);
    const time = version === 1 ? Number(view.getBigInt64(at + 8)) : view.getInt32(at + 4);
    if (time >= 0) return { duration, time };
  }
  return null;
}

function smpb(bytes, view, udta) {
  const latin = (box) => String.fromCharCode(...bytes.subarray(box.body, box.end));
  for (const meta of boxes(bytes, view, udta.body, udta.end)) {
    if (meta.type !== 'meta') continue;
    for (const ilst of boxes(bytes, view, meta.body + 4, meta.end)) {
      if (ilst.type !== 'ilst') continue;
      for (const item of boxes(bytes, view, ilst.body, ilst.end)) {
        if (item.type !== '----') continue;
        const parts = Object.fromEntries(
          [...boxes(bytes, view, item.body, item.end)].map((b) => [b.type, b])
        );
        if (!parts.name || !parts.data || latin(parts.name).slice(4) !== 'iTunSMPB') continue;
        const fields = latin(parts.data).slice(8).trim().split(/\s+/);
        const [skip, , length] = fields.slice(1, 4).map((x) => Number.parseInt(x, 16));
        if (fields.length >= 4 && length > 0) return { time: skip, duration: length };
      }
    }
  }
  return null;
}

function timescale(view, box) {
  return view.getUint32(box.body + (full(view, box).version === 1 ? 20 : 12));
}

function track(bytes, view, trak) {
  const t = { boxes: {} };
  const walk = (from, to) => {
    for (const box of boxes(bytes, view, from, to)) {
      if (TRAVERSE.has(box.type)) walk(box.body, box.end);
      else t.boxes[box.type] ??= box;
    }
  };
  walk(trak.body, trak.end);
  return t.boxes;
}

function tfhd(view, box, defaults) {
  const { flags } = full(view, box);
  let at = box.body + 8;
  const header = { id: view.getUint32(box.body + 4), base: null, size: defaults };
  if (flags & 1) {
    header.base = Number(view.getBigUint64(at));
    at += 8;
  }
  at += (flags & 2 ? 4 : 0) + (flags & 8 ? 4 : 0);
  if (flags & 0x10) header.size = view.getUint32(at);
  return header;
}

function trun(view, box, run, out) {
  const { flags } = full(view, box);
  const count = view.getUint32(box.body + 4);
  let at = box.body + 8;
  if (flags & 1) {
    run.cursor = run.base + view.getInt32(at);
    at += 4;
  }
  at += flags & 4 ? 4 : 0;
  const fields = [0x100, 0x200, 0x400, 0x800].filter((f) => flags & f).length;
  const sized = flags & 0x200 ? (flags & 0x100 ? 4 : 0) : -1;
  if (at + count * fields * 4 > box.end) throw corrupt();
  for (let i = 0; i < count; i++, at += fields * 4) {
    const size = sized < 0 ? run.size : view.getUint32(at + sized);
    out.push({ at: run.cursor, size });
    run.cursor += size;
  }
}

function fragments(bytes, view, moof, id, defaults, out) {
  for (const traf of boxes(bytes, view, moof.body, moof.end)) {
    if (traf.type !== 'traf') continue;
    let run = null;
    for (const box of boxes(bytes, view, traf.body, traf.end)) {
      if (box.type === 'tfhd') {
        const header = tfhd(view, box, defaults);
        if (header.id !== id) break;
        const base = header.base ?? moof.at;
        run = { base, cursor: base, size: header.size };
      } else if (box.type === 'trun' && run) trun(view, box, run, out);
    }
  }
}

function movie(bytes, view, moov) {
  const found = { scale: 0, audio: null, gapless: null, trex: new Map() };
  for (const box of boxes(bytes, view, moov.body, moov.end)) {
    if (box.type === 'mvhd') found.scale = timescale(view, box);
    else if (box.type === 'udta') found.gapless = smpb(bytes, view, box);
    else if (box.type === 'mvex') {
      for (const t of boxes(bytes, view, box.body, box.end))
        if (t.type === 'trex')
          found.trex.set(view.getUint32(t.body + 4), view.getUint32(t.body + 16));
    } else if (box.type === 'trak' && !found.audio) {
      const t = track(bytes, view, box);
      if (t.hdlr && t.stsd && text(bytes, t.hdlr.body + 8) === 'soun') found.audio = t;
    }
  }
  return found;
}

function trim(view, audio, { scale, gapless }) {
  const media = audio.mdhd ? timescale(view, audio.mdhd) : 0;
  if (!media) return null;
  if (gapless) return { skip: gapless.time / media, keep: gapless.duration / media };
  const cut = audio.elst ? edit(view, audio.elst) : null;
  if (!cut || !scale) return null;
  return { skip: cut.time / media, keep: cut.duration ? cut.duration / scale : Infinity };
}

export function mp4Audio(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const top = [...boxes(bytes, view, 0, bytes.length)];
  const moov = top.find((b) => b.type === 'moov');
  if (!moov) throw corrupt();
  const found = movie(bytes, view, moov);
  const { audio } = found;
  if (!audio) throw new AudioFormatError('the MP4 file has no audio track');
  const entry = sampleEntry(bytes, view, audio.stsd);
  const chunks = audio.stco ?? audio.co64;
  const samples =
    audio.stsz && audio.stsc && chunks
      ? chunked(view, audio.stsc, offsets(view, chunks), sizes(view, audio.stsz))
      : [];
  const id = audio.tkhd
    ? view.getUint32(audio.tkhd.body + (full(view, audio.tkhd).version ? 20 : 12))
    : 0;
  for (const box of top)
    if (box.type === 'moof') fragments(bytes, view, box, id, found.trex.get(id) ?? 0, samples);
  const frames = samples.map(({ at, size }) => {
    if (at + size > bytes.length) throw corrupt();
    return bytes.subarray(at, at + size);
  });
  return { ...entry, frames, edit: trim(view, audio, found) };
}

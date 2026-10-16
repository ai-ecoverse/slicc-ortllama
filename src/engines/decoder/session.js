import { posix } from 'node:path';
import { sliceModel } from './slice.js';

let halfTable = null;

function halves() {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) halfTable[h] = sign * frac * 2 ** -24;
    else if (exp === 31) halfTable[h] = frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    else halfTable[h] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return halfTable;
}

export function toFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) {
    const table = halves();
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) out[i] = table[data[i]];
    return out;
  }
  return Float32Array.from(data);
}

const TYPED = { float16: Uint16Array, float32: Float32Array };

function emptyOf(type, length) {
  return new TYPED[type](length);
}

function kvDims(meta, fallback) {
  const shape = meta?.shape ?? [];
  const heads = Number.isInteger(shape[1]) ? shape[1] : fallback.heads;
  const dim = Number.isInteger(shape[3]) ? shape[3] : fallback.dim;
  return { heads, dim };
}

function metaOf(session, name) {
  return (session.inputMetadata ?? []).find((m) => m.name === name);
}

async function sessionFor({ ort, files, part, options, node }) {
  if (node) return ort.InferenceSession.create(files.path(part.model), options);
  const externalData = [];
  for (const rel of part.data) {
    externalData.push({ path: posix.basename(rel), data: await files.read(rel) });
  }
  const model = await files.read(part.model);
  return ort.InferenceSession.create(model, { ...options, externalData });
}

function outputLocations(gpu, presents) {
  if (!gpu) return {};
  return {
    preferredOutputLocation: Object.fromEntries([
      ['logits', 'cpu'],
      ...presents.map((p) => [p, 'gpu-buffer']),
    ]),
  };
}

export async function openModel({ backend, files, part, shape }) {
  const { ort } = backend;
  const node = backend.track === 'node';
  const gpu = backend.executionProviders[0] === 'webgpu';
  for (const rel of [part.decoder.model, ...part.decoder.data]) {
    if (!(await files.exists(rel))) throw new Error(`${rel} is not in the model directory`);
  }
  const base = { ...backend.sessionOptions, executionProviders: backend.executionProviders };
  const embed = part.embed
    ? await sessionFor({ ort, files, part: part.embed, options: base, node })
    : null;
  const presentNames = [];
  for (let l = 0; l < shape.layers; l++)
    presentNames.push(`present.${l}.key`, `present.${l}.value`);
  let decoder;
  try {
    decoder = await sessionFor({
      ort,
      files,
      part: part.decoder,
      options: { ...base, ...outputLocations(gpu, presentNames) },
      node,
    });
  } catch (error) {
    await embed?.release();
    throw error;
  }
  const inputs = new Set(decoder.inputNames);
  const pasts = decoder.inputNames.filter((n) => n.startsWith('past_key_values.'));
  const presentOf = (name) => name.replace('past_key_values.', 'present.');
  const fetches = ['logits', ...pasts.map(presentOf)];
  const kvMeta = metaOf(decoder, pasts[0]);
  const kvType = kvMeta?.type ?? 'float32';
  const { heads, dim } = kvDims(kvMeta, shape);
  const ids64 = (ids) => new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);

  function emptyPast() {
    const past = {};
    for (const name of pasts)
      past[name] = new ort.Tensor(kvType, emptyOf(kvType, 0), [1, heads, 0, dim]);
    return past;
  }

  async function embeddings(ids) {
    const out = await embed.run({ input_ids: ids64(ids) });
    return out[embed.outputNames[0]];
  }

  function feedsFor(ids, pastLen) {
    const n = ids.length;
    const feeds = {};
    if (inputs.has('input_ids')) feeds.input_ids = ids64(ids);
    if (inputs.has('attention_mask'))
      feeds.attention_mask = new ort.Tensor('int64', new BigInt64Array(pastLen + n).fill(1n), [
        1,
        pastLen + n,
      ]);
    if (inputs.has('position_ids'))
      feeds.position_ids = new ort.Tensor(
        'int64',
        BigInt64Array.from({ length: n }, (_, i) => BigInt(pastLen + i)),
        [1, n]
      );
    if (inputs.has('num_logits_to_keep'))
      feeds.num_logits_to_keep = new ort.Tensor('int64', BigInt64Array.from([1n]), []);
    return feeds;
  }

  async function lastRow(tensor) {
    const data = await tensor.getData();
    const vocab = tensor.dims.at(-1);
    const rows = data.length / vocab;
    return toFloat32(data.subarray((rows - 1) * vocab, rows * vocab));
  }

  async function forward(ids, past, pastLen, wantLogits, splice = null) {
    const feeds = feedsFor(ids, pastLen);
    let out = null;
    try {
      if (inputs.has('inputs_embeds')) feeds.inputs_embeds = await embeddings(ids);
      if (splice) splice(await feeds.inputs_embeds.getData(), feeds.inputs_embeds.dims.at(-1));
      out = await decoder.run({ ...feeds, ...past }, fetches);
      const logits = wantLogits ? await lastRow(out.logits) : null;
      const next = {};
      for (const name of pasts) next[name] = out[presentOf(name)];
      out.logits.dispose?.();
      dispose(past);
      return { past: next, logits };
    } catch (error) {
      dispose(past);
      if (out) dispose(out);
      throw error;
    } finally {
      feeds.inputs_embeds?.dispose?.();
    }
  }

  let slicer = null;

  async function slice(past, to) {
    slicer ??= await ort.InferenceSession.create(sliceModel(pasts, kvType), {
      ...base,
      ...(gpu ? { preferredOutputLocation: 'gpu-buffer' } : {}),
    });
    const out = await slicer.run({
      ...past,
      ends: new ort.Tensor('int64', BigInt64Array.from([BigInt(to)]), [1]),
    });
    return Object.fromEntries(pasts.map((name, i) => [name, out[`out.${i}`]]));
  }

  function dispose(past) {
    for (const t of Object.values(past)) t?.dispose?.();
  }

  let visionSession = null;
  let visionReady = null;

  async function encodeImage(pixels) {
    visionSession ??= sessionFor({ ort, files, part: part.vision, options: base, node }).catch(
      (error) => {
        visionSession = null;
        throw error;
      }
    );
    const session = await visionSession;
    visionReady = session;
    const input = new ort.Tensor('float32', pixels.data, [1, 3, pixels.height, pixels.width]);
    const out = await session.run({ [session.inputNames[0]]: input });
    const features = out[session.outputNames[0]];
    const data = toFloat32(await features.getData());
    features.dispose?.();
    return data;
  }

  return {
    emptyPast,
    forward,
    encodeImage: part.vision ? encodeImage : null,
    slice,
    dispose,
    bytesPerToken: pasts.length * heads * dim * (kvType === 'float16' ? 2 : 4),
    async release() {
      await decoder.release();
      await embed?.release();
      await slicer?.release();
      await visionReady?.release();
    },
  };
}

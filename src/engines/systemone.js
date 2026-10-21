import { posix } from 'node:path';
import { badRequest, HttpError } from '../http/errors.js';
import { imageFromUrl } from '../images.js';
import { optional } from './optional.js';

export const KEV = '@ai-ecoverse/kev.js';
export const MINISTRAL_KEV = '1.2.0';

const HELD_BYTES = 2 ** 30;

export function heldReads(files, budget = HELD_BYTES) {
  const held = new Map();
  let bytes = 0;
  const read = (rel, offset, length) => {
    let data = held.get(rel);
    if (data) held.delete(rel);
    else {
      data = files.readSync(rel);
      if (data.length > budget) return data.subarray(offset, offset + length);
      bytes += data.length;
      for (const [name, old] of held) {
        if (bytes <= budget) break;
        held.delete(name);
        bytes -= old.length;
      }
    }
    held.set(rel, data);
    return data.subarray(offset, offset + length);
  };
  const clear = () => {
    held.clear();
    bytes = 0;
  };
  return { read, clear };
}

export function localWeights(backend, files) {
  if (backend.track === 'node')
    return { options: { filePath: (rel) => files.path(rel) }, clear() {} };
  const reads = heldReads(files);
  return { options: { readSync: reads.read }, clear: reads.clear };
}

async function manifestOf(files, accept) {
  if (!(await files.exists('manifest.json'))) return null;
  try {
    const manifest = await files.json('manifest.json');
    return manifest?.variants && manifest.files && accept(manifest) ? manifest : null;
  } catch {
    return null;
  }
}

export async function detectBundle(files, accept) {
  const manifest = await manifestOf(files, accept);
  if (!manifest) return null;
  const variants = [];
  for (const [variant, part] of Object.entries(manifest.variants))
    if (await files.exists(part.model)) variants.push(variant);
  if (variants.length === 0) return null;
  const defaultVariant = variants.includes('q8f32') ? 'q8f32' : variants[0];
  return {
    variants,
    defaultVariant,
    inputModalities: manifest.vision ? ['text', 'image'] : ['text'],
    outputModalities: ['decisions'],
    contextLength: manifest.variants[defaultVariant].max_positions ?? null,
    manifest,
  };
}

const MINISTRAL_VARIANT = 'q8f32';
const MINISTRAL_CONTEXT = 73728;
const MINISTRAL_FILES = ['tokenizer.json', 'tokenizer_config.json', 'head.json', 'head.bin'];
const MINISTRAL_GRAPHS = ['embed_tokens', 'decoder_model_merged', 'vision_encoder'];

function isMinistral(manifest) {
  const files = manifest?.files;
  return (
    Array.isArray(files?.embed_tokens) &&
    Array.isArray(files?.decoder_model_merged) &&
    Number.isInteger(manifest.layers)
  );
}

function ministralNeeded(manifest) {
  const graphs = MINISTRAL_GRAPHS.filter((name) => Array.isArray(manifest.files[name]));
  return [
    'manifest.json',
    ...MINISTRAL_FILES,
    ...graphs.flatMap((name) => [
      `onnx/${name}.onnx`,
      ...manifest.files[name].map((f) => `onnx/${f}`),
    ]),
  ];
}

async function detectMinistral(files) {
  if (!(await files.exists('manifest.json'))) return null;
  let manifest;
  try {
    manifest = await files.json('manifest.json');
  } catch {
    return null;
  }
  if (!isMinistral(manifest)) return null;
  if (!(await files.exists('onnx/decoder_model_merged.onnx'))) return null;
  return {
    ministral: true,
    variants: [MINISTRAL_VARIANT],
    defaultVariant: MINISTRAL_VARIANT,
    inputModalities: manifest.files.vision_encoder ? ['text', 'image'] : ['text'],
    outputModalities: ['decisions'],
    contextLength: MINISTRAL_CONTEXT,
    manifest,
  };
}

function weights(part) {
  return part ? [part.model, ...(part.data ?? [])] : [];
}

function needed(info, variant) {
  const { manifest } = info;
  if (info.ministral) return ministralNeeded(manifest);
  return [
    'manifest.json',
    ...Object.values(manifest.files),
    ...weights(manifest.variants[variant]),
    ...weights(manifest.vision),
  ];
}

function folder(part) {
  const dir = posix.dirname(part.model);
  return dir === '.' ? weights(part) : [`${dir}/*`];
}

function serving(kev, { ValidationError, ContextOverflow }) {
  return {
    probs: (record, image) => kev.probs(record, image),
    async systemOne(request) {
      if (request.image !== undefined && !kev.acceptsImages)
        throw badRequest('this System One model takes no images', 'image');
      try {
        return await kev.systemOne(request);
      } catch (error) {
        if (error instanceof ValidationError) throw badRequest(error.message);
        if (request.image !== undefined && error instanceof RangeError)
          throw badRequest(
            `the image does not fit this model's vision encoder: ${error.message}`,
            'image'
          );
        if (error instanceof ContextOverflow)
          throw new HttpError(422, error.message, { code: 'context_length_exceeded' });
        throw error;
      }
    },
    release: () => kev.release(),
  };
}

export const systemone = {
  name: 'systemone',
  capability: 'systemone',
  async detect(files) {
    return (await detectBundle(files, (manifest) => manifest.special)) ?? detectMinistral(files);
  },
  async missing(files, info, variant) {
    const out = [];
    for (const rel of needed(info, variant)) if (!(await files.exists(rel))) out.push(rel);
    return out;
  },
  include(info, variant) {
    const { manifest } = info;
    if (info.ministral) return ['manifest.json', ...MINISTRAL_FILES, 'onnx/*'];
    return [
      'manifest.json',
      ...Object.values(manifest.files),
      ...folder(manifest.variants[variant]),
      ...(manifest.vision ? folder(manifest.vision) : []),
    ];
  },
  async load({ backend, files, variant, info, overrides = {}, library = () => optional(KEV) }) {
    const kevjs = await library();
    const ministral = info?.ministral ?? false;
    const loader = ministral ? kevjs.loadKevMinistral : kevjs.loadKev;
    if (typeof loader !== 'function')
      throw new HttpError(501, `this model needs ${KEV} ${MINISTRAL_KEV} or later`, {
        code: 'engine_unavailable',
      });
    const local = localWeights(backend, files);
    const kev = await loader((rel) => files.read(rel), {
      ...local.options,
      ort: backend.ort,
      ...(ministral ? {} : { variant }),
      executionProviders: backend.executionProviders,
      sessionOptions: backend.sessionOptions,
      cacheName: null,
      ...overrides,
    }).finally(local.clear);
    return serving(kev, kevjs);
  },
};

const SMOKE = {
  state: 'Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.',
  questions: {
    department: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: {
        returns: 'Exchanges and refunds',
        shipping: 'Delivery and delays',
        billing: 'Charges and payments',
      },
    },
    escalate: { type: 'noul', instructions: 'Does this need urgent human attention?' },
    frustration: {
      type: 'score',
      instructions: 'How frustrated is the customer?',
      criteria: ['Calm', 'Frustrated', 'Very angry'],
    },
  },
};

const NEAR_TIE = 0.05;

export function compareFixtures(results, fixtures) {
  let worst = 0;
  let worstAt = '';
  let sum = 0;
  let questions = 0;
  const flips = [];
  results.forEach((probs, i) => {
    probs.forEach((got, q) => {
      const ref = fixtures[i].probs[q];
      const d = Math.max(...got.map((x, j) => Math.abs(x - ref[j])));
      questions++;
      sum += d;
      if (d > worst) {
        worst = d;
        worstAt = `${fixtures[i].name}#${q}`;
      }
      const a = ref.indexOf(Math.max(...ref));
      const b = got.indexOf(Math.max(...got));
      if (a !== b) flips.push({ at: `${fixtures[i].name}#${q}`, margin: ref[a] - ref[b] });
    });
  });
  const mean = questions ? sum / questions : 0;
  const clearFlips = flips.filter((f) => f.margin > NEAR_TIE);
  return {
    questions,
    mean,
    max: worst,
    worstAt,
    flips,
    ok: mean <= 0.02 && worst <= 0.25 && clearFlips.length === 0,
  };
}

systemone.checkOverrides = (fixtures) => (fixtures ? { temperature: 1 } : {});

systemone.check = async (instance, { fixtures, now = () => performance.now() } = {}) => {
  const started = now();
  const answer = await instance.systemOne(SMOKE);
  const requestMs = now() - started;
  if (!fixtures) return { ok: true, requestMs, answer };
  const results = [];
  for (const f of fixtures) {
    const image = f.image ? await imageFromUrl(f.image, { param: 'image' }) : undefined;
    results.push(await instance.probs(f.record, image));
  }
  const compared = compareFixtures(results, fixtures);
  return {
    requestMs,
    answer,
    ...compared,
    summary: `fixtures: ${compared.questions} questions, mean |dp| ${compared.mean.toFixed(4)}, max ${compared.max.toFixed(4)} (${compared.worstAt}), ${compared.flips.length} changed`,
  };
};

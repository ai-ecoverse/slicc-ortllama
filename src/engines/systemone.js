import { posix } from 'node:path';
import { badRequest, HttpError } from '../http/errors.js';
import { imageFromUrl } from '../images.js';
import { optional } from './optional.js';

export const KEV = '@ai-ecoverse/kev.js';

const HELD_BYTES = 2 ** 30;

export function heldReads(files, budget = HELD_BYTES) {
  const held = new Map();
  let bytes = 0;
  const read = (rel, offset, length) => {
    let data = held.get(rel);
    if (data) held.delete(rel);
    else {
      data = files.readSync(rel);
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

async function manifestOf(files) {
  if (!(await files.exists('manifest.json'))) return null;
  try {
    const manifest = await files.json('manifest.json');
    return manifest?.variants && manifest.special && manifest.files ? manifest : null;
  } catch {
    return null;
  }
}

function weights(part) {
  return part ? [part.model, ...(part.data ?? [])] : [];
}

function needed(manifest, variant) {
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

export const systemone = {
  name: 'systemone',
  capability: 'systemone',
  async detect(files) {
    const manifest = await manifestOf(files);
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
  },
  async missing(files, info, variant) {
    const out = [];
    for (const rel of needed(info.manifest, variant)) if (!(await files.exists(rel))) out.push(rel);
    return out;
  },
  include(info, variant) {
    const { manifest } = info;
    return [
      'manifest.json',
      ...Object.values(manifest.files),
      ...folder(manifest.variants[variant]),
      ...(manifest.vision ? folder(manifest.vision) : []),
    ];
  },
  async load({ backend, files, variant, overrides = {}, library = () => optional(KEV) }) {
    const { loadKev, ValidationError, ContextOverflow } = await library();
    const node = backend.track === 'node';
    const reads = node ? null : heldReads(files);
    const kev = await loadKev((rel) => files.read(rel), {
      ...(node ? { filePath: (rel) => files.path(rel) } : { readSync: reads.read }),
      ort: backend.ort,
      variant,
      executionProviders: backend.executionProviders,
      sessionOptions: backend.sessionOptions,
      cacheName: null,
      ...overrides,
    }).finally(() => reads?.clear());
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

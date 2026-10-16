import { posix } from 'node:path';
import { badRequest, HttpError } from '../http/errors.js';
import { optional } from './optional.js';

export const KEV = '@ai-ecoverse/kev.js';

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
    const variants = Object.keys(manifest.variants);
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
    const kev = await loadKev((rel) => files.read(rel), {
      ort: backend.ort,
      variant,
      executionProviders: backend.executionProviders,
      sessionOptions: backend.sessionOptions,
      cacheName: null,
      ...overrides,
    });
    return {
      probs: (record) => kev.probs(record),
      async systemOne(request) {
        if (request.image !== undefined && !kev.acceptsImages)
          throw badRequest('this System One model takes no images', 'image');
        try {
          return await kev.systemOne(request);
        } catch (error) {
          if (error instanceof ValidationError) throw badRequest(error.message);
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
  for (const f of fixtures) results.push(await instance.probs(f.record));
  return { requestMs, answer, ...compareFixtures(results, fixtures) };
};

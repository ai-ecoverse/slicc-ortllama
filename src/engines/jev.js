import { badRequest, HttpError } from '../http/errors.js';
import { toAnswers, toRecord, validate } from './decisions.js';
import { optional } from './optional.js';
import { detectBundle, localWeights, systemone } from './systemone.js';

export const JEV = '@ai-ecoverse/jev-omni.js';

const MIN_OPTIONS = 2;
const MAX_OPTIONS = 256;
const TOO_LONG = /^The prompt is \d+ tokens/;

const round1 = (x) => Math.round(x * 10) / 10;

export const jev = {
  name: 'jev',
  capability: 'systemone',
  detect: (files) =>
    detectBundle(
      files,
      (manifest) => manifest.head?.classes && manifest.sha256 && !manifest.special
    ),
  missing: systemone.missing,
  include: systemone.include,
  async load({ backend, files, variant, overrides = {}, library = () => optional(JEV) }) {
    const { loadJevOmni } = await library();
    const weights = localWeights(backend, files);
    const model = await loadJevOmni((rel) => files.read(rel), {
      ...weights.options,
      ort: backend.ort,
      variant,
      executionProviders: backend.executionProviders,
      sessionOptions: backend.sessionOptions,
      cacheName: null,
      ...overrides,
    }).finally(weights.clear);
    const run = async (record, image) => {
      const features = image ? await model.imageFeatures(image) : undefined;
      const probs = [];
      let tokens = 0;
      for (const q of record.questions) {
        const question = { state: record.state, question: q.instr, options: q.options };
        const ids = model.encode(question, features?.numSoftTokens ?? 0);
        tokens += ids.length;
        probs.push(await model.probsForIds(ids, q.options.length, features));
      }
      return { probs, tokens };
    };
    return {
      probs: async (record, image) => (await run(record, image)).probs,
      async systemOne(request) {
        validate(request, { min: MIN_OPTIONS, max: MAX_OPTIONS });
        if (request.image !== undefined && !model.supportsImages)
          throw badRequest('this System One model takes no images', 'image');
        const started = performance.now();
        const { record, meta } = toRecord(request);
        let result;
        try {
          result = await run(record, request.image);
        } catch (error) {
          if (TOO_LONG.test(error?.message))
            throw new HttpError(422, error.message, { code: 'context_length_exceeded' });
          if (request.image !== undefined && error instanceof RangeError)
            throw badRequest(
              `the image does not fit this model's vision embedder: ${error.message}`,
              'image'
            );
          throw error;
        }
        const answers = toAnswers(result.probs, meta);
        const output = model.tokenizer.encode(JSON.stringify(answers), {
          add_special_tokens: false,
        }).ids.length;
        return {
          answers,
          usage: { input_tokens: result.tokens, output_tokens: output },
          latency_ms: round1(performance.now() - started),
        };
      },
      release: () => model.release(),
    };
  },
  checkOverrides: () => ({}),
  check: systemone.check,
};

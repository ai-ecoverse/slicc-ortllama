import { badRequest } from './http/errors.js';
import { PRIORITY } from './runtime.js';
import { transcriptionRoutes } from './transcription.js';

function modelEntry(m, aliasOf = null) {
  return {
    id: aliasOf ? aliasOf.alias : m.id,
    object: 'model',
    created: 0,
    owned_by: 'ortllama',
    architecture: { input_modalities: m.inputModalities, output_modalities: m.outputModalities },
    context_length: m.contextLength,
    ortllama: {
      engine: m.engine,
      variants: m.variants,
      default_variant: m.defaultVariant,
      revision: m.revision,
      ...(aliasOf ? { alias_of: aliasOf.target } : {}),
    },
  };
}

export function createRoutes({ catalog, runtime, version }) {
  return {
    'GET /': { public: true, handle: () => ({ body: `ortllama ${version} is running` }) },
    'GET /v1/models': {
      async handle() {
        const models = await catalog.list();
        const data = models.map((m) => modelEntry(m));
        for (const [alias, target] of Object.entries(catalog.aliases)) {
          const base = target.split(':')[0].split('@')[0];
          const m = models.find((x) => x.id === base);
          if (m) data.push(modelEntry(m, { alias, target }));
        }
        return { object: 'list', data };
      },
    },
    'GET /ortllama/ps': { handle: () => ({ models: runtime.ps() }) },
    'POST /v1/systemone': {
      async handle({ body, signal }) {
        if (typeof body !== 'object' || body === null || Array.isArray(body))
          throw badRequest('request body must be a JSON object');
        const resolved = await catalog.resolve(body.model ?? 'decisions');
        if (resolved.engine.capability !== 'systemone')
          throw badRequest(`model ${resolved.id} does not answer System One requests`, 'model');
        const { model, ...request } = body;
        const result = await runtime.use(
          resolved,
          PRIORITY.decisions,
          (kev) => kev.systemOne(request),
          signal
        );
        return { ...result, model: model ?? 'decisions' };
      },
    },
    ...transcriptionRoutes({ catalog, runtime }),
  };
}

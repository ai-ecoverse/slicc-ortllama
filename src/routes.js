import { chatRoute, refuse } from './chat.js';
import { badRequest } from './http/errors.js';
import { requestImage } from './images.js';
import { PRIORITY } from './runtime.js';
import { speechRoute } from './speech.js';
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
      ...m.details,
      ...(aliasOf ? { alias_of: aliasOf.target } : {}),
    },
  };
}

export function createRoutes({ catalog, runtime, version, maxTokens, aac }) {
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
    'GET /ortllama/ps': { handle: () => ({ models: runtime.ps(), memory: runtime.memory() }) },
    'POST /v1/chat/completions': chatRoute({ catalog, runtime, maxTokens }),
    'POST /v1/systemone': {
      async handle({ body, signal }) {
        if (typeof body !== 'object' || body === null || Array.isArray(body))
          throw badRequest('request body must be a JSON object');
        const resolved = await catalog.resolve(body.model ?? 'decisions');
        refuse(resolved, 'systemone', 'System One requests');
        const { model, image, ...text } = body;
        const request =
          image === undefined ? text : { ...text, image: await requestImage(image, resolved) };
        const result = await runtime.use(
          resolved,
          PRIORITY.decisions,
          (instance) => instance.systemOne(request),
          signal
        );
        return { ...result, model: model ?? 'decisions' };
      },
    },
    ...transcriptionRoutes({ catalog, runtime, aac }),
    'POST /v1/audio/speech': speechRoute({ catalog, runtime, aac }),
  };
}

import { optional } from '../../engines/optional.js';

const LANGUAGES = ['en-US', 'en-GB'];
const EVENTS = ['uncaughtException', 'unhandledRejection'];

export const xenova = {
  name: 'phonemizer',
  install: 'npm install phonemizer',
  languages: LANGUAGES,
  async load({ load = optional, env, events = process } = {}) {
    const before = Object.fromEntries(EVENTS.map((e) => [e, events.listeners(e)]));
    const lib = await load('phonemizer', { env });
    const added = EVENTS.flatMap((e) =>
      events
        .listeners(e)
        .filter((l) => !before[e].includes(l))
        .map((listener) => [e, listener])
    );
    const strip = () => {
      for (const [e, listener] of added) events.removeListener(e, listener);
    };
    strip();
    return {
      name: 'phonemizer',
      languages: LANGUAGES,
      async phonemize(parts, language) {
        const voice = language === 'en-GB' ? 'en' : 'en-us';
        const out = [];
        try {
          for (const part of parts) out.push((await lib.phonemize(part, voice)).join(' '));
        } finally {
          strip();
        }
        return out;
      },
    };
  },
};

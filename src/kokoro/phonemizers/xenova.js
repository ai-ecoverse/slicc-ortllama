import { optional } from '../../engines/optional.js';

const LANGUAGES = ['en-US', 'en-GB'];
const EVENTS = ['uncaughtException', 'unhandledRejection'];

export const xenova = {
  name: 'phonemizer',
  install: 'npm install phonemizer',
  languages: LANGUAGES,
  async load({ load = optional, env, events = process } = {}) {
    const kept = Object.fromEntries(EVENTS.map((e) => [e, events.listeners(e)]));
    const strip = () => {
      for (const e of EVENTS)
        for (const listener of events.listeners(e))
          if (!kept[e].includes(listener)) events.removeListener(e, listener);
    };
    const lib = await load('phonemizer', { env });
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

export const LANGUAGES = {
  a: { espeak: 'en-us', language: 'en-US' },
  b: { espeak: 'en-gb', language: 'en-GB' },
  e: { espeak: 'es', language: 'es-ES' },
  f: { espeak: 'fr-fr', language: 'fr-FR' },
  h: { espeak: 'hi', language: 'hi-IN' },
  i: { espeak: 'it', language: 'it-IT' },
  p: { espeak: 'pt-br', language: 'pt-BR' },
};

export const VOICES = [
  'af_heart',
  'af_alloy',
  'af_aoede',
  'af_bella',
  'af_jessica',
  'af_kore',
  'af_nicole',
  'af_nova',
  'af_river',
  'af_sarah',
  'af_sky',
  'am_adam',
  'am_echo',
  'am_eric',
  'am_fenrir',
  'am_liam',
  'am_michael',
  'am_onyx',
  'am_puck',
  'am_santa',
  'bf_alice',
  'bf_emma',
  'bf_isabella',
  'bf_lily',
  'bm_daniel',
  'bm_fable',
  'bm_george',
  'bm_lewis',
  'ef_dora',
  'em_alex',
  'em_santa',
  'ff_siwis',
  'hf_alpha',
  'hf_beta',
  'hm_omega',
  'hm_psi',
  'if_sara',
  'im_nicola',
  'pf_dora',
  'pm_alex',
  'pm_santa',
];

export const DEFAULT_VOICE = VOICES[0];

export function voiceInfo(id) {
  const name = id.slice(3);
  return {
    id,
    name: name[0].toUpperCase() + name.slice(1),
    languages: [LANGUAGES[id[0]].language],
  };
}

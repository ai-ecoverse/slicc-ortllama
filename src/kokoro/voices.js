export const LANGUAGES = {
  a: { language: 'en-US' },
  b: { language: 'en-GB' },
  e: { language: 'es-ES' },
  f: { language: 'fr-FR' },
  h: { language: 'hi-IN' },
  i: { language: 'it-IT' },
  p: { language: 'pt-BR' },
};

export const GENDERS = { f: 'female', m: 'male' };

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

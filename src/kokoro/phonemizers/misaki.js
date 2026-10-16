import { join } from 'node:path';
import { hubCache, repoDir, resolveRevision } from '../../hf/cache.js';
import { modelFiles } from '../../hf/files.js';
import { HttpError } from '../../http/errors.js';

export const LEXICON = 'hexgrad/misaki';

const FILES = {
  'en-US': ['us_gold.json', 'us_silver.json'],
  'en-GB': ['gb_gold.json', 'gb_silver.json'],
};

const ONES =
  'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(
    ' '
  );
const TENS = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split(' ');
const SCALES = [
  [1e9, 'billion'],
  [1e6, 'million'],
  [1e3, 'thousand'],
];

export function numberWords(n) {
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? ` ${ONES[n % 10]}` : '');
  if (n < 1000)
    return `${ONES[Math.floor(n / 100)]} hundred${n % 100 ? ` ${numberWords(n % 100)}` : ''}`;
  for (const [size, name] of SCALES)
    if (n >= size)
      return `${numberWords(Math.floor(n / size))} ${name}${n % size ? ` ${numberWords(n % size)}` : ''}`;
}

function value(entry) {
  if (typeof entry === 'string') return entry;
  if (!entry || typeof entry !== 'object') return null;
  return entry.DEFAULT ?? Object.values(entry).find((v) => typeof v === 'string') ?? null;
}

function plural(stem, language) {
  if (/[szʃʒʧʤ]$/.test(stem)) return `${stem}${language === 'en-GB' ? 'ɪ' : 'ᵻ'}z`;
  return `${stem}${/[ptkfθ]$/.test(stem) ? 's' : 'z'}`;
}

function past(stem, language) {
  if (/[td]$/.test(stem)) return `${stem}${language === 'en-GB' ? 'ɪ' : 'ᵻ'}d`;
  return `${stem}${/[pkfθsʃʧ]$/.test(stem) ? 't' : 'd'}`;
}

const SUFFIXES = [
  { test: /'s$/, min: 3, stems: (w) => [w.slice(0, -2)], join: plural },
  {
    test: /s$/,
    min: 4,
    stems: (w) => [w.slice(0, -1), /es$/i.test(w) ? w.slice(0, -2) : null],
    join: plural,
  },
  {
    test: /ed$/,
    min: 5,
    stems: (w) => [w.slice(0, -2), w.slice(0, -1), w.slice(0, -3)],
    join: past,
  },
  {
    test: /ing$/,
    min: 6,
    stems: (w) => [w.slice(0, -3), `${w.slice(0, -3)}e`, w.slice(0, -4)],
    join: (stem) => `${stem}ɪŋ`,
  },
];

const capital = (word) => word[0].toUpperCase() + word.slice(1).toLowerCase();

export function createMisaki(lexicons, { fallback = () => null } = {}) {
  function lookup(word, language) {
    if (!word) return null;
    for (const dict of lexicons[language])
      for (const key of [word, word.toLowerCase(), capital(word)]) {
        const found = value(Object.hasOwn(dict, key) ? dict[key] : null);
        if (found) return found;
      }
    return null;
  }

  function stemmed(word, language) {
    for (const rule of SUFFIXES) {
      if (word.length < rule.min || !rule.test.test(word.toLowerCase())) continue;
      const stem = rule
        .stems(word)
        .map((w) => lookup(w, language))
        .find(Boolean);
      if (stem) return rule.join(stem, language);
    }
    return null;
  }

  function spelled(word, language) {
    const letters = [...word].map((c) => lookup(c.toUpperCase(), language));
    return letters.every(Boolean) ? letters.join('') : null;
  }

  function known(word, language) {
    if (/^\d+$/.test(word))
      return word.length <= 12
        ? say(numberWords(Number(word)), language)
        : say([...word].map((d) => ONES[d]).join(' '), language);
    if (word.includes('-'))
      return word
        .split('-')
        .filter(Boolean)
        .map((w) => known(w, language))
        .reduce((all, w) => (all === null || w === null ? null : `${all} ${w}`.trim()), '');
    const found = lookup(word, language) ?? stemmed(word, language);
    if (found) return found;
    return /^[A-Z]{2,5}$/.test(word) ? spelled(word, language) : null;
  }

  function say(text, language) {
    const words = text.split(' ').map((w) => lookup(w, language));
    return words.every(Boolean) ? words.join(' ') : null;
  }

  return {
    name: 'misaki',
    languages: Object.keys(lexicons),
    async phonemize(parts, language) {
      const words = parts.map((part) => part.split(/(\s+)/));
      const unknown = new Set();
      const said = new Map();
      for (const tokens of words)
        for (const token of tokens) {
          if (!/\S/.test(token) || said.has(token)) continue;
          const found = known(token, language);
          if (found === null) unknown.add(token);
          else said.set(token, found);
        }
      if (unknown.size > 0) {
        const backup = fallback(language);
        const list = [...unknown];
        const guesses = backup
          ? await backup.phonemize(list, language)
          : list.map((w) => spelled(w, language) ?? '');
        list.forEach((w, i) => {
          said.set(w, guesses[i]);
        });
      }
      return words.map((tokens) =>
        tokens
          .map((t) => (/\S/.test(t) ? said.get(t) : ' '))
          .join('')
          .replace(/\s+/g, ' ')
          .trim()
      );
    },
  };
}

function unavailable() {
  return new HttpError(
    501,
    `the misaki lexicon is not in the Hugging Face cache; run: hf download ${LEXICON} --repo-type dataset --include 'us_*.json' 'gb_*.json'`,
    { code: 'engine_unavailable' }
  );
}

export const misaki = {
  name: 'misaki',
  install: `hf download ${LEXICON} --repo-type dataset --include 'us_*.json' 'gb_*.json'`,
  languages: Object.keys(FILES),
  async load({ cache = hubCache(), fallback } = {}) {
    const rev = await resolveRevision(cache, LEXICON, 'main', 'dataset');
    if (!rev) throw unavailable();
    const blobs = join(repoDir(cache, LEXICON, 'dataset'), 'blobs');
    const files = modelFiles(rev.snapshot, { allow: [rev.snapshot, blobs] });
    const lexicons = {};
    for (const [language, names] of Object.entries(FILES)) {
      if (!(await files.exists(names[0]))) continue;
      lexicons[language] = await Promise.all(
        names.map(async (name) => ((await files.exists(name)) ? files.json(name) : {}))
      );
    }
    if (Object.keys(lexicons).length === 0) throw unavailable();
    return createMisaki(lexicons, { fallback });
  },
};

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

const VERBS = ['VERB', 'VBD', 'VBN', 'VBP'];
const AS_VERB = ['VERB', 'VBP', 'DEFAULT'];
const ORDINALS = {
  one: 'first',
  two: 'second',
  three: 'third',
  five: 'fifth',
  eight: 'eighth',
  nine: 'ninth',
  twelve: 'twelfth',
};
const SYMBOLS = {
  '%': 'percent',
  '&': 'and',
  '+': 'plus',
  '@': 'at',
  '#': 'number',
  '=': 'equals',
};

export function ordinalWords(n) {
  const words = numberWords(n).split(' ');
  const last = words.pop();
  const ordinal = ORDINALS[last] ?? (last.endsWith('y') ? `${last.slice(0, -1)}ieth` : `${last}th`);
  return [...words, ordinal].join(' ');
}

const digits = (text) => [...text].map((d) => ONES[d]).join(' ');

export function expand(word) {
  if (/^\d+$/.test(word)) return word.length <= 12 ? numberWords(Number(word)) : digits(word);
  const ordinal = /^(\d{1,12})(st|nd|rd|th)$/i.exec(word);
  if (ordinal) return ordinalWords(Number(ordinal[1]));
  const decimal = /^(\d{1,12})\.(\d+)$/.exec(word);
  if (decimal) return `${numberWords(Number(decimal[1]))} point ${digits(decimal[2])}`;
  if (!/[\d%&+@#=]/.test(word)) return null;
  return word
    .match(/\d+|[^\d%&+@#=]+|./g)
    .map((run) => SYMBOLS[run] ?? (/^\d+$/.test(run) ? expand(run) : run))
    .join(' ');
}

function pick(entry, prefer) {
  if (typeof entry === 'string') return entry;
  if (!entry || typeof entry !== 'object') return null;
  for (const pos of prefer) if (typeof entry[pos] === 'string') return entry[pos];
  return Object.values(entry).find((v) => typeof v === 'string') ?? null;
}

function verbal(entry) {
  if (!entry || typeof entry !== 'object') return false;
  const base = pick(entry, ['DEFAULT']);
  return VERBS.some((pos) => typeof entry[pos] === 'string' && entry[pos] !== base);
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
  function entry(word, language) {
    if (!word) return null;
    for (const dict of lexicons[language])
      for (const key of [word, word.toLowerCase(), capital(word)])
        if (Object.hasOwn(dict, key) && pick(dict[key], ['DEFAULT'])) return dict[key];
    return null;
  }

  const lookup = (word, language, prefer = ['DEFAULT']) => pick(entry(word, language), prefer);

  function stemmed(word, language) {
    for (const rule of SUFFIXES) {
      if (word.length < rule.min || !rule.test.test(word.toLowerCase())) continue;
      const stem = rule
        .stems(word)
        .map((w) => lookup(w, language, AS_VERB))
        .find(Boolean);
      if (stem) return rule.join(stem, language);
    }
    return null;
  }

  function spelled(word, language) {
    const letters = [...word].map((c) => lookup(c.toUpperCase(), language));
    return letters.every(Boolean) ? letters.join('') : null;
  }

  function loosely(word, language) {
    return [...word]
      .map(
        (c) =>
          lookup(c.toUpperCase(), language) ?? (/\d/.test(c) ? lookup(ONES[c], language) : null)
      )
      .filter(Boolean)
      .join('');
  }

  const joined = (parts) =>
    parts.reduce((all, w) => (all === null || w === null ? null : `${all} ${w}`.trim()), '');

  function known(token, language, deferring) {
    const word = token.replace(/^[^\p{L}\p{N}']+|[^\p{L}\p{N}'%]+$/gu, '');
    if (!word) return '';
    if (word.includes('-'))
      return joined(
        word
          .split('-')
          .filter(Boolean)
          .map((w) => known(w, language, deferring))
      );
    const direct = entry(word, language);
    if (direct !== null) return deferring && verbal(direct) ? null : pick(direct, ['DEFAULT']);
    const stem = stemmed(word, language);
    if (stem) return stem;
    const words = expand(word);
    if (words !== null) return joined(words.split(' ').map((w) => known(w, language, deferring)));
    return /^[A-Z]{2,5}$/.test(word) ? spelled(word, language) : null;
  }

  async function guessed(rows, language, backup) {
    const asked = rows.filter((row) => row.some((t) => t.ipa === null));
    if (asked.length === 0) return;
    if (!backup) {
      for (const t of asked.flat()) t.ipa ??= loosely(t.text, language);
      return;
    }
    const lines = await backup.phonemize(
      asked.map((row) => row.map((t) => t.text).join(' ')),
      language
    );
    asked.forEach((row, i) => {
      const words = lines[i].split(/\s+/).filter(Boolean);
      if (words.length !== row.length) row.splice(0, row.length, { text: '', ipa: lines[i] });
      else
        row.forEach((t, j) => {
          t.ipa ??= words[j];
        });
    });
  }

  return {
    name: 'misaki',
    languages: Object.keys(lexicons),
    async phonemize(parts, language) {
      const backup = fallback(language);
      const rows = parts.map((part) =>
        part
          .split(/\s+/)
          .filter(Boolean)
          .map((text) => ({ text, ipa: known(text, language, Boolean(backup)) }))
      );
      await guessed(rows, language, backup);
      return rows.map((row) =>
        row
          .map((t) => t.ipa)
          .join(' ')
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

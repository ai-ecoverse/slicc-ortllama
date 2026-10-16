const PUNCTUATION = ';:,.!?¡¿—…"«»“”(){}[]';
const PUNCTUATION_RUN = new RegExp(
  `(\\s*[${PUNCTUATION.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}]+\\s*)+`,
  'g'
);

function spellYear(match) {
  if (match.includes('.')) return match;
  if (match.includes(':')) {
    const [h, m] = match.split(':').map(Number);
    if (m === 0) return `${h} o'clock`;
    return m < 10 ? `${h} oh ${m}` : `${h} ${m}`;
  }
  const year = Number.parseInt(match.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) return match;
  const left = match.slice(0, 2);
  const right = Number.parseInt(match.slice(2, 4), 10);
  const plural = match.endsWith('s') ? 's' : '';
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) return `${left} hundred${plural}`;
    if (right < 10) return `${left} oh ${right}${plural}`;
  }
  return `${left} ${right}${plural}`;
}

function spellMoney(match) {
  const unit = match[0] === '$' ? 'dollar' : 'pound';
  const amount = match.slice(1);
  if (Number.isNaN(Number(amount))) return `${amount} ${unit}s`;
  if (!amount.includes('.')) return `${amount} ${unit}${amount === '1' ? '' : 's'}`;
  const [whole, fraction] = amount.split('.');
  const cents = Number.parseInt(fraction.padEnd(2, '0'), 10);
  const small =
    match[0] === '$' ? (cents === 1 ? 'cent' : 'cents') : cents === 1 ? 'penny' : 'pence';
  return `${whole} ${unit}${whole === '1' ? '' : 's'} and ${cents} ${small}`;
}

function spellDecimal(match) {
  const [whole, fraction] = match.split('.');
  return `${whole} point ${fraction.split('').join(' ')}`;
}

export function normalizeEnglish(text) {
  return text
    .replace(/[‘’]/g, "'")
    .replace(/«/g, '“')
    .replace(/»/g, '”')
    .replace(/[“”]/g, '"')
    .replace(/\(/g, '«')
    .replace(/\)/g, '»')
    .replace(/、/g, ', ')
    .replace(/。/g, '. ')
    .replace(/！/g, '! ')
    .replace(/，/g, ', ')
    .replace(/：/g, ': ')
    .replace(/；/g, '; ')
    .replace(/？/g, '? ')
    .replace(/[^\S \n]/g, ' ')
    .replace(/ {2,}/g, ' ')
    .replace(/(?<=\n) +(?=\n)/g, '')
    .replace(/\bD[Rr]\.(?= [A-Z])/g, 'Doctor')
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, 'Mister')
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, 'Miss')
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, 'Mrs')
    .replace(/\betc\.(?! [A-Z])/gi, 'etc')
    .replace(/\b(y)eah?\b/gi, "$1e'a")
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, spellYear)
    .replace(/(?<=\d),(?=\d)/g, '')
    .replace(
      /[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi,
      spellMoney
    )
    .replace(/\d*\.\d+/g, spellDecimal)
    .replace(/(?<=\d)-(?=\d)/g, ' to ')
    .replace(/(?<=\d)S/g, ' S')
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, 's')
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, '-'))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, '-')
    .trim();
}

export function splitPunctuation(text) {
  const parts = [];
  let cursor = 0;
  for (const match of text.matchAll(PUNCTUATION_RUN)) {
    if (cursor < match.index) parts.push({ punct: false, text: text.slice(cursor, match.index) });
    if (match[0].length > 0) parts.push({ punct: true, text: match[0] });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) parts.push({ punct: false, text: text.slice(cursor) });
  return parts;
}

export function fixPhonemes(phonemes, english, american) {
  let out = phonemes.replace(/kəkˈoːɹoʊ/g, 'kˈoʊkəɹoʊ').replace(/kəkˈɔːɹəʊ/g, 'kˈəʊkəɹəʊ');
  if (english)
    out = out.replace(/ʲ/g, 'j').replace(/r/g, 'ɹ').replace(/x/g, 'k').replace(/ɬ/g, 'l');
  out = out.replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, ' ').replace(/ z(?=[;:,.!?¡¿—…"«»“” ]|$)/g, 'z');
  if (american) out = out.replace(/(?<=nˈaɪn)ti(?!ː)/g, 'di');
  return out.trim();
}

const ENDS = '.!?…。？！';
const CLOSING = '"\')]}」』';
const ABBREVIATIONS = new Set(
  'mr mrs ms dr prof sr jr sgt col gen rep sen gov lt maj capt st mt etc co inc ltd dept vs p pg jan feb mar apr jun jul aug sep sept oct nov dec sun mon tu tue tues wed th thu thur thurs fri sat'.split(
    ' '
  )
);
const PAIRS = new Map([
  [')', '('],
  [']', '['],
  ['}', '{'],
  ['》', '《'],
  ['〉', '〈'],
  ['›', '‹'],
  ['»', '«'],
  ['」', '「'],
  ['』', '『'],
  ['〕', '〔'],
  ['】', '【'],
]);
const OPENERS = new Set(PAIRS.values());

const ends = (c, newline = true) => ENDS.includes(c) || (newline && c === '\n');

function wordAt(text, start) {
  let end = start;
  while (end < text.length && !/\s/.test(text[end])) end++;
  return text.substring(start, end);
}

function abbreviation(word) {
  return ABBREVIATIONS.has(
    word
      .replace(/['’]s$/i, '')
      .replace(/\.+$/, '')
      .toLowerCase()
  );
}

function track(c, stack, i, text) {
  if (c === '"' || c === "'") {
    if (c === "'" && i > 0 && /[A-Za-z]/.test(text[i - 1]) && /[A-Za-z]/.test(text[i + 1] ?? ''))
      return;
    if (stack.at(-1) === c) stack.pop();
    else stack.push(c);
    return;
  }
  if (OPENERS.has(c)) stack.push(c);
  else if (PAIRS.has(c) && stack.at(-1) === PAIRS.get(c)) stack.pop();
}

function boundary(text, i) {
  let end = i;
  while (end + 1 < text.length && ends(text[end + 1], false)) end++;
  while (end + 1 < text.length && CLOSING.includes(text[end + 1])) end++;
  let next = end + 1;
  while (next < text.length && /\s/.test(text[next])) next++;
  return { end, next };
}

function breaksAt(text, start, i) {
  const c = text[i];
  if (/(^|\n)\d+$/.test(text.slice(start, i))) return null;
  const { end, next } = boundary(text, i);
  if (i === next - 1 && c !== '\n') return null;
  if (next === text.length) return null;
  let from = i - 1;
  while (from >= 0 && /\S/.test(text[from])) from--;
  from = Math.max(start, from + 1);
  const word = wordAt(text, from);
  if (!word) return null;
  if ((/https?[,:]\/\//.test(word) || word.includes('@')) && !ends(word.at(-1)))
    return { skip: from + word.length };
  if (abbreviation(word)) return null;
  if (/^([A-Za-z]\.)+$/.test(word) && /[A-Z]/.test(text[next])) return null;
  if (c === '.' && /[a-z]/.test(text[next])) return null;
  const sentence = text.substring(start, end + 1).trim();
  if (sentence === '...' || sentence === '…') return null;
  return { end, sentence };
}

export function splitSentences(text) {
  const out = [];
  const stack = [];
  let start = 0;
  let i = 0;
  while (i < text.length) {
    track(text[i], stack, i, text);
    const hit = stack.length === 0 && ends(text[i]) ? breaksAt(text, start, i) : null;
    if (hit?.skip !== undefined) i = hit.skip;
    else if (hit) {
      if (hit.sentence) out.push(hit.sentence);
      i = start = hit.end + 1;
    } else i++;
  }
  const rest = text.substring(start).trim();
  if (rest) out.push(rest);
  return out;
}

const CLAUSE = /[,;:—]/g;

export function splitLong(text, max = 300) {
  const out = [];
  let rest = text.trim();
  while (rest.length > max) {
    const head = rest.slice(0, max + 1);
    const clause = Math.max(-1, ...[...head.matchAll(CLAUSE)].map((m) => m.index)) + 1;
    const space = head.lastIndexOf(' ');
    const cut = clause >= max / 3 ? clause : space > 0 ? space : clause > 0 ? clause : max;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) out.push(rest);
  return out;
}

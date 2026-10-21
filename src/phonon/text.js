import { isTld } from './tlds.js';

const WHITESPACE = /^\p{White_Space}$/u;
const ALPHANUMERIC = /^[\p{Alphabetic}\p{Nd}\p{Nl}\p{No}]$/u;
const ALPHABETIC = /^\p{Alphabetic}$/u;
const UPPERCASE = /^\p{Uppercase}$/u;
const LOWERCASE = /^\p{Lowercase}$/u;
const CONTROL = /^\p{Cc}$/u;
const ASCII_PUNCTUATION = /^[!-/:-@[-`{-~]$/;

const isWhitespace = (c) => c !== undefined && WHITESPACE.test(c);
const isAlphanumeric = (c) => c !== undefined && ALPHANUMERIC.test(c);
const isAlphabetic = (c) => c !== undefined && ALPHABETIC.test(c);
const isDigit = (c) => c !== undefined && c >= '0' && c <= '9';
const isAsciiPunctuation = (c) => c !== undefined && ASCII_PUNCTUATION.test(c);
const isAsciiUpper = (c) => c >= 'A' && c <= 'Z';

const DOUBLE_QUOTES = new Set('»«‹›“”„‟″‶⹂❝❞❠〝〞〟＂');
const SINGLE_QUOTES = new Set('‘’‚‛′‵❛❜❟＇ʼʻʹ´`');
const BULLETS = new Set('•‣◦·→←↑↓➡➜');
const SPACERS = new Set('–*—[]{}');
const CURRENCY = new Set('$€£¥¢');
const PUNCTUATION = '!?.:,;…';

const FOLDS =
  'ÌIÒOÝYìiòoýyÿyĀAāaĂAăaĄAąaĆCćcĈCĉcĊCċcČCčcĎDďdĒEēeĔEĕeĖEėeĘEęeĚEěeĜGĝgĞGğgĠGġgĢGģgĤHĥhĨIĩiĪIīiĬIĭiĮIįiİIĴJĵjĶKķkĹLĺlĻLļlĽLľlŃNńnŅNņnŇNňnŌOōoŎOŏoŐOőoŔRŕrŖRŗrŘRřrŚSśsŜSŝsŞSşsŠSšsŢTţtŤTťtŨUũuŪUūuŬUŭuŮUůuŰUűuŲUųuŴWŵwŶYŷyŸYŹZźzŻZżzŽZžzƠOơoƯUưuǍAǎaǏIǐiǑOǒoǓUǔuǕUǖuǗUǘuǙUǚuǛUǜuǞAǟaǠAǡaǦGǧgǨKǩkǪOǫoǬOǭoǰjǴGǵgǸNǹnǺAǻaȀAȁaȂAȃaȄEȅeȆEȇeȈIȉiȊIȋiȌOȍoȎOȏoȐRȑrȒRȓrȔUȕuȖUȗuȘSșsȚTțtȞHȟhȦAȧaȨEȩeȪOȫoȬOȭoȮOȯoȰOȱoȲYȳyḀAḁaḂBḃbḄBḅbḆBḇbḈCḉcḊDḋdḌDḍdḎDḏdḐDḑdḒDḓdḔEḕeḖEḗeḘEḙeḚEḛeḜEḝeḞFḟfḠGḡgḢHḣhḤHḥhḦHḧhḨHḩhḪHḫhḬIḭiḮIḯiḰKḱkḲKḳkḴKḵkḶLḷlḸLḹlḺLḻlḼLḽlḾMḿmṀMṁmṂMṃmṄNṅnṆNṇnṈNṉnṊNṋnṌOṍoṎOṏoṐOṑoṒOṓoṔPṕpṖPṗpṘRṙrṚRṛrṜRṝrṞRṟrṠSṡsṢSṣsṤSṥsṦSṧsṨSṩsṪTṫtṬTṭtṮTṯtṰTṱtṲUṳuṴUṵuṶUṷuṸUṹuṺUṻuṼVṽvṾVṿvẀWẁwẂWẃwẄWẅwẆWẇwẈWẉwẊXẋxẌXẍxẎYẏyẐZẑzẒZẓzẔZẕzẖhẗtẘwẙyẠAạaẢAảaẤAấaẦAầaẨAẩaẪAẫaẬAậaẮAắaẰAằaẲAẳaẴAẵaẶAặaẸEẹeẺEẻeẼEẽeẾEếeỀEềeỂEểeỄEễeỆEệeỈIỉiỊIịiỌOọoỎOỏoỐOốoỒOồoỔOổoỖOỗoỘOộoỚOớoỜOờoỞOởoỠOỡoỢOợoỤUụuỦUủuỨUứuỪUừuỬUửuỮUữuỰUựuỲYỳyỴYỵyỶYỷyỸYỹy';

const FOLD = new Map();
{
  const chars = [...FOLDS];
  for (let i = 0; i + 1 < chars.length; i += 2) FOLD.set(chars[i], chars[i + 1]);
}

const SUBSCRIPTS = new Map([
  ...[...'₀₁₂₃₄₅₆₇₈₉'].map((c, i) => [c, String(i)]),
  ...[...'₊+₋-ₐaₑeₒoₓxₕhₖkₗlₘmₙnₚpₛsₜtᵢiⱼj'].flatMap((c, i, all) =>
    i % 2 ? [] : [[c, all[i + 1]]]
  ),
]);

const SUPERSCRIPTS = new Map([
  ...[...'⁰¹²³⁴⁵⁶⁷⁸⁹'].map((c, i) => [c, String(i)]),
  ['⁻', '-'],
  ['⁺', '+'],
  ['ⁿ', 'n'],
]);

const GREEK = [
  'alpha',
  'beta',
  'gamma',
  'delta',
  'epsilon',
  'zeta',
  'eta',
  'theta',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'xi',
  'omicron',
  'pi',
  'rho',
  'sigma',
  'tau',
  'upsilon',
  'phi',
  'chi',
  'psi',
  'omega',
];

const SYMBOLS = new Map([
  ['×', 'times'],
  ['÷', 'divided by'],
  ['≤', 'less than or equal to'],
  ['≥', 'greater than or equal to'],
  ['≠', 'is not equal to'],
  ['≈', 'approximately'],
  ['√', 'square root of'],
  ['∞', 'infinity'],
  ['∑', 'sigma'],
  ['‰', 'per mille'],
]);

function greek(c) {
  const code = c.codePointAt(0);
  let i = -1;
  if (code >= 0x3b1 && code <= 0x3c1) i = code - 0x3b1;
  else if (code === 0x3c2) i = 17;
  else if (code >= 0x3c3 && code <= 0x3c9) i = code - 0x3b1 - 1;
  else if (code >= 0x391 && code <= 0x3a1) i = code - 0x391;
  else if (code >= 0x3a3 && code <= 0x3a9) i = code - 0x391 - 1;
  return i < 0 ? null : GREEK[i];
}

function isEmoji(c) {
  const code = c.codePointAt(0);
  return (
    (code >= 0x1f000 && code <= 0x1faff) ||
    (code >= 0x2600 && code <= 0x27bf) ||
    (code >= 0x2b00 && code <= 0x2bff) ||
    code === 0xfe0f ||
    code === 0x200d
  );
}

const power = (n) => (n === '2' ? 'squared' : n === '3' ? 'cubed' : `to the power of ${n}`);

class Appender {
  constructor() {
    this.buffer = [];
  }

  last() {
    return this.buffer[this.buffer.length - 1];
  }

  keeps(last) {
    if ('"\'@+%#&'.includes(last)) return true;
    return last === '$' && this.buffer.length >= 2 && isDigit(this.buffer[this.buffer.length - 2]);
  }

  push(c) {
    if (c === '.' || c === ',')
      while (
        this.buffer.length &&
        (isWhitespace(this.last()) || (isAsciiPunctuation(this.last()) && !this.keeps(this.last())))
      )
        this.buffer.pop();
    this.buffer.push(c);
  }

  pushStr(s) {
    for (const c of s) this.push(c);
  }

  endsWithOne() {
    let n = this.buffer.length;
    while (n > 0 && isWhitespace(this.buffer[n - 1])) n--;
    const b = this.buffer.slice(0, n);
    const before = b[b.length - 2];
    return b[b.length - 1] === '1' && !(isDigit(before) || before === '.' || before === ',');
  }

  endsWithN() {
    const b = this.last() === '.' ? this.buffer.slice(0, -1) : this.buffer;
    const last = b[b.length - 1];
    return (last === 'n' || last === 'N') && !isAlphanumeric(b[b.length - 2]);
  }

  endsWithSpacedDigit() {
    const i = this.buffer.findLastIndex((c) => !isWhitespace(c));
    return isWhitespace(this.last()) && isDigit(this.buffer[i]);
  }

  pushSpoken(word) {
    this.pushWhitespace();
    this.pushStr(word);
    this.pushWhitespace();
  }

  pushWhitespace() {
    if (!isWhitespace(this.last()) && this.buffer.length) this.push(' ');
  }

  popWhitespace() {
    while (isWhitespace(this.last())) this.buffer.pop();
  }

  toString() {
    this.popWhitespace();
    return this.buffer.join('');
  }
}

function trimEndWith(s, test) {
  const chars = [...s];
  while (chars.length && test(chars[chars.length - 1])) chars.pop();
  return chars.join('');
}

const suffix = (rest) => ([...rest].every((c) => PUNCTUATION.includes(c)) ? rest : null);

function splitSuffix(word) {
  const body = trimEndWith(word, (c) => PUNCTUATION.includes(c));
  return [body, word.slice(body.length)];
}

const allDigits = (s) => s.length > 0 && /^[0-9]+$/.test(s);

function spellShort(word) {
  if (word === 'www' || word === 'WWW') return 'W-W-W';
  if (/^[\x00-\x7f]{2}$/.test(word)) return `${word[0].toUpperCase()}-${word[1].toUpperCase()}`;
  return word;
}

const DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

function spell(s) {
  return [...s]
    .filter((c) => c !== '-')
    .map((c) => (isDigit(c) ? DIGITS[c.charCodeAt(0) - 48] : c))
    .join(' ');
}

function range(body) {
  const dash = body.indexOf('-');
  if (dash < 0) return null;
  const from = body.slice(0, dash);
  const to = body.slice(dash + 1);
  const small = (s) =>
    s.length >= 2 && s.length <= 3 && allDigits(s) && (s.length === 1 || !s.startsWith('0'));
  const year = (s) => s.length === 4 && allDigits(s) && (s.startsWith('1') || s.startsWith('20'));
  let ok = false;
  if (small(from) && small(to)) ok = Number(from) < Number(to);
  else if (from.length === 4 && to.length === 4) ok = year(from) && year(to) && from < to;
  else if (from.length === 4 && to.length === 2)
    ok = year(from) && allDigits(to) && from.slice(2) < to;
  return ok ? `${from} to ${to}` : null;
}

const SCALES = [
  [1_000_000_000, 'billion'],
  [1_000_000, 'million'],
  [1000, 'thousand'],
];

export function numberWords(word) {
  if (word[0] === '0' && isDigit(word[1])) return null;
  const negative = word.startsWith('-');
  let rest = negative ? word.slice(1) : word;
  let end = rest.search(/[^0-9,]/);
  if (end < 0) end = rest.length;
  const int = trimEndWith(rest.slice(0, end), (c) => c === ',');
  if (int === '') return null;
  rest = rest.slice(int.length);
  let decimal = null;
  if (rest.startsWith('.')) {
    const after = rest.slice(1);
    let n = after.search(/[^0-9]/);
    if (n < 0) n = after.length;
    if (n > 0) {
      decimal = after.slice(0, n);
      rest = after.slice(n);
    }
  }
  const tail = suffix(rest);
  if (tail === null) return null;
  const value = Number(int.replace(/[^0-9]/g, ''));
  if (value >= 1_000_000_000_000) return null;
  let words;
  if (value < 1000 || (value >= 1900 && value < 2100)) words = String(value);
  else {
    const parts = [];
    for (const [div, name] of SCALES) {
      const group = Math.floor(value / div) % 1000;
      if (group) parts.push(`${group} ${name}`);
    }
    if (value % 1000) parts.push(String(value % 1000));
    words = parts.join(' ');
  }
  if (decimal !== null) words = `${words} point ${decimal}`;
  if (negative) words = `minus ${words}`;
  return [words, tail];
}

function numbers(word) {
  const [body, end] = splitSuffix(word);
  if (body.length >= 2 && body.startsWith('0') && allDigits(body)) return `${spell(body)}${end}`;
  const r = range(body);
  if (r) return `${r}${end}`;
  const read = numberWords(word);
  if (read) return `${read[0]}${read[1]}`;
  const dash = word.indexOf('-');
  if (dash < 0) return null;
  const head = word.slice(0, dash);
  const tail = word.slice(dash);
  if (!isAlphabetic([...tail.slice(1)][0])) return null;
  const words = numberWords(head);
  return words && words[1] === '' ? `${words[0]}${tail}` : null;
}

function digitRun(words) {
  const group = (s) => s.length >= 1 && s.length <= 8 && /^[0-9A-Z]+$/.test(s) && /[0-9]/.test(s);
  if (!words.length) return null;
  const plus = words[0].startsWith('+');
  const bodies = [];
  let end = '';
  for (const word of words) {
    let [body, tail] = splitSuffix(word);
    if (!bodies.length && plus) body = body.slice(1);
    if (!group(body)) break;
    bodies.push(body);
    if (tail) {
      end = tail;
      break;
    }
  }
  const years = bodies.every((b) => b.length === 4 && ['19', '20'].includes(b.slice(0, 2)));
  const long = bodies.length >= 3 && !years;
  if (!long && !(bodies.length === 2 && (plus || bodies.some((b) => b.startsWith('0')))))
    return null;
  const spelled = bodies.map(spell).join(', ');
  return [`${plus ? `plus ${spelled}` : spelled}${end}`, bodies.length];
}

function endingIn(prev, words) {
  if (prev === undefined || !['ending', 'ends', 'ended'].includes(prev.toLowerCase())) return null;
  if (words[0] !== 'in' || words[1] === undefined) return null;
  const [body, end] = splitSuffix(words[1]);
  if (!body.startsWith('-')) return null;
  const code = body.slice(1);
  if (!/[0-9]/.test(code) || !/^[0-9A-Z-]*$/.test(code)) return null;
  return [`in ${spell(code)}${end}`, 2];
}

const CURRENCIES = {
  $: ['dollar', 'dollars'],
  '€': ['euro', 'euros'],
  '£': ['pound', 'pounds'],
  '¥': ['yen', 'yen'],
  '¢': ['cent', 'cents'],
};

const SCALE_SUFFIXES = [
  ['thousand', 'thousand'],
  ['million', 'million'],
  ['billion', 'billion'],
  ['trillion', 'trillion'],
  ['bn', 'billion'],
  ['BN', 'billion'],
  ['mn', 'million'],
  ['tn', 'trillion'],
  ['k', 'thousand'],
  ['K', 'thousand'],
  ['m', 'million'],
  ['M', 'million'],
  ['b', 'billion'],
  ['B', 'billion'],
];

const SCALE_WORDS = ['thousand', 'million', 'billion', 'trillion'];

function splitSymbol(word) {
  const first = [...word][0];
  if (CURRENCY.has(first)) return [first, word.slice(first.length)];
  const last = [...word].at(-1);
  if (last !== undefined && CURRENCY.has(last)) return [last, word.slice(0, -last.length)];
  return null;
}

function splitScale(amount) {
  for (const [end, scale] of SCALE_SUFFIXES)
    if (amount.endsWith(end) && isDigit(amount.slice(0, -end.length).at(-1)))
      return [amount.slice(0, -end.length), scale];
  return null;
}

function withScale(amount, scale, symbol) {
  const read = numberWords(amount);
  if (read?.[1] !== '') return null;
  return `${read[0]} ${scale} ${CURRENCIES[symbol][1]}`;
}

function withCents(amount, symbol) {
  const dot = amount.lastIndexOf('.');
  if (dot < 0) return null;
  const whole = amount.slice(0, dot);
  const cents = amount.slice(dot + 1);
  const grouped = whole
    .split(',')
    .every((g, i) => allDigits(g) && (i === 0 ? g.length <= 3 : g.length === 3));
  if (!(allDigits(whole) || grouped)) return null;
  if (cents.length !== 2 || !allDigits(cents)) return null;
  const read = numberWords(whole);
  if (!read) return null;
  const [one, many] = CURRENCIES[symbol];
  const units = `${read[0]} ${read[0] === '1' ? one : many}`;
  if (cents === '00') return units;
  const spoken = cents.startsWith('0') ? cents.slice(1) : cents;
  if (read[0] === '0') return `${spoken} cents`;
  return `${units} and ${spoken} cents`;
}

function currency(word) {
  const [body, end] = splitSuffix(word);
  if (body.startsWith('-')) {
    const sym = [...body.slice(1)][0];
    if (CURRENCY.has(sym)) return currency(`${sym}-${body.slice(1 + sym.length)}${end}`);
  }
  const split = splitSymbol(body);
  if (!split) return null;
  const [symbol, amount] = split;
  const scaled = splitScale(amount);
  if (scaled) {
    const spoken = withScale(scaled[0], scaled[1], symbol);
    return spoken === null ? null : `${spoken}${end}`;
  }
  if (symbol !== '¢') {
    const spoken = withCents(amount, symbol);
    if (spoken) return `${spoken}${end}`;
  }
  const read = numberWords(amount);
  if (!read) return null;
  const [one, many] = CURRENCIES[symbol];
  const name = symbol === '¢' && amount === '1' ? one : many;
  return `${read[0]} ${name}${end}`;
}

function scaled(words) {
  if (words.length < 2) return null;
  const [scale, end] = splitSuffix(words[1]);
  const split = splitSymbol(words[0]);
  if (!split || !SCALE_WORDS.includes(scale)) return null;
  const spoken = withScale(split[1], scale, split[0]);
  return spoken === null ? null : [`${spoken}${end}`, 2];
}

const isSeparator = (c) => c === '-' || c === '_';

function dashGroups(body) {
  const groups = [];
  let start = 0;
  for (let i = 0; i < body.length; i++) {
    if (!isSeparator(body[i])) continue;
    if (i === start) return null;
    groups.push([body.slice(start, i), body[i]]);
    start = i + 1;
  }
  if (start === body.length) return null;
  groups.push([body.slice(start), null]);
  return groups;
}

function dayMonth(x, y) {
  if (x.length > 2 || y.length > 2) return false;
  const a = Number(x);
  const b = Number(y);
  return (a >= 1 && a <= 31 && b >= 1 && b <= 12) || (a >= 1 && a <= 12 && b >= 1 && b <= 31);
}

function looksLikeDate(groups) {
  if (groups.length !== 3) return false;
  const [[a], [b], [c]] = groups;
  return (
    (a.length === 4 && dayMonth(b, c)) || ((c.length === 4 || c.length === 2) && dayMonth(a, b))
  );
}

function dashedDigits(word) {
  const [body, end] = splitSuffix(word);
  if (!/^[0-9_-]*$/.test(body) || [...body].filter(isSeparator).length < 2) return null;
  const groups = dashGroups(body);
  if (!groups) return null;
  if (!body.includes('_') && looksLikeDate(groups)) return null;
  const words = [];
  for (const [group, separator] of groups) {
    if (group.length >= 4) words.push(...group);
    else words.push(group);
    if (separator === '_') words.push('underscore');
  }
  return `${words.join(' ')}${end}`;
}

function speakParts(part, words, spoken) {
  let start = 0;
  for (let i = 0; i < part.length; i++) {
    const word = spoken(part[i], i);
    if (word === null) continue;
    if (start < i) words.push(part.slice(start, i));
    words.push(word);
    start = i + 1;
  }
  if (start < part.length) words.push(part.slice(start));
}

function label(s) {
  const n = s.match(/^[A-Za-z0-9-]*/)[0].length;
  return n > 0 ? [s.slice(0, n), s.slice(n)] : null;
}

const EMAIL_USER = { '-': 'dash', '.': 'dot', '+': 'plus', _: 'underscore' };

function emails(word) {
  const at = word.indexOf('@');
  if (at < 0) return null;
  const user = word.slice(0, at);
  if (!user || !/^[A-Za-z0-9._+-]+$/.test(user)) return null;
  const first = label(word.slice(at + 1));
  if (!first) return null;
  const labels = [first[0]];
  let rest = first[1];
  for (;;) {
    const next = rest.startsWith('.') ? label(rest.slice(1)) : null;
    if (!next) break;
    labels.push(next[0]);
    rest = next[1];
  }
  const end = suffix(rest);
  if (end === null || labels.length < 2) return null;
  const tld = labels.pop();
  const words = [];
  speakParts(user, words, (c) => EMAIL_USER[c] ?? null);
  words.push('at');
  labels.forEach((l, i) => {
    if (i > 0) words.push('dot');
    speakParts(l, words, (c) => (c === '-' ? 'dash' : null));
  });
  words.push('dot');
  return `${words.join(' ')} ${spellShort(tld)}${end}`;
}

function run(s, keep) {
  let n = 0;
  for (const c of s) {
    if (!keep(c)) break;
    n += c.length;
  }
  return n > 0 ? [s.slice(0, n), s.slice(n)] : null;
}

function speakUrl(part, words) {
  let start = 0;
  for (let i = 0; i < part.length; i++) {
    const c = part[i];
    const word = c === '-' ? 'dash' : c === '/' ? 'slash' : isDigit(c) ? c : null;
    if (word === null) continue;
    if (start < i) words.push(spellShort(part.slice(start, i)));
    words.push(word);
    start = i + 1;
  }
  if (start < part.length) words.push(spellShort(part.slice(start)));
}

function urls(word) {
  const words = [];
  let rest = word;
  if (rest.startsWith('https://')) {
    words.push('H-T-T-P-S');
    rest = rest.slice(8);
  } else if (rest.startsWith('http://')) {
    words.push('H-T-T-P');
    rest = rest.slice(7);
  }
  if (words.length) words.push('colon', 'slash', 'slash');
  const scheme = words.length > 0;
  const labelChar = (c) => isAlphanumeric(c) || c === '-';
  const first = run(rest, labelChar);
  if (!first) return null;
  const labels = [first[0]];
  rest = first[1];
  for (;;) {
    const next = rest.startsWith('.') ? run(rest.slice(1), labelChar) : null;
    if (!next) break;
    labels.push(next[0]);
    rest = next[1];
  }
  const path = run(rest, (c) => labelChar(c) || c === '/');
  if (path) rest = path[1];
  const end = suffix(rest);
  if (end === null) return null;
  const tld = labels.at(-1);
  if (labels.length < 2 || !isTld(tld)) return null;
  const chars = [...tld];
  const capitalized = chars.some((c) => UPPERCASE.test(c)) && chars.some((c) => LOWERCASE.test(c));
  if (capitalized && !scheme && labels[0].toLowerCase() !== 'www') return null;
  labels.forEach((l, i) => {
    if (i > 0) words.push('dot');
    speakUrl(l, words);
  });
  if (path) speakUrl(path[0], words);
  return `${words.join(' ')}${end}`;
}

const ABBREVIATIONS = new Map([
  ['Lt', 'Lieutenant'],
  ['Lt.', 'Lieutenant'],
  ['Col.', 'Colonel'],
  ['Sgt.', 'Sergeant'],
  ['Sgt', 'Sergeant'],
  ['Capt.', 'Captain'],
  ['Cpl.', 'Corporal'],
  ['Gen.', 'General'],
  ['Adm.', 'Admiral'],
  ['Sen.', 'Senator'],
  ['Sens.', 'Senators'],
  ['Rep.', 'Representative'],
  ['Reps.', 'Representatives'],
  ['Gov.', 'Governor'],
  ['Pres.', 'President'],
  ['Det.', 'Detective'],
  ['Insp.', 'Inspector'],
  ['Supt.', 'Superintendent'],
  ['Ft.', 'Fort'],
  ['Mt.', 'Mount'],
  ['Jan.', 'January'],
  ['Feb.', 'February'],
  ['Mar.', 'March'],
  ['Apr.', 'April'],
  ['Jun.', 'June'],
  ['Jul.', 'July'],
  ['Aug.', 'August'],
  ['Sep.', 'September'],
  ['Sept.', 'September'],
  ['Oct.', 'October'],
  ['Nov.', 'November'],
  ['Dec.', 'December'],
  ['Mon.', 'Monday'],
  ['Tue.', 'Tuesday'],
  ['Tues.', 'Tuesday'],
  ['Wed.', 'Wednesday'],
  ['Thu.', 'Thursday'],
  ['Thur.', 'Thursday'],
  ['Thurs.', 'Thursday'],
  ['Fri.', 'Friday'],
  ['Sat.', 'Saturday'],
  ['Sun.', 'Sunday'],
  ['vol.', 'volume'],
  ['Vol.', 'Volume'],
  ['ed.', 'edition'],
  ['approx.', 'approximately'],
  ['Approx.', 'Approximately'],
  ['dept.', 'department'],
  ['Dept.', 'Department'],
  ['vs.', 'versus'],
  ['w/', 'with'],
  ['W/', 'With'],
  ['w/o', 'without'],
  ['W/o', 'Without'],
]);

function abbreviations(word) {
  const [body, end] = splitSuffix(word);
  const dotted = end.startsWith('.');
  const short = dotted ? word.slice(0, body.length + 1) : body;
  const expansion = ABBREVIATIONS.get(short);
  return expansion === undefined ? null : `${expansion}${dotted ? end.slice(1) : end}`;
}

function ranks(words) {
  if (words.length < 2 || words[0] !== 'Lt') return null;
  const [rank, end] = splitSuffix(words[1]);
  return rank === 'Col' ? [`Lieutenant Colonel${end}`, 2] : null;
}

const stretched = (c) => 'aeiouyhw'.includes(c.toLowerCase());

function elongations(word) {
  const [body, end] = splitSuffix(word);
  const chars = [...body];
  const windows = (list) => list.slice(2).map((c, i) => [list[i], list[i + 1], c]);
  if (
    !chars.every((c) => /^[A-Za-z]$/.test(c)) ||
    chars.every(isAsciiUpper) ||
    !windows(chars).some(([a, b, c]) => a === b && b === c && stretched(a))
  )
    return null;
  let out = '';
  chars.forEach((c, i) => {
    const repeat = i >= 1 && chars[i - 1] === c && stretched(c);
    const near = chars.slice(Math.max(0, i - 2), Math.min(chars.length, i + 3));
    const long = windows(near).some((w) => w.every((x) => x === c));
    if (!(repeat && long)) out += c;
  });
  return `${out}${end}`;
}

const RULES = [numbers, currency, dashedDigits, emails, urls, abbreviations, elongations];

export function rewriteWord(word) {
  for (const rule of RULES) {
    const out = rule(word);
    if (out !== null) return out;
  }
  return null;
}

const SLASH_UNITS = new Set(
  'km kg mg ml mL dL mol kJ kW kWh Hz min hr hrs sec ft lb lbs oz mi mph kph yd cm mm yr wk mo kcal rpm'.split(
    ' '
  )
);

function splitSlash(word) {
  const body = trimEndWith(word, (c) => isAsciiPunctuation(c) && c !== '/');
  const parts = body.split('/');
  const wordlike = (p) => [...p].length >= 2 && [...p].every(isAlphabetic) && !SLASH_UNITS.has(p);
  return parts.length > 1 && parts.every(wordlike)
    ? `${parts.join(' ')}${word.slice(body.length)}`
    : null;
}

const rewriteAll = (text) =>
  text
    .split(' ')
    .map((w) => rewriteWord(w) ?? w)
    .join(' ');

function spellSymbols(word) {
  const slashed = splitSlash(word);
  if (slashed !== null) return rewriteAll(slashed);
  if (!word.includes('@') && !word.includes('+')) return word;
  const res = new Appender();
  for (const c of word) {
    if (c === '@') res.pushSpoken('at');
    else if (c === '+') res.pushSpoken('plus');
    else res.push(c);
  }
  return rewriteAll(res.toString());
}

function rewriteText(text) {
  const words = text.split(' ');
  const out = [];
  let i = 0;
  while (i < words.length) {
    const rest = words.slice(i);
    const span = scaled(rest) ?? digitRun(rest) ?? endingIn(out.at(-1), rest) ?? ranks(rest);
    if (span) {
      out.push(span[0]);
      i += span[1];
      continue;
    }
    out.push(rewriteWord(words[i]) ?? spellSymbols(words[i]));
    i++;
  }
  return out.join(' ');
}

function stripClosingQuote(text) {
  const trimmed = trimEndWith(text, (c) => c === '"');
  if (trimmed.endsWith("'")) {
    const rest = trimmed.slice(0, -1);
    if (/[.!?,;]$/.test(rest)) return rest;
  }
  return trimmed;
}

function quotes(c, s) {
  if (DOUBLE_QUOTES.has(c)) s.res.push('"');
  else if (SINGLE_QUOTES.has(c)) s.res.push("'");
  else if ('‐‑‒―'.includes(c)) s.res.push('-');
  else return false;
  return true;
}

function stars(c, s) {
  if (c !== '★' && c !== '☆') return false;
  let full = c === '★' ? 1 : 0;
  let empty = c === '☆' ? 1 : 0;
  while (s.chars[s.i + 1] === '★' || s.chars[s.i + 1] === '☆') {
    if (s.chars[++s.i] === '★') full++;
    else empty++;
  }
  const half = s.chars[s.i + 1] === '½';
  if (half) s.i++;
  const n = half ? `${full}.5` : String(full);
  const of = empty > 0 ? full + empty + (half ? 1 : 0) : null;
  s.res.pushSpoken(of === null ? `${n} stars` : `${n} out of ${of} stars`);
  return true;
}

function degrees(c, s) {
  const { res, chars } = s;
  if ((c === '°' || c === 'º') && res.endsWithN()) {
    if (res.last() === '.') res.buffer.pop();
    res.pushSpoken(res.buffer.pop() === 'N' ? 'Number' : 'number');
    return true;
  }
  if (c !== '°') return false;
  res.pushSpoken(res.endsWithOne() ? 'degree' : 'degrees');
  const next = chars[s.i + 1];
  const scale = next === 'C' ? 'Celsius' : next === 'F' ? 'Fahrenheit' : null;
  if (scale && !isAlphanumeric(chars[s.i + 2])) {
    s.i++;
    res.pushStr(scale);
  }
  return true;
}

function letters(c, s) {
  const { res, chars } = s;
  if (c.codePointAt(0) >= 0xcc && FOLD.has(c)) res.push(FOLD.get(c));
  else if (SUBSCRIPTS.has(c)) res.push(SUBSCRIPTS.get(c));
  else if (SUPERSCRIPTS.has(c)) {
    let n = SUPERSCRIPTS.get(c);
    while (SUPERSCRIPTS.has(chars[s.i + 1])) n += SUPERSCRIPTS.get(chars[++s.i]);
    if (isAlphanumeric(s.prev) || s.prev === ')') res.pushSpoken(power(n));
    else res.pushStr(n);
  } else if (greek(c)) res.pushSpoken(greek(c));
  else if (SYMBOLS.has(c)) res.pushSpoken(SYMBOLS.get(c));
  else if (c === '±') res.pushSpoken('plus or minus');
  else if (c === '❤' || c === '♥') {
    if (chars[s.i + 1] === '\u{FE0F}') s.i++;
    res.pushSpoken('love');
  } else return false;
  return true;
}

const opensWord = (prev) => prev === undefined || isWhitespace(prev);

function marks(c, s) {
  const { res, prev } = s;
  const next = s.chars[s.i + 1];
  if (c === '~' && isDigit(next)) res.pushSpoken('about');
  else if (c === '*' && opensWord(prev) && isDigit(next)) res.pushSpoken('star');
  else if (SPACERS.has(c) || BULLETS.has(c)) res.pushWhitespace();
  else if (CURRENCY.has(c) && res.endsWithSpacedDigit() && !isDigit(next)) {
    res.popWhitespace();
    res.push(c);
  } else if (c === '…') res.push('.');
  else if (c === '=') res.pushSpoken('equals');
  else if (c === ':' && !opensWord(prev) && !opensWord(next)) res.push(':');
  else if (';:()'.includes(c)) {
    res.popWhitespace();
    res.push(',');
    res.pushWhitespace();
  } else return false;
  return true;
}

const CHARACTERS = [quotes, stars, degrees, letters, marks];

export function normalizeEnglish(input) {
  const s = { res: new Appender(), chars: [...input], i: 0, prev: undefined };
  for (; s.i < s.chars.length; s.i++) {
    const c = s.chars[s.i];
    if (!CHARACTERS.some((rule) => rule(c, s))) {
      if (isEmoji(c) || CONTROL.test(c) || isWhitespace(c)) s.res.pushWhitespace();
      else s.res.push(c);
    }
    s.prev = c;
  }
  return stripClosingQuote(rewriteText(s.res.toString()));
}

export function preparePrompt(input) {
  const trimmed = input.trim();
  if (!trimmed) return { text: '', framesAfterEos: 3 };
  const words = trimmed
    .replace(/[\n\r]/g, ' ')
    .split(/\p{White_Space}+/u)
    .filter(Boolean);
  const framesAfterEos = words.length <= 4 ? 3 : 1;
  let text = words.join(' ');
  const first = [...text][0];
  text = first.toUpperCase() + text.slice(first.length);
  if (isAlphanumeric([...text].at(-1))) text += '.';
  return { text, framesAfterEos };
}

export const MAX_TOKENS_PER_CHUNK = 50;
export const MAX_FIT_TOKENS = 200;

export const frameBudget = (tokens, frameRate = 12.5) => Math.ceil((tokens / 3 + 2) * frameRate);

export function splitSentences(tokenizer, text, max = MAX_TOKENS_PER_CHUNK) {
  const prepared = preparePrompt(text).text.trim();
  const tokens = tokenizer.encode(prepared);
  const marker = tokenizer.encode('.!...?');
  const ends = new Set(marker.length > 1 ? marker.slice(1) : marker);
  const bounds = [0];
  let previous = false;
  tokens.forEach((t, i) => {
    if (ends.has(t)) previous = true;
    else {
      if (previous) bounds.push(i);
      previous = false;
    }
  });
  bounds.push(tokens.length);
  const chunks = [];
  let current = '';
  let count = 0;
  for (let i = 0; i + 1 < bounds.length; i++) {
    const n = bounds[i + 1] - bounds[i];
    const sentence = tokenizer.decode(tokens.slice(bounds[i], bounds[i + 1]));
    if (!current) {
      current = sentence;
      count = n;
    } else if (count + n > max) {
      chunks.push(current.trim());
      current = sentence;
      count = n;
    } else {
      current += ` ${sentence}`;
      count += n;
    }
  }
  if (current) chunks.push(current.trim());
  return chunks;
}

function chunk(tokenizer, text, frameRate) {
  const { text: prepared, framesAfterEos } = preparePrompt(text);
  const tokens = tokenizer.encode(prepared);
  return { text, tokens, framesAfterEos, frameBudget: frameBudget(tokens.length, frameRate) };
}

const CLOSERS = '"\')]”’»';

function cut(tokenizer, words, minSide) {
  const ends = [];
  let total = 0;
  for (const w of words) {
    total += tokenizer.encode(w).length;
    ends.push(total);
  }
  const inner = ends.slice(0, -1).map((end, i) => [i, end]);
  const imbalance = ([, end]) => Math.abs(end - (total - end));
  const best = (list) =>
    list.reduce((a, b) => (a === null || imbalance(b) < imbalance(a) ? b : a), null);
  const after = (marks) =>
    best(
      inner.filter(
        ([i, end]) =>
          marks.includes(trimEndWith(words[i], (c) => CLOSERS.includes(c)).at(-1)) &&
          end >= minSide &&
          total - end >= minSide
      )
    );
  const found = after('.!?') ?? after(',;:') ?? best(inner);
  return found[0] + 1;
}

export function fit(tokenizer, chunks, max = MAX_FIT_TOKENS, frameRate = 12.5) {
  const fitted = [];
  const todo = [...chunks].reverse();
  while (todo.length) {
    const c = todo.pop();
    if (c.tokens.length <= max) {
      fitted.push(c);
      continue;
    }
    const words = c.text.split(/\p{White_Space}+/u).filter(Boolean);
    let halves;
    if (words.length >= 2) {
      const at = cut(tokenizer, words, Math.ceil(max / 4));
      halves = [words.slice(0, at).join(' '), words.slice(at).join(' ')];
    } else {
      const chars = [...c.text.trim()];
      const mid = Math.floor(chars.length / 2);
      if (mid === 0) {
        fitted.push(c);
        continue;
      }
      halves = [chars.slice(0, mid).join(''), chars.slice(mid).join('')];
    }
    for (const half of halves.reverse()) todo.push(chunk(tokenizer, half, frameRate));
  }
  return fitted;
}

export function plan(tokenizer, text, { frameRate = 12.5 } = {}) {
  const normalized = normalizeEnglish(text);
  const chunks = splitSentences(tokenizer, normalized).map((t) => chunk(tokenizer, t, frameRate));
  return fit(tokenizer, chunks, MAX_FIT_TOKENS, frameRate);
}

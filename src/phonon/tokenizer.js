const UNK_PENALTY = 10;
const SPACE = '\u2581';
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function trie(vocab) {
  const root = new Map();
  vocab.forEach(([piece], id) => {
    let node = root;
    for (const byte of encoder.encode(piece)) {
      if (!node.has(byte)) node.set(byte, new Map());
      node = node.get(byte);
    }
    node.set(-1, id);
  });
  return root;
}

const charLength = (byte) => (byte < 0x80 ? 1 : byte < 0xe0 ? 2 : byte < 0xf0 ? 3 : 4);

export function createTokenizer(json) {
  const { model, added_tokens: added = [] } = json;
  if (model?.type !== 'Unigram') throw new Error('the phonon tokenizer must be a Unigram model');
  const vocab = model.vocab;
  const ids = new Map(vocab.map(([piece], id) => [piece, id]));
  const root = trie(vocab);
  const unk = model.unk_id;
  const unkScore = vocab.reduce((m, [, score]) => Math.min(m, score), Infinity) - UNK_PENALTY;
  const special = new Set(added.filter((t) => t.special).map((t) => t.id));

  function lattice(bytes) {
    const size = bytes.length;
    const best = { score: new Float64Array(size + 1), start: new Int32Array(size + 1).fill(-1) };
    best.token = new Int32Array(size + 1);
    const offer = (at, from, id, s) => {
      if (best.start[at] >= 0 && s <= best.score[at]) return;
      best.score[at] = s;
      best.start[at] = from;
      best.token[at] = id;
    };
    for (let at = 0; at < size; ) {
      const base = best.score[at];
      const step = charLength(bytes[at]);
      let single = false;
      let node = root;
      for (let end = at; end < size; end++) {
        node = node.get(bytes[end]);
        if (!node) break;
        const id = node.get(-1);
        if (id === undefined) continue;
        offer(end + 1, at, id, vocab[id][1] + base);
        single ||= end + 1 - at === step;
      }
      if (!single) offer(at + step, at, unk, unkScore + base);
      at += step;
    }
    return best;
  }

  function viterbi(bytes) {
    const { start, token } = lattice(bytes);
    const pieces = [];
    let fused = [];
    for (let end = bytes.length; end > 0; end = start[end]) {
      if (token[end] === unk) {
        fused.push(bytes.subarray(start[end], end));
        continue;
      }
      if (fused.length) pieces.push({ unk: fused.reverse() });
      fused = [];
      pieces.push({ id: token[end] });
    }
    if (fused.length) pieces.push({ unk: fused.reverse() });
    return pieces.reverse();
  }

  function word(bytes, out) {
    for (const piece of viterbi(bytes)) {
      if (piece.id !== undefined) {
        out.push(piece.id);
        continue;
      }
      const raw = piece.unk.flatMap((p) => [...p]);
      const fallback = raw.map((b) =>
        ids.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`)
      );
      if (model.byte_fallback && fallback.every((id) => id !== undefined)) out.push(...fallback);
      else out.push(unk);
    }
  }

  function encode(text) {
    if (!text) return [];
    const spaced = `${SPACE}${text}`.replaceAll(' ', SPACE);
    const out = [];
    for (const piece of spaced.split(new RegExp(`(?=${SPACE})`, 'u')))
      if (piece) word(encoder.encode(piece), out);
    return out;
  }

  function decode(list) {
    const parts = [];
    let bytes = [];
    const flush = () => {
      if (bytes.length) parts.push(decoder.decode(Uint8Array.from(bytes)));
      bytes = [];
    };
    let first = true;
    for (const id of list) {
      if (special.has(id)) continue;
      const piece = vocab[id]?.[0] ?? '';
      const text = first ? piece.replaceAll(SPACE, '') : piece.replaceAll(SPACE, ' ');
      first = false;
      const byte = /^<0x([0-9A-F]{2})>$/.exec(text);
      if (byte) bytes.push(Number.parseInt(byte[1], 16));
      else {
        flush();
        parts.push(text);
      }
    }
    flush();
    return parts.join('');
  }

  return { encode, decode };
}

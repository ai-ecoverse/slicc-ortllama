const PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function byteTables() {
  const bytes = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = bytes.map((b) => String.fromCharCode(b));
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(String.fromCharCode(256 + extra++));
    }
  }
  const encoder = new Array(256);
  const decoder = new Map();
  bytes.forEach((b, i) => {
    encoder[b] = chars[i];
    decoder.set(chars[i], b);
  });
  return { encoder, decoder };
}

export function createTokenizer(json) {
  const vocab = new Map(Object.entries(json.model.vocab));
  const tokens = [];
  for (const [token, id] of vocab) tokens[id] = token;
  const special = new Set();
  for (const added of json.added_tokens ?? []) {
    tokens[added.id] = added.content;
    vocab.set(added.content, added.id);
    if (added.special) special.add(added.id);
  }
  const ranks = new Map();
  json.model.merges.forEach((merge, i) => {
    ranks.set(Array.isArray(merge) ? merge.join(' ') : merge, i);
  });
  const { encoder, decoder } = byteTables();
  const utf8 = new TextEncoder();
  const text = new TextDecoder();
  const cache = new Map();

  function bpe(word) {
    const hit = cache.get(word);
    if (hit) return hit;
    let parts = [...word];
    while (parts.length > 1) {
      let best = -1;
      let rank = Number.POSITIVE_INFINITY;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (r !== undefined && r < rank) {
          rank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      const next = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          next.push(a + b);
          i++;
        } else next.push(parts[i]);
      }
      parts = next;
    }
    const ids = parts.map((p) => vocab.get(p));
    cache.set(word, ids);
    return ids;
  }

  return {
    id: (token) => vocab.get(token),
    token: (id) => tokens[id],
    encode(input) {
      const out = [];
      for (const piece of input.match(PATTERN) ?? []) {
        const mapped = Array.from(utf8.encode(piece), (b) => encoder[b]).join('');
        out.push(...bpe(mapped));
      }
      return out.filter((id) => id !== undefined);
    },
    decode(ids) {
      const bytes = [];
      for (const id of ids) {
        if (special.has(id)) continue;
        const token = tokens[id];
        if (token === undefined || /^<\|.*\|>$/.test(token)) continue;
        for (const ch of token) {
          const b = decoder.get(ch);
          if (b !== undefined) bytes.push(b);
        }
      }
      return text.decode(new Uint8Array(bytes));
    },
  };
}

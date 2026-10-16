export function commonPrefix(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

export function createPrefixCache({ model, maxEntries = 2, maxTokens = Number.POSITIVE_INFINITY }) {
  let entries = [];

  const drop = (entry) => model.dispose(entry.past);

  return {
    get size() {
      return entries.length;
    },
    async take(ids) {
      let best = null;
      let length = 0;
      for (const entry of entries) {
        const n = Math.min(commonPrefix(entry.ids, ids), ids.length - 1);
        if (n > length) {
          best = entry;
          length = n;
        }
      }
      if (!best) return { past: model.emptyPast(), length: 0 };
      if (length < best.ids.length) return { past: await model.slice(best.past, length), length };
      entries = entries.filter((e) => e !== best);
      return { past: best.past, length };
    },
    put(ids, past) {
      entries.unshift({ ids, past });
      let tokens = 0;
      entries = entries.filter((entry, i) => {
        tokens += entry.ids.length;
        if (i < maxEntries && (i === 0 || tokens <= maxTokens)) return true;
        drop(entry);
        return false;
      });
    },
    clear() {
      for (const entry of entries) drop(entry);
      entries = [];
    },
  };
}

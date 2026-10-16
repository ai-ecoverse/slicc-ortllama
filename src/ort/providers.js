export function providersFrom(env, fallback) {
  const list = (env.ORTLLAMA_EP ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return list.length > 0 ? list : fallback;
}

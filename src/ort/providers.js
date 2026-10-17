export function providersFrom(env, fallback = null) {
  const list = (env.ORTLLAMA_EP ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return list.length > 0 ? list : fallback;
}

export function providerOrder(backend, { providers, engine, info }) {
  if (backend.overrideProviders) return { order: backend.overrideProviders, source: 'env' };
  if (providers) return { order: providers, source: 'hint' };
  const table =
    typeof engine?.providers === 'function' ? engine.providers(info) : engine?.providers;
  const preferred = table?.[backend.track];
  if (preferred) return { order: preferred, source: 'engine' };
  return { order: backend.executionProviders, source: 'default' };
}

export function providerAttempts(order, { fallbackProviders, availableProviders } = {}) {
  const [base] = fallbackProviders ?? [];
  const chain = [...new Set(base ? [...order, base] : order)];
  const usable = (ep) => ep === base || !availableProviders || availableProviders.includes(ep);
  return {
    attempts: chain.filter(usable).map((ep) => (base && ep !== base ? [ep, base] : [ep])),
    skipped: chain.filter((ep) => !usable(ep)),
  };
}

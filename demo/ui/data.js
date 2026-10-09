// Presets and the offline fallback (precomputed score() results), loaded tolerantly.

export const BUILTIN_PRESETS = [
  { key: '1', id: 'focused', name: 'Focused', prompt: 'Add rate limiting to user authentication on the /login endpoint: max 5 failed attempts per IP per minute, then return HTTP 429.' },
  { key: '2', id: 'scattered', name: 'Scattered', prompt: 'Build my React frontend, set up the Postgres database, add Google login, integrate Stripe payments, write the landing page copy, deploy it with Docker and make it look beautiful.' },
  { key: '3', id: 'negation', name: 'Negation', prompt: "Build a simple admin dashboard. Don't use React." },
  { key: '4', id: '3am', name: '3 AM', prompt: 'why is he not replying to my texts, should I text him again?' },
  { key: '5', id: 'mixed', name: 'Mixed', prompt: "create me an ai model and a react website for it and also why isn't she texting me?" },
];

export const keyOf = s => String(s ?? '').toLowerCase().replace(/[’‘]/g, "'").replace(/\s+/g, ' ').trim();
const ids = p => [p.key, p.id, p.name].map(keyOf);

export async function loadJson(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

/** Flattens any reasonable JSON shape into [{ key, name, prompt, ...rest }]. */
function entries(json) {
  if (Array.isArray(json)) return json.map((v, i) => (typeof v === 'string' ? { key: String(i + 1), prompt: v } : { key: String(i + 1), ...v }));
  if (json && typeof json === 'object') {
    for (const k of ['presets', 'results', 'items']) if (Array.isArray(json[k])) return entries(json[k]);
    return Object.entries(json)
      .filter(([, v]) => v && (typeof v === 'string' || typeof v === 'object'))
      .map(([k, v]) => (typeof v === 'string' ? { key: k, prompt: v } : { key: k, ...v }));
  }
  return [];
}

/**
 * demo/presets.json (owned by the Engine session) is the source of truth: its order sets the keys (1–9),
 * and it may add presets or retune wording. The built-ins are only used if it can't be loaded.
 */
export async function loadPresets(base = BUILTIN_PRESETS, url = 'presets.json') {
  try {
    const list = entries(await loadJson(url))
      .map((e, i) => ({ ...e, prompt: e.prompt ?? e.text }))
      .filter(e => typeof e.prompt === 'string' && e.prompt.trim())
      .slice(0, 9)
      .map((e, i) => {
        const known = base.find(p => [e.key, e.id, e.name].filter(Boolean).map(keyOf).some(x => ids(p).includes(x)));
        return { key: String(i + 1), id: e.id ?? known?.id ?? `preset-${i + 1}`, name: e.name ?? known?.name ?? `Preset ${i + 1}`, prompt: e.prompt };
      });
    if (list.length) return list;
    throw new Error(`${url}: no prompts inside`);
  } catch (err) {
    console.info('[demo] using built-in presets:', err.message);
    return base.map(p => ({ ...p }));
  }
}

const isResult = o => o && typeof o === 'object' && Array.isArray(o.clauses) && o.whole && typeof o.whole === 'object';

/** demo/data/presets.results.json -> { find(promptText) -> ScoreResult | null } */
export async function loadFallback(presets, url = 'data/presets.results.json') {
  const found = entries(await loadJson(url))
    .map(e => ({ ...e, result: isResult(e) ? e : isResult(e.result) ? e.result : isResult(e.score) ? e.score : null }))
    .filter(e => e.result);
  if (!found.length) throw new Error(`${url}: no score results inside`);
  const byPrompt = new Map(found.filter(e => e.prompt).map(e => [keyOf(e.prompt), e.result]));
  const byPreset = presets.map((p, i) => {
    const hit = byPrompt.get(keyOf(p.prompt))
      ?? found.find(e => [e.key, e.id, e.name].filter(Boolean).map(keyOf).some(x => ids(p).includes(x)))?.result
      ?? found[i]?.result;
    return hit ?? null;
  });
  return {
    count: byPreset.filter(Boolean).length,
    find(text) {
      const k = keyOf(text);
      const i = presets.findIndex(p => keyOf(p.prompt) === k);
      return byPrompt.get(k) ?? (i >= 0 ? byPreset[i] : null);
    },
  };
}

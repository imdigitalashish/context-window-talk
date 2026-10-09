import { Map3D } from './map3d.js';
import { Flights } from './flights.js';
import { BUILTIN_PRESETS, keyOf, loadFallback, loadJson, loadPresets } from './data.js';

// ── Engine switch: the ONE place the engine is chosen. Real engine: '../engine.js' ──
const ENGINE_URL = '../engine.js';
// ─────────────────────────────────────────────────────────────────────────────

const q = new URLSearchParams(location.search);
const ENGINE = { mock: '../engine.mock.js', real: '../engine.js' }[q.get('engine')] ?? ENGINE_URL; // ?engine=mock|real for testing
const IS_MOCK = ENGINE.includes('mock');
const LOAD_TIMEOUT_MS = 60000;
const FLIGHT_MS = 1300;
const BAR_MAX = 0.6; // bar length = sim / BAR_MAX; fixed (real scores top out near 0.55) so runs stay comparable
const TOP_BARS = 6;
const SPIN = { idle: 0.6, result: 0.2 }; // auto-rotate, revolutions per minute

const $ = id => document.getElementById(id);
const stage = $('stage'), card = $('card'), promptEl = $('prompt'), chipsEl = $('chips'), presetsEl = $('presets');
const statusText = $('statusText'), banner = $('banner'), toastEl = $('toast');
const nTouched = $('nTouched'), focusEl = $('focus'), focusNote = $('focusNote'), barsEl = $('bars'), barsNote = $('barsNote');
const loaderTitle = $('loaderTitle'), loaderBar = $('loaderBar'), loaderDetail = $('loaderDetail');

const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
const still = () => motionQuery.matches || q.has('still');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clamp01 = x => Math.min(1, Math.max(0, x));

let map = null;
try {
  map = new Map3D({ host: $('map'), canvas: $('gl'), labels: $('labels'), leaders: $('leaders') });
} catch (err) {
  console.error('[demo] WebGL unavailable', err);
}
const flights = new Flights({ map, svg: $('trails'), layer: $('marks') });
const partOf = id => map?.part(id);

let engine = null;
let presets = BUILTIN_PRESETS.map(p => ({ ...p }));
let fallback = null;
let live = false;
let phase = 'loading';
let runId = 0;
let pending = null;
let shown = null;     // { text, result } on screen
let lastRun = null;   // { text, name, focus } of the previous finished run
let activePreset = -1;
const cache = new Map();

/* ───────── status ───────── */

const idlePhase = () => (live ? 'ready' : 'error');
function setPhase(p, text) {
  phase = p;
  stage.dataset.phase = p;
  statusText.textContent = text ?? {
    loading: 'Loading…',
    warming: 'Warming up…',
    running: 'Running…',
    ready: IS_MOCK ? 'Mock engine' : 'Live model',
    error: 'Offline',
  }[p];
}

let toastTimer = 0;
function toast(msg, ms = 2800) {
  toastEl.textContent = msg;
  toastEl.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('on'), ms);
}

function onProgress(e) {
  if (phase !== 'loading') return;
  const frac = e?.total > 0 ? clamp01(e.loaded / e.total) : null;
  const isMap = e?.stage === 'map';
  loaderTitle.textContent = isMap ? 'Loading the knowledge map' : 'Loading the embedding model';
  loaderBar.style.transform = `scaleX(${frac ?? 0})`;
  const mb = n => (n / 1e6).toFixed(1);
  const amount = e?.total > 2e5 ? `${mb(e.loaded)} of ${mb(e.total)} MB` : frac != null ? `${Math.round(frac * 100)}%` : '…';
  loaderDetail.textContent = `${amount} · ${isMap ? 'questions + vectors' : 'all-MiniLM-L6-v2, running offline'}`;
  statusText.textContent = frac != null ? `Loading ${Math.round(frac * 100)}%` : 'Loading…';
}

/* ───────── stage ───────── */

function fit() {
  const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  stage.style.transform = `translate(${(innerWidth - 1920 * k) / 2}px,${(innerHeight - 1080 * k) / 2}px) scale(${k})`;
  map?.resize(k);
}

function showMap(data) {
  if (!map || !data) return;
  map.setMap(data);
  const n = data.points?.length ?? 0;
  $('mapMeta').textContent = `${n.toLocaleString('en-US')} questions · ${map.parts.length} parts`;
}

function stagePoint(el) {
  const s = stage.getBoundingClientRect(), r = el.getBoundingClientRect(), k = s.width / 1920;
  return { x: (r.left + r.width / 2 - s.left) / k, y: (r.top + r.height / 2 - s.top) / k };
}

/* ───────── presets + prompt ───────── */

function renderPresets() {
  presetsEl.textContent = '';
  presets.forEach((p, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.title = p.prompt;
    b.innerHTML = '<kbd></kbd><span></span>';
    b.querySelector('kbd').textContent = String(i + 1);
    b.querySelector('span').textContent = p.name;
    b.addEventListener('click', () => { b.blur(); loadPreset(i); });
    presetsEl.append(b);
  });
  promptEl.placeholder = `Type a prompt, or press ${keysLabel()}`;
  markPreset(activePreset);
}

function markPreset(i) {
  activePreset = i;
  [...presetsEl.children].forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
}

const presetIndexOf = text => presets.findIndex(p => keyOf(p.prompt) === keyOf(text));
const keysLabel = () => (presets.length > 1 ? `1–${presets.length}` : '1');

function loadPreset(i) {
  if (!presets[i]) return;
  clearResults();
  markPreset(i);
  promptEl.value = presets[i].prompt;
  card.dataset.view = 'input';
  promptEl.blur();
}

function editPrompt() {
  const text = shown?.text ?? promptEl.value;
  clearResults();
  promptEl.value = text;
  card.dataset.view = 'input';
  promptEl.focus();
  promptEl.setSelectionRange(text.length, text.length);
}

function clearAll() {
  clearResults();
  promptEl.value = '';
  markPreset(-1);
  card.dataset.view = 'input';
  promptEl.blur();
  toastEl.classList.remove('on');
}

/* ───────── results ───────── */

function setCount(n) {
  nTouched.textContent = n == null ? '–' : String(n);
  nTouched.classList.toggle('empty', n == null);
}

const runLabel = r => `${r.name ?? 'last run'}: ${r.focus.toFixed(2)}`;

function clearResults() {
  runId++;
  flights.clear();
  map?.setAll('idle');
  map?.setRotation(SPIN.idle);
  chipsEl.textContent = '';
  shown = null;
  setCount(null);
  focusEl.textContent = '–';
  focusEl.classList.add('empty');
  focusNote.textContent = lastRun ? runLabel(lastRun) : '';
  barsEl.innerHTML = '<li class="empty">Run a prompt to see this</li>';
  barsNote.textContent = '';
  if (phase === 'running') setPhase(idlePhase());
}

/** Chips for the clauses; if they overflow the card, the tail collapses into one "+N more" chip. */
function renderChips(texts) {
  chipsEl.textContent = '';
  chipsEl.className = 'chips';
  card.dataset.view = 'chips';
  const make = (n, text) => {
    const li = document.createElement('li');
    li.className = 'chip pending';
    li.innerHTML = '<b></b><span></span>';
    li.querySelector('b').textContent = n;
    li.querySelector('span').textContent = text;
    chipsEl.append(li);
    return li;
  };
  let els = texts.map((t, i) => make(String(i + 1), t));
  const overflows = () => chipsEl.scrollHeight > chipsEl.clientHeight + 1;
  if (overflows()) chipsEl.classList.add('dense');
  if (overflows()) chipsEl.classList.add('tight');
  if (overflows()) {
    let keep = els.length - 1;
    for (;;) {
      els.slice(keep).forEach(el => el.remove());
      const more = make('+', `${texts.length - keep} more clauses`);
      if (!overflows() || keep <= 1) { els = [...els.slice(0, keep), more]; break; }
      more.remove();
      keep--;
    }
  }
  return els;
}

const xyzOf = o => (Array.isArray(o?.xyz) ? o.xyz : Array.isArray(o?.xy) ? [o.xy[0], o.xy[1], 0.5] : null);

function buildLegs(clauses, chipEls) {
  return clauses.map((c, i) => {
    const chip = chipEls[Math.min(i, chipEls.length - 1)];
    const part = partOf(c.part);
    return {
      kind: 'clause', chip, n: i + 1,
      from: stagePoint(chip.querySelector('b')),
      xyz: xyzOf(c) ?? [0.5, 0.5, 0.5],
      color: part?.color ?? '#69717E',
      on: part?.on ?? '#fff',
      touched: !!c.touched,
    };
  });
}

function showWhole(whole, text) {
  const sims = whole?.sims ?? {};
  const parts = map?.parts.length ? map.parts : Object.keys(sims).map(id => ({ id, label: id, color: '#69717E' }));
  const rows = parts.map(p => ({ id: p.id, label: p.label, color: p.color, sim: Number(sims[p.id] ?? 0) })).sort((a, b) => b.sim - a.sim);
  const top = rows.slice(0, TOP_BARS);
  barsEl.textContent = '';
  for (const r of top) {
    const li = document.createElement('li');
    li.className = 'bar';
    li.style.setProperty('--c', r.color);
    li.innerHTML = '<span class="l"></span><span class="t"><i class="f"></i></span><span class="v"></span>';
    li.querySelector('.l').textContent = r.label;
    li.querySelector('.v').textContent = r.sim.toFixed(2);
    barsEl.append(li);
  }
  barsNote.textContent = rows.length > TOP_BARS ? `top ${TOP_BARS} of ${rows.length}` : '';
  const grow = () => barsEl.querySelectorAll('.f').forEach((f, i) => { f.style.transform = `scaleX(${clamp01(top[i].sim / BAR_MAX)})`; });
  if (still()) grow(); else requestAnimationFrame(() => requestAnimationFrame(grow));

  const best = whole?.top?.part != null ? whole.top : rows[0] && { part: rows[0].id, sim: rows[0].sim };
  const focus = best && Number.isFinite(Number(best.sim)) ? Number(best.sim) : null;
  focusEl.textContent = focus == null ? '–' : focus.toFixed(2);
  focusEl.classList.toggle('empty', focus == null);
  focusNote.textContent = lastRun && keyOf(lastRun.text) !== keyOf(text) ? runLabel(lastRun) : '';
  if (focus != null) lastRun = { text, name: presets[presetIndexOf(text)]?.name, focus };
}

async function run(text = promptEl.value) {
  text = String(text ?? '').trim();
  if (!text) { toast(`Type a prompt, or press ${keysLabel()}`); return; }
  clearResults();
  const id = runId;
  promptEl.value = text;
  promptEl.blur();
  markPreset(presetIndexOf(text));

  if (phase === 'loading' || phase === 'warming') {
    pending = text;
    toast('Model still loading: this will run as soon as it is ready');
    return;
  }

  let result = null;
  if (!live) {
    result = fallback?.find(text) ?? null;
    if (!result) {
      toast(fallback ? `The live model is offline, so only presets ${keysLabel()} work` : 'The live model is offline and the saved results are missing');
      card.dataset.view = 'input';
      return;
    }
  } else {
    result = cache.get(keyOf(text));
    if (!result) {
      try { const pre = engine.splitClauses(text); if (pre?.length) renderChips(pre); } catch { /* chips come with the result */ }
      setPhase('running', 'Embedding…');
      try {
        result = await engine.score(text);
        cache.set(keyOf(text), result);
      } catch (err) {
        console.error('[demo] score failed', err);
        if (id !== runId) return;
        setPhase(idlePhase());
        card.dataset.view = 'input';
        toast(`Scoring failed: ${err?.message ?? err}`);
        return;
      }
      if (id !== runId) return;
    }
  }
  await present(text, result, id);
}

async function present(text, result, id) {
  shown = { text, result };
  const clauses = Array.isArray(result?.clauses) ? result.clauses : [];
  const chipEls = renderChips(clauses.map(c => c.text));
  setPhase('running');
  map?.setAll('dim');
  map?.setRotation(0);
  if (!still()) await sleep(350);
  if (id !== runId) return;

  const legs = buildLegs(clauses, chipEls);
  const touched = new Set();
  setCount(0);
  let done = await flights.play(legs, {
    reduced: still(),
    duration: FLIGHT_MS,
    stagger: legs.length <= 1 ? 0 : Math.max(320, Math.min(650, 2800 / (legs.length - 1))),
    onLand: i => {
      const c = clauses[i], chip = legs[i].chip;
      if (c.touched) { chip.style.setProperty('--c', legs[i].color); chip.style.setProperty('--on', legs[i].on); }
      chip.classList.remove('pending');
      chip.classList.add(c.touched ? 'landed' : 'miss');
      if (!c.touched) return;
      map?.light(c.part);
      if (!touched.has(c.part)) { touched.add(c.part); setCount(touched.size); }
    },
  });
  if (!done || id !== runId) return;

  const wholeXyz = xyzOf(result.whole);
  if (wholeXyz) {
    const cr = card.getBoundingClientRect(), s = stage.getBoundingClientRect(), k = s.width / 1920;
    done = await flights.play([{ kind: 'whole', label: 'whole prompt', xyz: wholeXyz, color: '#111418',
      from: { x: (cr.left - s.left) / k + 2, y: (cr.top + cr.height / 2 - s.top) / k } }], { reduced: still(), duration: FLIGHT_MS });
    if (!done || id !== runId) return;
  }

  const final = Array.isArray(result.partsTouched) ? result.partsTouched : [...touched];
  final.forEach(p => map?.light(p));
  setCount(final.length);
  showWhole(result.whole, text);
  map?.setRotation(SPIN.result);
  setPhase(idlePhase());
}

/** Clicker-friendly: → runs the loaded prompt, or loads the next preset. */
function advance(dir) {
  if (dir > 0 && card.dataset.view === 'input' && promptEl.value.trim()) { run(); return; }
  const n = presets.length;
  loadPreset(activePreset < 0 ? (dir > 0 ? 0 : n - 1) : (activePreset + dir + n) % n);
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

addEventListener('keydown', e => {
  if (e.isComposing) return;
  const typing = e.target === promptEl;
  if (e.key === 'Escape') { e.preventDefault(); clearAll(); return; }
  if (e.key === 'Enter' && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    run(card.dataset.view === 'chips' && shown ? shown.text : promptEl.value);
    return;
  }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (/^[1-9]$/.test(e.key) && presets[+e.key - 1]) { e.preventDefault(); loadPreset(+e.key - 1); }
  else if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFullscreen(); }
  else if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); advance(1); }
  else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); advance(-1); }
  else if (e.key === '/') { e.preventDefault(); editPrompt(); }
});
card.addEventListener('click', e => { if (card.dataset.view === 'chips' && !e.target.closest('button')) editPrompt(); });
$('run').addEventListener('click', () => run());
addEventListener('resize', fit);
motionQuery.addEventListener?.('change', () => { if (map) map.reduced = still(); });

/* ───────── boot ───────── */

const withTimeout = (p, ms, msg) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(msg)), ms))]);

async function goOffline(err) {
  live = false;
  let mapData = null;
  try { mapData = engine?.getMap?.(); } catch { /* not initialised */ }
  if (!mapData) { try { mapData = await loadJson('data/map.json'); } catch (e) { console.warn('[demo] no map', e); } }
  if (mapData && !map?.parts.length) showMap(mapData);
  try { fallback = await loadFallback(presets); } catch (e) { console.warn('[demo] no saved results', e); fallback = null; }
  const why = String(err?.message ?? err ?? 'unknown error').replace(/\s+/g, ' ').slice(0, 70);
  banner.textContent = fallback
    ? `Live model didn't load (${why}). Showing saved results for presets ${keysLabel()}.`
    : `Live model didn't load (${why}), and the saved results are missing too.`;
  banner.hidden = false;
  setPhase('error');
}

async function boot() {
  if (map) map.reduced = still();
  else $('loaderTitle').textContent = 'This browser cannot show the 3-D map (WebGL is off)';
  fit();
  renderPresets();
  setPhase('loading');
  presets = await loadPresets(presets);
  renderPresets();
  try {
    if (q.has('fail')) throw new Error('simulated with ?fail');
    engine = await import(ENGINE);
    engine.onProgress?.(onProgress);
    await withTimeout(engine.init(), LOAD_TIMEOUT_MS, 'loading took over a minute');
    showMap(engine.getMap());
    setPhase('warming');
    const warm = presets[0]?.prompt;
    if (warm) cache.set(keyOf(warm), await withTimeout(engine.score(warm), 30000, 'warm-up took too long'));
    live = true;
    setPhase('ready');
  } catch (err) {
    console.error('[demo] live engine unavailable:', err);
    await goOffline(err);
  }
  if (pending) { const t = pending; pending = null; run(t); }
}

boot();

// Lost in the middle: every number on this page is read from data/*.json (measured runs). Nothing is computed live.

const $ = id => document.getElementById(id);
const stage = $('stage'), viz = $('viz');
const NS = 'http://www.w3.org/2000/svg';
const q = new URLSearchParams(location.search);
const motion = matchMedia('(prefers-reduced-motion: reduce)');

// ── geometry (stage px). The prompt's fixed parts get fixed-width caps; the document between them is linear. ──
const X0 = 444, X1 = 1852, CAP = 64;
const XD0 = X0 + CAP, XD1 = X1 - CAP;
const docX = f => XD0 + f * (XD1 - XD0);
const R = { top: 226, bottom: 446 };         // recall plot: 10/10 at top, 0 at bottom
const S = { top: 516, h: 52 };               // document strip
const A = { top: 724, bottom: 960 };         // attention plot (log scale)
const NEEDLE_TOP = 204, NEEDLE_BOTTOM = 968;

const DEFAULTS = { L: '8192', look: true, fix: false, d: 0 };
const state = { ...DEFAULTS };
let recall = null, attn = null;

// ── tiny SVG helpers ──
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  parent?.appendChild(e);
  return e;
}
function text(parent, x, y, str, cls = '', anchor = 'start') {
  const t = el('text', { x, y, class: cls, 'text-anchor': anchor }, parent);
  t.textContent = str;
  return t;
}
const clear = g => { while (g.firstChild) g.removeChild(g.firstChild); };
const pathOf = pts => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
const fmtInt = n => n.toLocaleString('en-US');
const kTok = L => `${Math.round(+L / 1024)}K`;
const pct = f => `${Math.round(f * 100)}%`;
function setPath(p, d) {
  // CSS `d` transitions in Chromium give a smooth morph; the attribute is the fallback.
  if (CSS.supports?.('d', 'path("M0,0")')) p.style.d = `path("${d}")`;
  else p.setAttribute('d', d);
}

// ── data access ──
const condKey = (look = state.look, fix = state.fix) => `${look ? 'lookalikes' : 'single'}-${fix ? 'both' : 'after'}`;
const recallFor = (key = condKey(), L = state.L) => recall?.conditions?.[key]?.by_length?.[L] ?? null;
const attnFor = (key = condKey()) => attn?.runs?.[key] ?? null;
const lengths = () => Object.keys(recall.conditions[condKey(true, false)]?.by_length ?? {}).sort((a, b) => a - b);
const fixAvailable = () => !!recallFor(condKey(state.look, true));
const lookAvailable = () => !!recallFor(condKey(!state.look, state.fix)) || !!recallFor(condKey(!state.look, false));
const N_DEPTHS = () => recall.depths.length;

// ── static scaffolding ──
function drawStatic() {
  const g = $('gStrip'); clear(g);
  el('rect', { x: X0, y: S.top, width: X1 - X0, height: S.h, rx: 10, class: 'bar' }, g);
  const capPath = (x, w, left) => {
    const r = 10, y = S.top, h = S.h;
    return left
      ? `M${x + w},${y} H${x + r} Q${x},${y} ${x},${y + r} V${y + h - r} Q${x},${y + h} ${x + r},${y + h} H${x + w} Z`
      : `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
  };
  el('path', { d: capPath(X0 + .75, CAP, true), class: 'capfill' }, g);
  el('path', { d: capPath(XD1, CAP - .75, false), class: 'capfill' }, g);
  // document texture: rows of word-like blocks (deterministic)
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let row = 0; row < 3; row++) {
    let x = XD0 + 10 + rnd() * 8;
    const y = S.top + 12 + row * 12;
    while (x < XD1 - 14) {
      const w = Math.min(10 + rnd() * 34, XD1 - 12 - x);
      if (w > 4) el('rect', { x: x.toFixed(1), y, width: w.toFixed(1), height: 5, rx: 2.5, class: 'word' }, g);
      x += w + 6 + (rnd() < 0.08 ? 18 : 0);
    }
  }
  el('g', { id: 'gPins' }, g);
  // labels under the strip
  text(g, XD0 - 8, S.top + S.h + 32, 'prompt', 'lbl', 'end');
  text(g, XD1 + 8, S.top + S.h + 32, 'question', 'lbl', 'start');
  [0, .25, .5, .75, 1].forEach(f => text(g, docX(f), S.top + S.h + 32, pct(f), '', f === 0 ? 'start' : f === 1 ? 'end' : 'middle'));

  // needle: one line through all three panels, a knob on the strip, a dot on each curve
  const n = $('gNeedle'); clear(n);
  const mover = el('g', { id: 'needleMover', class: 'glide' }, n);
  el('line', { x1: 0, x2: 0, y1: NEEDLE_TOP, y2: NEEDLE_BOTTOM, class: 'n-line' }, mover);
  el('circle', { cx: 0, cy: S.top + S.h / 2, r: 15, class: 'n-knob' }, mover);
  el('circle', { id: 'curHalo', cx: 0, cy: 0, r: 22, class: 'n-halo glide' }, n);
  el('circle', { id: 'curDot', cx: 0, cy: 0, r: 11, class: 'n-cur glide' }, n);
  el('circle', { id: 'attDot', cx: 0, cy: 0, r: 9, class: 'n-cur glide' }, n);
  const flag = el('g', { id: 'needleFlag', class: 'glide' }, n);
  el('rect', { x: 0, y: -17, width: 10, height: 34, rx: 8, class: 'n-tagbg' }, flag);
  text(flag, 0, 8, '', 'n-say', 'middle');
  const tag = el('g', { id: 'attTag', class: 'glide' }, n);
  el('rect', { x: 0, y: -19, width: 10, height: 34, rx: 8, class: 'n-tagbg' }, tag);
  text(tag, 0, 7, '', 'n-tag', 'middle');
}

// ── recall plot ──
const ry = c => R.bottom - (c / 10) * (R.bottom - R.top);

function drawRecall() {
  const g = $('gRecall'); clear(g);
  const cur = recallFor();
  const n = cur.points[0].n;
  const yOf = c => R.bottom - (c / n) * (R.bottom - R.top);
  [0, .5, 1].forEach(f => {
    const y = yOf(f * n);
    el('line', { x1: XD0, x2: XD1, y1: y, y2: y, class: 'grid' }, g);
    text(g, XD0 - 26, y + 8, `${Math.round(f * n)}/${n}`, '', 'end');
  });
  const pts = cur.points.map(p => [docX(p.depth), yOf(p.correct)]);
  if (state.fix) {
    const base = recallFor(condKey(state.look, false));
    if (base) {
      const bp = base.points.map(p => [docX(p.depth), yOf(p.correct)]);
      el('path', { d: pathOf(bp), class: 'u-ghost' }, g);
      bp.forEach(([x, y]) => el('circle', { cx: x, cy: y, r: 5, class: 'u-ghost-dot' }, g));
      const [lx, ly] = bp[Math.floor(bp.length / 2)];
      const lab = text(g, lx, Math.min(ly + 40, R.bottom - 8), 'question only after', 'lbl halo', 'middle');
      lab.style.fill = 'var(--ink-3)';
    }
  }
  el('path', { d: `${pathOf(pts)}L${XD1},${R.bottom}L${XD0},${R.bottom}Z`, class: 'u-area' }, g);
  el('path', { d: pathOf(pts), class: 'u-line' }, g);
  pts.forEach(([x, y], i) => {
    const c = el('circle', { cx: x, cy: y, r: 6, class: 'u-dot' }, g);
    c.addEventListener('click', () => setDepth(i));
  });
  return { yOf };
}

// ── attention plot (log scale, in multiples of an even share) ──
let aScale = null;
const needleRatio = s => s.needle_mass / (s.needle_span[1] - s.needle_span[0]) / s.uniform;  // per needle token

function attnScale(run) {
  const s = run.no_needle, u = s.uniform;
  const vals = [...s.bins, ...s.tail_bins, s.prefix_rest_per_token].map(v => v / u).concat(run.by_depth.map(needleRatio));
  const dLo = Math.floor(Math.log10(Math.min(...vals))), dHi = Math.ceil(Math.log10(s.sink / u));
  const y = r => A.bottom - (Math.log10(Math.max(r, 10 ** dLo)) - dLo) / (dHi - dLo) * (A.bottom - A.top);
  return { dLo, dHi, y };
}
const ratioLabel = r => r >= 10 ? `${fmtInt(Math.round(r))}×` : r >= 1 ? `${r.toFixed(1)}×` : `${r.toFixed(2)}×`;

function drawAttn() {
  const g = $('gAttn'); clear(g);
  const run = attnFor();
  if (!run) return;
  aScale = attnScale(run);
  for (let d = aScale.dLo; d <= aScale.dHi; d++) {
    const y = aScale.y(10 ** d);
    el('line', { x1: X0, x2: X1, y1: y, y2: y, class: d === 0 ? 'even' : 'grid' }, g);
    text(g, X0 - 10, y + 8, d < 0 ? `${(10 ** d).toFixed(-d)}×` : `${fmtInt(10 ** d)}×`, '', 'end');
  }
  // the document with no needle in it: mean attention per token in 60 bins
  const s = run.no_needle, u = s.uniform, [d0, d1] = s.doc_span, span = d1 - d0;
  const pts = s.bins.map((v, i) => [docX(((s.bin_edges[i] + s.bin_edges[i + 1]) / 2 - d0) / span), aScale.y(v / u)]);
  const tail = s.tail_bins.map((v, i) => [XD1 + (i + .5) / s.tail_bins.length * (CAP - 8), aScale.y(v / u)]);
  const area = el('path', { class: 'a-area glide' }, g);
  setPath(area, `${pathOf(pts)}L${pts.at(-1)[0]},${A.bottom}L${pts[0][0]},${A.bottom}Z`);
  setPath(el('path', { class: 'a-line glide' }, g), pathOf(pts));
  setPath(el('path', { class: 'a-cap glide' }, g), pathOf([pts.at(-1), ...tail]));
  const yp = aScale.y(s.prefix_rest_per_token / u);
  el('line', { x1: X0 + 30, x2: XD0 - 4, y1: yp, y2: yp, class: 'a-cap' }, g);
  // token 0: the attention sink
  const ys = aScale.y(s.sink / u), xs = X0 + 14;
  el('line', { x1: xs, x2: xs, y1: A.bottom, y2: ys, class: 'sink-stem' }, g);
  el('circle', { cx: xs, cy: ys, r: 9, class: 'sink-dot' }, g);
  el('path', { d: `M${xs + 14},${ys} H${XD0 + 22}`, class: 'leader' }, g);
  const sl = text(g, XD0 + 30, ys + 8, '', 'lbl halo');
  const b = el('tspan', { class: 'lbl-strong' }, sl); b.textContent = 'attention sink: ';
  el('tspan', {}, sl).textContent = 'parked attention, not “reads your first line harder”';
  // recency: the most recent tokens (end of the document + the question itself)
  const [ex, ey] = tail.at(-1);
  text(g, ex + 4, Math.max(ey - 20, A.top - 8), 'recent tokens', 'lbl-strong halo', 'end');
  const cap = text(g, XD0, A.bottom + 22, '', 'lbl', 'start');
  const t1 = el('tspan', { class: 'lbl-strong' }, cap); t1.textContent = 'line: ';
  el('tspan', {}, cap).textContent = 'the document, no needle  ·  ';
  const t2 = el('tspan', { class: 'lbl-acc' }, cap); t2.textContent = 'dots: ';
  el('tspan', {}, cap).textContent = 'the needle at each spot  ·  per token, × an even share (dashed = 1×)';
  // the needle's own attention (per needle token) at each of the 11 measured spots
  const np = run.by_depth.map(d => [docX(d.depth), aScale.y(needleRatio(d))]);  // nominal depth, to sit on the needle line
  el('path', { d: pathOf(np), class: 'nd-line' }, g);
  np.forEach(([x, y], i) => {
    const c = el('circle', { cx: x, cy: y, r: 6, class: 'nd-dot' }, g);
    c.addEventListener('click', () => setDepth(i));
  });
}

// ── needle + readouts ──
function renderNeedle(yOf) {
  const p = recallFor().points[state.d];
  const x = docX(p.depth);
  $('needleMover').style.transform = `translate(${x}px,0)`;
  for (const id of ['curDot', 'curHalo']) $(id).style.transform = `translate(${x}px,${yOf(p.correct)}px)`;
  const flag = $('needleFlag'), ft = flag.querySelector('text'), fr = flag.querySelector('rect');
  if (!ft.dataset.done) {
    el('tspan', { class: 'n-tag' }, ft).textContent = 'needle: ';
    el('tspan', {}, ft).textContent = `“${recall.needles[0].statement}”`;
    ft.dataset.done = '1';
  }
  const fw = ft.getComputedTextLength?.() || 600;
  fr.setAttribute('x', -fw / 2 - 14); fr.setAttribute('width', fw + 28);
  flag.style.transform = `translate(${Math.min(Math.max(x, X0 + fw / 2 + 14), X1 - fw / 2 - 14)}px,${S.top - 20}px)`;
  const s = attnFor()?.by_depth?.[state.d];
  const show = !!s?.needle_span;
  $('attDot').style.opacity = $('attTag').style.opacity = show ? 1 : 0;
  if (show) {
    const r = needleRatio(s), xa = docX(s.depth), ya = aScale.y(r);
    $('attDot').style.transform = `translate(${xa}px,${ya}px)`;
    const tag = $('attTag'), t = tag.querySelector('text'), bg = tag.querySelector('rect');
    t.textContent = `needle ${ratioLabel(r)}`;
    const w = t.getComputedTextLength?.() || 140;
    bg.setAttribute('x', -w / 2 - 12); bg.setAttribute('width', w + 24);
    const tx = Math.min(Math.max(xa, XD0 + w / 2 + 14), XD1 - w / 2 - 14);
    tag.style.transform = `translate(${tx}px,${ya - 38}px)`;
  }
  const hit = $('hit');
  hit.setAttribute('aria-valuenow', Math.round(p.depth * 100));
  hit.setAttribute('aria-valuetext', `Needle at ${pct(p.depth)} of the document`);
}

const KIND = { lookalike: 'gave a look‑alike’s answer', garbled: 'mixed two facts up', not_found: 'said it wasn’t there' };

function renderReadouts() {
  const cur = recallFor(), p = cur.points[state.d];
  $('where').textContent = `Needle at ${pct(p.depth)}`;
  $('hits').textContent = p.correct;
  $('outOf').textContent = `/${p.n}`;
  const note = $('missNote');
  note.innerHTML = '';
  if (!p.misses.length) {
    note.textContent = `Found in all ${p.n} documents.`;
  } else {
    const counts = {};
    p.misses.forEach(m => { counts[m.kind] = (counts[m.kind] ?? 0) + 1; });
    const kinds = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const how = kinds.length === 1
      ? `${p.misses.length === 1 ? '1 miss' : `${p.misses.length} misses`}: ${p.misses.length === 2 ? 'both ' : p.misses.length > 2 ? 'all ' : ''}${KIND[kinds[0][0]]}`
      : `${p.misses.length} misses: ${kinds.map(([k, c]) => `${c} ${KIND[k]}`).join(', ')}`;
    const ex = p.misses.find(m => m.kind === 'lookalike') ?? p.misses[0];
    const said = document.createElement('q'), want = document.createElement('q');
    said.textContent = ex.said.length > 26 ? `${ex.said.slice(0, 24)}…` : ex.said;
    want.textContent = ex.answer;
    note.append(`${how}. e.g. `, said, ' for ', want);
  }
  if (state.fix) {
    const base = recallFor(condKey(state.look, false))?.points[state.d];
    if (base) note.append(` Without the fix: ${base.correct}/${base.n}.`);
  }
  const needle = recall.needles[0];
  $('stripCap').textContent = `${fmtInt(cur.prompt_tokens[1])} tokens`;
  $('pinKey').hidden = !state.look;
  $('recallMeta').innerHTML = `<b>${recall.model_short}</b> · MLX 4-bit · ${cur.points[0].n} documents per point · measured`;

  const run = attnFor();
  $('sinkPct').textContent = run ? Math.round(run.no_needle.sink * 100) : '–';
  $('attnNote').innerHTML = '';
  if (run) {
    const code = document.createElement('code');
    code.textContent = run.token0;
    $('attnNote').append(code, ' is a template token, not your words.');
  }
  const tag = state.L !== String(attn.context_tokens_target) ? `<span class="tag">measured at ${kTok(attn.context_tokens_target)} only</span>` : '';
  $('attnMeta').innerHTML = `<b>${attn.model_short}</b> · MLX 4-bit · ${kTok(attn.context_tokens_target)} tokens · mean of ${attn.layers} layers × ${attn.heads} heads · measured${tag}`;

  // pins: where the look-alikes sat in document 1 (positions differ per document)
  const pins = $('gPins'); clear(pins);
  if (state.look && cur.lookalike_pos?.[0]) {
    for (const f of cur.lookalike_pos[0]) {
      const x = docX(f), y = S.top + S.h / 2;
      el('path', { d: `M${x},${y - 9}L${x + 7},${y}L${x},${y + 9}L${x - 7},${y}Z`, class: 'pin' }, pins);
    }
  }
}

function renderControls() {
  const seg = $('lenSeg');
  seg.innerHTML = '';
  lengths().forEach((L, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-pressed', String(L === state.L));
    b.innerHTML = `<kbd>${i + 1}</kbd>${kTok(L)} tokens`;
    b.addEventListener('click', () => set({ L }));
    seg.append(b);
  });
  const lb = $('lookBtn');
  lb.setAttribute('aria-pressed', String(state.look));
  lb.disabled = !lookAvailable();
  lb.title = lb.disabled ? 'Not measured for this setting' : '';
  const fb = $('fixBtn');
  fb.disabled = !fixAvailable();
  fb.title = fb.disabled ? 'Not measured for this setting' : '';
  fb.setAttribute('aria-pressed', String(state.fix && !fb.disabled));
}

function render() {
  if (!recall) return;
  if (state.fix && !fixAvailable()) state.fix = false;
  if (!recallFor()) Object.assign(state, { look: DEFAULTS.look, fix: false });
  renderControls();
  const { yOf } = drawRecall();
  drawAttn();
  renderReadouts();
  renderNeedle(yOf);
}

function set(patch) {
  Object.assign(state, patch);
  render();
}
function setDepth(d) {
  const n = N_DEPTHS();
  const nd = Math.max(0, Math.min(n - 1, d));
  if (nd !== state.d) set({ d: nd });
}

// ── input ──
function stageX(clientX) {
  const r = stage.getBoundingClientRect();
  return (clientX - r.left) * 1920 / r.width;
}
const hit = $('hit');
hit.addEventListener('pointerdown', e => {
  hit.setPointerCapture(e.pointerId);
  const go = ev => setDepth(Math.round((stageX(ev.clientX) - XD0) / (XD1 - XD0) * (N_DEPTHS() - 1)));
  go(e);
  const move = ev => go(ev);
  const up = () => { hit.removeEventListener('pointermove', move); hit.removeEventListener('pointerup', up); hit.removeEventListener('pointercancel', up); };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerup', up);
  hit.addEventListener('pointercancel', up);
});
const toggleLook = () => { if (lookAvailable()) set({ look: !state.look, fix: state.fix && !!recallFor(condKey(!state.look, true)) }); };
$('lookBtn').addEventListener('click', toggleLook);
$('fixBtn').addEventListener('click', () => fixAvailable() && set({ fix: !state.fix }));

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}
addEventListener('keydown', e => {
  if (!recall || e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;
  const k = e.key;
  if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'ArrowUp') setDepth(state.d + 1);
  else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'ArrowDown') setDepth(state.d - 1);
  else if (k === 'Home') setDepth(0);
  else if (k === 'End') setDepth(N_DEPTHS() - 1);
  else if (/^[1-9]$/.test(k) && lengths()[+k - 1]) set({ L: lengths()[+k - 1] });
  else if (k === 'l' || k === 'L') toggleLook();
  else if (k === 'q' || k === 'Q') { if (fixAvailable()) set({ fix: !state.fix }); }
  else if (k === 'f' || k === 'F') toggleFullscreen();
  else if (k === 'Escape') set({ ...DEFAULTS });
  else return;
  e.preventDefault();
});

// ── boot ──
function fit() {
  const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  stage.style.transform = `translate(${(innerWidth - 1920 * k) / 2}px,${(innerHeight - 1080 * k) / 2}px) scale(${k})`;
}
addEventListener('resize', fit);

async function getJson(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function boot() {
  fit();
  stage.classList.toggle('still', motion.matches || q.has('still'));
  try {
    [recall, attn] = await Promise.all([getJson('data/recall.json'), getJson('data/attention.json')]);
  } catch (err) {
    console.warn('[u-shape] data missing', err);
    const b = $('banner');
    b.textContent = location.protocol === 'file:'
      ? 'Open this through ./serve.sh: http://localhost:8000/u-shape/ (browsers block data files on file://).'
      : `The measured data didn't load (${err.message}). Run the experiments in u-shape/experiments/.`;
    b.hidden = false;
    return;
  }
  if (!lengths().includes(state.L)) DEFAULTS.L = state.L = lengths().at(-1);
  const date = new Date(recall.env.date);
  const when = date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  $('footMeasured').textContent = `All numbers measured on this Mac (${recall.env.machine.split(',')[0]}), ${when} · documents: ${recall.needles.length} Project Gutenberg books · greedy decoding, exact-match scoring`;
  drawStatic();
  render();
  requestAnimationFrame(() => { stage.dataset.ready = 'true'; });
}
boot();

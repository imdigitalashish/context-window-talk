import { NOLIMA, NOLIMA_LENGTHS, SEGMENTS, MCP_GITHUB_TOKENS, ZONES, WINDOWS, ILLUSTRATION } from './data.js';

const MAX_TURN = 40;
const $ = id => document.getElementById(id);
const stage = $('stage');
const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
const still = () => motionQuery.matches || new URLSearchParams(location.search).has('still');

const INITIAL = { turn: 0, win: 200000, compactAt: null, model: 'gpt-4o' };
const state = { ...INITIAL };
let overhead = null;      // fixed tokens before you type: { system, tools, mcp, memory }
let growth = [];          // growth[t] = { total, read, tool, chat } tokens added since the first call, t = 0..40
let summaryTokens = 0;
let noteTimer = 0;

const fmt = n => (n >= 1e6 ? `${+(n / 1e6).toFixed(2)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${Math.round(n)}`);
const fmtLen = n => (n >= 1e6 ? `${n / 1e6}M` : `${n / 1000}K`);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

/* ───────── measured data → token model ───────── */

function buildModel(m) {
  overhead = {
    system: m.system_prompt_tokens.median + m.developer_messages_tokens.median,
    tools: m.tool_definitions_tokens_derived.median,
    mcp: MCP_GITHUB_TOKENS,
    memory: m.agents_md_tokens.median,
  };
  summaryTokens = m.compactions.after_minus_same_session_first_call_median;
  // Median growth per step, held non-decreasing (running max) so the bar never shrinks between steps.
  growth = [{ total: 0, read: 0, tool: 0, chat: 0 }];
  for (let t = 1; t <= MAX_TURN; t++) {
    const s = m.growth_per_step[t - 1];
    const parts = s.read + s.tool + s.chat || 1;
    const total = Math.max(growth[t - 1].total, s.median_growth_since_first_call);
    growth.push({ total, read: total * s.read / parts, tool: total * s.tool / parts, chat: total * s.chat / parts });
  }
  $('nSessions').textContent = String(m.sessions);
}

function tokens() {
  const g = growth[state.turn], c = state.compactAt;
  const seg = { ...overhead, summary: 0, read: g.read, tool: g.tool, chat: g.chat };
  if (c != null) {
    const after = Math.max(0, g.total - growth[c].total), k = g.total ? after / g.total : 0;
    Object.assign(seg, { summary: summaryTokens, read: g.read * k, tool: g.tool * k, chat: g.chat * k });
  }
  return seg;
}
const sum = seg => Object.values(seg).reduce((a, b) => a + b, 0);

/* ───────── meter ───────── */

const barEl = $('bar'), legendEl = $('legend');
const segEls = {}, legendEls = {};

function buildMeter() {
  for (const s of SEGMENTS) {
    const d = document.createElement('div');
    d.className = `seg ${s.id}`;
    d.style.setProperty('--c', s.color);
    d.style.width = '0%';
    barEl.append(d);
    segEls[s.id] = d;

    const li = document.createElement('li');
    li.className = s.id;
    li.style.setProperty('--c', s.color);
    li.innerHTML = '<i></i><span></span><b></b>';
    li.querySelector('span').textContent = s.label;
    li.style.order = String(s.legend);
    legendEl.append(li);
    legendEls[s.id] = li;
  }
  const wins = $('windows');
  for (const w of WINDOWS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = w.label;
    b.dataset.tokens = String(w.tokens);
    b.addEventListener('click', () => { b.blur(); setWindow(w.tokens); });
    wins.append(b);
  }
}

function zoneOf(used, win) {
  const p = used / win;
  return used > win ? 'over' : p >= ZONES.deep ? 'deep' : p >= ZONES.dumb ? 'dumb' : 'smart';
}
const ZONE_NAMES = { smart: 'Smart zone', dumb: 'Dumb zone', deep: 'Deep dumb zone', over: 'Over the window' };

function renderMeter(seg, used) {
  const win = state.win, scale = used > win ? 1 / used : 1 / win;
  for (const s of SEGMENTS) {
    segEls[s.id].style.width = `${seg[s.id] * scale * 100}%`;
    const li = legendEls[s.id];
    li.querySelector('b').textContent = seg[s.id] >= 1 ? fmt(seg[s.id]) : '0';
    li.classList.toggle('zero', seg[s.id] < 1);
  }
  const zone = zoneOf(used, win);
  stage.dataset.zone = zone;
  $('zoneName').textContent = ZONE_NAMES[zone];
  $('used').textContent = fmt(used);
  $('ofWin').textContent = `of ${fmtLen(win)}`;
  $('pct').textContent = `${Math.round(used / win * 100)}%`;
  for (const b of $('windows').children) b.setAttribute('aria-pressed', String(+b.dataset.tokens === win));
}

/* ───────── chart ───────── */

const SVGNS = 'http://www.w3.org/2000/svg';
const P = { w: 780, h: 318, left: 70, right: 590, top: 52, bottom: 270 };
const xOf = t => P.left + (Math.log2(clamp(t, 1000, 128000) / 1000) / 7) * (P.right - P.left);
const yOf = v => P.top + (1 - v / 100) * (P.bottom - P.top);
const svg = (tag, attrs = {}, parent) => {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  parent?.append(el);
  return el;
};
let markerEl, markerLine, markerText, markerRect, markerDot;
const seriesEls = {};

function buildChart() {
  const root = $('plot');
  root.setAttribute('viewBox', `0 0 ${P.w} ${P.h}`);
  root.setAttribute('width', P.w);
  root.setAttribute('height', P.h);
  const grid = svg('g', { class: 'grid' }, root), axis = svg('g', { class: 'axis' }, root);
  for (const v of [0, 25, 50, 75, 100]) {
    svg('line', { x1: P.left, x2: P.right, y1: yOf(v), y2: yOf(v) }, v === 0 ? axis : grid);
    svg('text', { x: P.left - 12, y: yOf(v) + 8, 'text-anchor': 'end' }, axis).textContent = `${v}%`;
  }
  NOLIMA_LENGTHS.forEach(L => {
    svg('line', { x1: xOf(L), x2: xOf(L), y1: P.top, y2: P.bottom }, grid);
    svg('text', { x: xOf(L), y: P.bottom + 34, 'text-anchor': 'middle' }, axis).textContent = fmtLen(L);
  });

  markerLine = svg('g', { class: 'marker' }, root); // under the series, so label halos interrupt the dashed line
  svg('line', { x1: 0, x2: 0, y1: P.top - 8, y2: P.bottom }, markerLine);

  const labels = [];
  for (const m of NOLIMA) {
    const g = svg('g', { class: 'series' }, root);
    const pts = m.scores.map((v, i) => (v == null ? null : [xOf(NOLIMA_LENGTHS[i]), yOf(v)])).filter(Boolean);
    svg('path', { d: `M${pts.map(p => p.join(',')).join('L')}` }, g);
    pts.forEach(([x, y]) => svg('circle', { cx: x, cy: y, r: 4 }, g));
    svg('path', { class: 'hit', d: `M${pts.map(p => p.join(',')).join('L')}`, style: 'stroke:transparent;stroke-width:22;fill:none' }, g);
    const at = m.label?.at ?? pts.length - 1, [lx, ly] = pts[at];
    const text = svg('text', { x: lx + (m.label?.dx ?? 12), 'paint-order': 'stroke', stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round' }, g);
    text.textContent = m.name;
    labels.push({ text, x: lx, y: ly + (m.label?.dy ?? 0), fixed: !!m.label });
    g.addEventListener('click', () => setModel(m.id));
    seriesEls[m.id] = g;
  }
  // nudge end labels that share an x so they never overlap (24px text needs ~28px)
  labels.filter(l => l.fixed).forEach(l => l.text.setAttribute('y', l.y + 8));
  for (const group of Object.values(Object.groupBy(labels.filter(l => !l.fixed), l => Math.round(l.x)))) {
    group.sort((a, b) => a.y - b.y);
    for (let i = 1; i < group.length; i++) group[i].y = Math.max(group[i].y, group[i - 1].y + 28);
    group.forEach(l => l.text.setAttribute('y', l.y + 8));
  }

  markerEl = svg('g', { class: 'marker' }, root);
  markerDot = svg('circle', { cx: 0, cy: P.bottom, r: 8 }, markerEl);
  markerRect = svg('rect', { x: -80, y: P.top - 50, width: 160, height: 38, rx: 8 }, markerEl);
  markerText = svg('text', { x: 0, y: P.top - 23, 'text-anchor': 'middle' }, markerEl);
}

/** Linear in log-length between measured points: only used to place the dot on the drawn line. */
function lineYAt(m, t) {
  const pts = m.scores.map((v, i) => [NOLIMA_LENGTHS[i], v]).filter(p => p[1] != null);
  const c = clamp(t, pts[0][0], pts[pts.length - 1][0]);
  for (let i = 1; i < pts.length; i++) {
    if (c <= pts[i][0]) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const f = Math.log2(c / x0) / Math.log2(x1 / x0);
      return y0 + (y1 - y0) * f;
    }
  }
  return pts[pts.length - 1][1];
}

function renderChart(used) {
  const m = NOLIMA.find(x => x.id === state.model);
  for (const [id, g] of Object.entries(seriesEls)) {
    g.classList.toggle('on', id === m.id);
    if (id === m.id) g.parentNode.append(g); // draw the highlighted line on top
  }
  $('plot').append(markerEl);

  const x = xOf(used), beyond = used > 128000;
  markerEl.style.transform = markerLine.style.transform = `translateX(${x}px)`;
  const label = beyond ? `you: ${fmt(used)} →` : `you: ${fmt(used)}`;
  markerText.textContent = label;
  const w = 22 + label.length * 13.2;
  const shift = clamp(0, P.left - x + w / 2 - 30, P.w - x - w / 2) || 0;
  markerRect.setAttribute('x', -w / 2 + shift);
  markerRect.setAttribute('width', w);
  markerText.setAttribute('x', shift);
  // Only mark the line where this model was actually tested (a context past 128K pins to the 128K point).
  const lastLen = NOLIMA_LENGTHS[m.scores.findLastIndex(v => v != null)];
  markerDot.style.display = Math.min(used, 128000) <= lastLen ? '' : 'none';
  markerDot.setAttribute('cy', yOf(lineYAt(m, used)));

  // Readout: the measured score at the longest tested length at or below the current context.
  let i = -1;
  NOLIMA_LENGTHS.forEach((L, j) => { if (L <= used && m.scores[j] != null) i = j; });
  if (i < 0) {
    $('roLabel').textContent = `Under 1K tokens, ${m.name}`;
    $('roValue').textContent = `${m.base}%`;
    $('roBase').textContent = 'its short-length score';
  } else {
    $('roLabel').textContent = `At ${fmtLen(NOLIMA_LENGTHS[i])} tokens, ${m.name}:`;
    $('roValue').textContent = `${m.scores[i].toFixed(1)}%`;
    $('roBase').textContent = `from ${m.base}% at short length`;
  }
  $('roNote').textContent = i < 0 ? ''
    : used > lastLen ? `Your ${fmt(used)} is past the longest length tested for ${m.name} (${fmtLen(lastLen)})`
    : `Longest tested length at or below your ${fmt(used)}`;
  $('claimed').textContent = m.claimed;
  $('effective').textContent = m.effective;
}

/* ───────── self-conditioning illustration ───────── */

function buildStrips() {
  const { turns, mistakeAt, freshFrom, sameChatWrong, freshChatWrong } = ILLUSTRATION;
  const fill = (el, wrong, cut) => {
    el.style.gridTemplateColumns = `repeat(${turns},1fr)`;
    for (let t = 1; t <= turns; t++) {
      const i = document.createElement('i');
      if (wrong.includes(t)) i.className = 'bad';
      if (t === mistakeAt) i.classList.add('first');
      if (cut && t === freshFrom) i.classList.add('cut');
      i.style.transitionDelay = still() ? '0ms' : `${200 + t * 22}ms`;
      i.title = `Turn ${t}: ${wrong.includes(t) ? 'wrong' : 'correct'} (illustration)`;
      el.append(i);
    }
  };
  fill($('stripSame'), sameChatWrong, false);
  fill($('stripFresh'), freshChatWrong, true);
  requestAnimationFrame(() => requestAnimationFrame(() => document.querySelectorAll('.cells').forEach(c => c.classList.add('on'))));
}

/* ───────── state changes ───────── */

function render() {
  const seg = tokens(), used = sum(seg);
  renderMeter(seg, used);
  renderChart(used);
  $('turn').innerHTML = `<b>Turn ${state.turn}</b> <span>of ${MAX_TURN}</span>`;
  $('meterSub').textContent = state.turn === 0 ? 'before you type anything' : `after ${state.turn} agent turn${state.turn === 1 ? '' : 's'}`;
  $('prev').disabled = state.turn === 0;
  $('next').disabled = state.turn === MAX_TURN;
  if (state.compactAt != null) {
    showNote(`<b>Compacted at turn ${state.compactAt}.</b> Details not in the summary are gone.`, 0);
  }
}

function showNote(html, ms = 4200) {
  const el = $('note');
  el.innerHTML = html;
  el.classList.add('on');
  clearTimeout(noteTimer);
  if (ms) noteTimer = setTimeout(() => el.classList.remove('on'), ms);
}
const hideNote = () => { clearTimeout(noteTimer); $('note').classList.remove('on'); };

function step(d) {
  const t = clamp(state.turn + d, 0, MAX_TURN);
  if (t === state.turn) return;
  state.turn = t;
  if (state.compactAt != null && t < state.compactAt) { state.compactAt = null; hideNote(); }
  render();
}

function flash(id) {
  const b = $(id);
  b.classList.add('flash');
  setTimeout(() => b.classList.remove('flash'), 450);
}

function compact() {
  flash('compact');
  const seg = tokens(), loose = seg.read + seg.tool + seg.chat;
  const before = loose + seg.summary;
  if (loose < 1 || before <= summaryTokens) {
    showNote(`<b>Nothing worth compacting yet:</b> the chat is smaller than a ${fmt(summaryTokens)} summary.`);
    return;
  }
  state.compactAt = state.turn;
  render();
  showNote(`<b>Compacted ${fmt(before)} into a ${fmt(summaryTokens)} summary.</b> Details not in the summary are gone.`, 0);
}

function freshChat() {
  flash('fresh');
  state.turn = 0;
  state.compactAt = null;
  render();
  showNote(`<b>Fresh chat.</b> Only the ${fmt(sum(tokens()))} you pay before typing is left.`);
}

function reset() {
  Object.assign(state, INITIAL);
  hideNote();
  render();
}

function setWindow(win) { state.win = win; render(); }
function setModel(id) { state.model = id; render(); }
const cycle = (list, cur) => list[(list.indexOf(cur) + 1) % list.length];

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

addEventListener('keydown', e => {
  if (e.isComposing || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === 'ArrowRight' || k === 'PageDown' || k === ' ') { e.preventDefault(); step(1); }
  else if (k === 'ArrowLeft' || k === 'PageUp') { e.preventDefault(); step(-1); }
  else if (k === 'c' || k === 'C') { e.preventDefault(); compact(); }
  else if (k === 'r' || k === 'R') { e.preventDefault(); freshChat(); }
  else if (k === 'f' || k === 'F') { e.preventDefault(); toggleFullscreen(); }
  else if (k === 'Escape') { e.preventDefault(); reset(); }
  else if (k === 'w' || k === 'W') { e.preventDefault(); setWindow(cycle(WINDOWS.map(w => w.tokens), state.win)); }
  else if (k === 'm' || k === 'M') { e.preventDefault(); setModel(cycle(NOLIMA.map(m => m.id), state.model)); }
  else if (k === 'Home') { e.preventDefault(); step(-MAX_TURN); }
  else if (k === 'End') { e.preventDefault(); step(MAX_TURN); }
});
$('prev').addEventListener('click', e => { e.currentTarget.blur(); step(-1); });
$('next').addEventListener('click', e => { e.currentTarget.blur(); step(1); });
$('compact').addEventListener('click', e => { e.currentTarget.blur(); compact(); });
$('fresh').addEventListener('click', e => { e.currentTarget.blur(); freshChat(); });

function fit() {
  const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  stage.style.transform = `translate(${(innerWidth - 1920 * k) / 2}px,${(innerHeight - 1080 * k) / 2}px) scale(${k})`;
}
addEventListener('resize', fit);

/* ───────── boot ───────── */

async function boot() {
  fit();
  buildMeter();
  buildChart();
  buildStrips();
  try {
    const res = await fetch('data/codex-sessions.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    buildModel(await res.json());
  } catch (err) {
    console.error('[dumb-zone] measured data missing', err);
    const b = $('banner');
    b.textContent = `Couldn't load data/codex-sessions.json (${err.message}). Run ./serve.sh and open http://localhost:8000/dumb-zone/`;
    b.hidden = false;
    return;
  }
  const p = new URLSearchParams(location.search);
  if (p.has('turn')) state.turn = clamp(parseInt(p.get('turn'), 10) || 0, 0, MAX_TURN);
  if (p.has('win')) state.win = WINDOWS.find(w => w.label === p.get('win'))?.tokens ?? state.win;
  render();
  if (p.has('compact')) compact();
}

boot();

// Builds demo/data/map.json + demo/data/vectors.i8 from scripts/data/corpus.json.
//
//  1. Take the top-voted PER_PART titles of each part in PARTS.
//  2. Embed every title with demo/engine.js, i.e. the same transformers.js build, wasm runtime and
//     model files the browser uses (mean pooling, L2-normalised).
//  3. Lay the points out in 3-D with UMAP (n_components=3) on exact cosine kNN; each axis is
//     normalised to [0,1] with padding.
//  4. Per part: L2-normalised centroid, label anchor (densest spot of its island), colour, 3 examples.
//  5. Quantise vectors to int8 (scale 127) in the same order as `points`.
//
// Thresholds are kept from an existing map.json; run calibrate.mjs afterwards to (re)set them.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { UMAP } from 'umap-js';
import * as engine from '../demo/engine.js';
import { CANDIDATE_PARTS } from './parts.mjs';

// Final selection. 'workplace' is fetched but left out: it competed with Relationships for 3 AM-style
// prompts and overlaps law/money. 'writing' gives "write the landing page copy" a sensible home.
const PARTS = process.env.PARTS?.split(',') ?? ['react', 'css', 'auth', 'postgres', 'docker', 'payments', 'ml',
  'android', 'testing', 'ux', 'security', 'writing', 'law', 'money', 'interpersonal', 'cooking'];
const PER_PART = 125;
const SEED = 20261010;
const DIMS = 384;
const UMAP_PARAMS = { nComponents: 3, nNeighbors: 20, minDist: 0.12, spread: 1.0, nEpochs: 600 };

const DATA = new URL('../demo/data/', import.meta.url);
const MAP_FILE = new URL('map.json', DATA);
const VEC_FILE = new URL('vectors.i8', DATA);

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const normalise = (v) => { const n = Math.sqrt(dot(v, v)) || 1; return v.map((x) => x / n); };
const r4 = (v) => Math.round(v * 1e4) / 1e4;

// ---------------------------------------------------------------- colours
// Evenly spaced OKLCH hues, alternating lightness for adjacent slots; mid lightness and high chroma
// so dots read on both light and dark backgrounds through a projector.
function oklchToRgb(L, C, hDeg) {
  const h = (hDeg * Math.PI) / 180;
  const a = C * Math.cos(h), b = C * Math.sin(h);
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return lin.map((x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055));
}
function slotColor(i, n) {
  const L = i % 2 ? 0.74 : 0.62;
  const hue = 20 + (360 * i) / n;
  for (let C = 0.19; C > 0; C -= 0.005) {
    const rgb = oklchToRgb(L, C, hue);
    if (rgb.every((x) => x >= 0 && x <= 1)) {
      return { hex: '#' + rgb.map((x) => Math.round(x * 255).toString(16).padStart(2, '0')).join('').toUpperCase(), lab: [L, C * Math.cos((hue * Math.PI) / 180), C * Math.sin((hue * Math.PI) / 180)] };
    }
  }
  throw new Error('no in-gamut colour');
}
// Greedy: parts whose islands sit close together on the map get colours far apart in OKLab.
function assignColors(anchors) {
  const n = anchors.length;
  const slots = Array.from({ length: n }, (_, i) => slotColor(i, n));
  const near = anchors.map((a) => anchors.map((b) => (a === b ? 0 : Math.exp(-(((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) / 0.25 ** 2)))));
  const order = [...anchors.keys()].sort((p, q) => near[q].reduce((s, v) => s + v, 0) - near[p].reduce((s, v) => s + v, 0));
  const taken = new Map(); // part -> slot
  for (const p of order) {
    let best = -1, bestCost = Infinity;
    slots.forEach((slot, si) => {
      if ([...taken.values()].includes(si)) return;
      let cost = 0;
      for (const [q, sq] of taken) {
        const d = Math.hypot(...slot.lab.map((v, k) => v - slots[sq].lab[k]));
        cost += near[p][q] / (d + 0.02);
      }
      if (cost < bestCost) { bestCost = cost; best = si; }
    });
    taken.set(p, best);
  }
  return anchors.map((_, p) => slots[taken.get(p)].hex);
}

// ---------------------------------------------------------------- main
async function main() {
  const corpus = JSON.parse(await readFile(new URL('data/corpus.json', import.meta.url), 'utf8'));
  const byId = new Map(corpus.parts.map((p) => [p.id, p]));
  const meta = new Map(CANDIDATE_PARTS.map((p) => [p.id, p]));

  const items = [];
  for (const id of PARTS) {
    const part = byId.get(id);
    if (!part) throw new Error(`part ${id} missing from corpus.json; run fetch-corpus.mjs`);
    if (part.questions.length < 120) console.warn(`warning: ${id} has only ${part.questions.length} titles`);
    for (const q of part.questions.slice(0, PER_PART)) items.push({ part: id, t: q.title, score: q.score });
  }
  const rand = mulberry32(SEED);
  for (let i = items.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [items[i], items[j]] = [items[j], items[i]]; }
  console.log(`${items.length} titles in ${PARTS.length} parts`);

  // 2. Embed with the engine (same build + model as the browser).
  await engine.init({ map: false });
  const t0 = Date.now();
  for (let i = 0; i < items.length; i++) {
    items[i].v = await engine.embed(items[i].t);
    if ((i + 1) % 500 === 0) console.log(`  embedded ${i + 1}/${items.length}`);
  }
  console.log(`embedded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  // Quantise first, so the layout and centroids see exactly what the engine will see at runtime.
  const n = items.length;
  const i8 = new Int8Array(n * DIMS);
  items.forEach((it, i) => {
    for (let d = 0; d < DIMS; d++) i8[i * DIMS + d] = Math.max(-127, Math.min(127, Math.round(it.v[d] * 127)));
    it.q = normalise(Array.from(i8.subarray(i * DIMS, (i + 1) * DIMS), (x) => x / 127));
  });
  let worst = 1;
  for (const it of items) worst = Math.min(worst, dot(it.q, it.v));
  console.log(`int8 round-trip: worst cosine to float vector ${worst.toFixed(5)}`);

  // 3. Exact cosine kNN -> UMAP.
  const k = UMAP_PARAMS.nNeighbors;
  const knnIndices = [], knnDistances = [];
  const F = items.map((it) => Float32Array.from(it.q));
  for (let i = 0; i < n; i++) {
    const sims = new Float32Array(n);
    for (let j = 0; j < n; j++) sims[j] = i === j ? 2 : dot(F[i], F[j]);
    const idx = Array.from(sims.keys()).sort((a, b) => sims[b] - sims[a]).slice(0, k);
    knnIndices.push(idx);
    knnDistances.push(idx.map((j) => (j === i ? 0 : Math.max(0, 1 - sims[j]))));
  }
  const umap = new UMAP({ ...UMAP_PARAMS, random: mulberry32(SEED + 1), distanceFn: (a, b) => 1 - dot(a, b) });
  umap.setPrecomputedKNN(knnIndices, knnDistances);
  const t1 = Date.now();
  const emb = umap.fit(items.map((it) => it.q));
  console.log(`UMAP in ${((Date.now() - t1) / 1000).toFixed(1)}s`);

  // Normalise each axis to [0,1] with padding, robust to a few far outliers (clamped into the padding).
  const PAD = 0.05;
  const q = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(p * (s.length - 1))]; };
  const clamp = (v) => Math.min(1 - PAD / 2, Math.max(PAD / 2, v));
  ['x', 'y', 'z'].forEach((axis, a) => {
    const vals = emb.map((e) => e[a]);
    const lo = q(vals, 0.002), hi = q(vals, 0.998);
    items.forEach((it, i) => { it[axis] = r4(clamp(PAD + ((emb[i][a] - lo) / (hi - lo)) * (1 - 2 * PAD))); });
  });

  // 4. Parts.
  const parts = PARTS.map((id) => {
    const members = items.filter((it) => it.part === id);
    const centroid = normalise(Array.from({ length: DIMS }, (_, d) => members.reduce((s, it) => s + it.q[d], 0)));
    // Label anchor: the member with the most same-part neighbours within R (in 3-D), averaged with them.
    const R = 0.08;
    const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2;
    let bestPeak = null, bestCount = -1;
    for (const a of members) {
      const c = members.filter((b) => d2(a, b) < R * R).length;
      if (c > bestCount) { bestCount = c; bestPeak = a; }
    }
    const around = members.filter((b) => d2(bestPeak, b) < R * R);
    const avg = (k) => around.reduce((s, b) => s + b[k], 0) / around.length;
    const [x, y, z] = [avg('x'), avg('y'), avg('z')];
    // Examples: recognisable (high-voted) among the most central, display-friendly titles.
    const central = members.map((it) => ({ it, sim: dot(it.q, centroid) })).sort((a, b) => b.sim - a.sim)
      .filter(({ it }) => it.t.length >= 15 && it.t.length <= 75).slice(0, 30)
      .sort((a, b) => b.it.score - a.it.score).slice(0, 3).map(({ it }) => it.t);
    const m = meta.get(id);
    return {
      id, label: m.label, color: null, x: r4(x), y: r4(y), z: r4(z), count: members.length,
      centroid: centroid.map((v) => Math.round(v * 1e6) / 1e6), examples: central,
      source: m.tagged ? `stackoverflow.com, tag [${m.tagged}]` : `${m.site}.stackexchange.com`,
    };
  });
  const colors = assignColors(parts);
  parts.forEach((p, i) => { p.color = colors[i]; });

  // Island quality: share of points whose nearest centroid is their own part.
  let own = 0;
  for (const it of items) {
    let best = null, bs = -2;
    for (const p of parts) { const s = dot(it.q, p.centroid); if (s > bs) { bs = s; best = p.id; } }
    if (best === it.part) own++;
  }
  console.log(`points nearest to their own centroid: ${(100 * own / n).toFixed(1)}%`);

  const previous = existsSync(MAP_FILE) ? JSON.parse(await readFile(MAP_FILE, 'utf8')) : null;
  const map = {
    model: 'Xenova/all-MiniLM-L6-v2',
    dims: DIMS,
    source: 'Stack Exchange question titles, CC BY-SA 4.0',
    thresholds: previous?.thresholds ?? { touch: 0.3, strong: 0.45 },
    parts,
    points: items.map((it) => ({ x: it.x, y: it.y, z: it.z, part: it.part, t: it.t })),
    vectors: { file: 'vectors.i8', dtype: 'int8', scale: 127, count: n },
  };
  await mkdir(DATA, { recursive: true });
  await writeFile(MAP_FILE, JSON.stringify(map));
  await writeFile(VEC_FILE, i8);
  const kb = (b) => (b / 1024).toFixed(0) + ' KB';
  console.log(`wrote map.json (${kb(JSON.stringify(map).length)}) + vectors.i8 (${kb(i8.byteLength)})`);
  for (const p of parts) console.log(`  ${p.id.padEnd(14)} ${p.color}  (${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})  ${p.examples[0]}`);
}

main().catch((e) => { console.error(e); process.exit(1); });

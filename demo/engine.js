// Prompt-footprint engine for the clustering-law demo. Contract: docs/demo-spec.md.
//
// Runs in the browser (served over http) and in Node 18+. Everything is local:
//   demo/vendor/   transformers.js browser build + onnxruntime-web wasm runtime
//   demo/models/   Xenova/all-MiniLM-L6-v2 (quantized ONNX, tokenizer, config)
//   demo/data/     map.json (3-D layout) + vectors.i8 (corpus embeddings made with this same file and model)
// Node runs the very same transformers.min.js (on its web code path) with the same onnxruntime-web
// wasm binary and model, so clause vectors and corpus vectors live in one vector space.

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DIMS = 384;
const K = 12; // neighbours for the kNN part vote and the landing position
const XYZ_TEMPERATURE = 0.05; // softmax(sim / T) weights for the landing position

const IS_NODE = typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined';
const DEFAULT_BASE = new URL('./', import.meta.url); // the demo/ directory

let base = DEFAULT_BASE;
let mapPromise = null;
let modelPromise = null;
let initPromise = null;

let map = null;
let vectors = null; // Float32Array(count * DIMS), dequantised and re-normalised
let pointPart = null; // Int32Array: point index -> part index
let partIds = [];
let centroids = null; // Float32Array(parts * DIMS)
let extractor = null;

const listeners = new Set();

/** Register cb({ stage: 'model' | 'map', loaded, total }). Returns an unsubscribe function. */
export function onProgress(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function emit(stage, loaded, total) {
  for (const cb of listeners) {
    try { cb({ stage, loaded, total }); } catch (e) { console.error(e); }
  }
}

/**
 * Loads the map (map.json + vectors) and the model. Idempotent: later calls return the same promise.
 * If loading fails the promise rejects and the next call retries. The map loads first and getMap()
 * returns it as soon as it is ready, even if the model then fails (so the UI can draw the dots and
 * fall back to data/presets.results.json).
 *
 * opts.base: URL of the demo/ directory (defaults to the directory this file is served from).
 * opts.onProgress: shorthand for onProgress(cb).
 * opts.map === false: load only the model (used by the build script before map.json exists).
 */
export function init(opts = {}) {
  if (opts.onProgress) onProgress(opts.onProgress);
  if (!initPromise) {
    if (opts.base) {
      base = new URL(opts.base, IS_NODE ? undefined : globalThis.location?.href);
      if (!base.pathname.endsWith('/')) base.pathname += '/';
    }
    const steps = opts.map === false ? [loadModel()] : [loadMap(), loadModel()];
    initPromise = Promise.all(steps).then(() => undefined, (err) => { initPromise = null; throw err; });
  }
  return initPromise;
}

/** The parsed map.json, or null until it has loaded. */
export function getMap() {
  return map;
}

/** Splits a prompt into clauses. See docs/demo-spec.md for the rules. */
export function splitClauses(prompt) {
  if (typeof prompt !== 'string') return [];
  const text = prompt.replace(/\r\n?/g, '\n').replace(/[\u2018\u2019\u02BC]/g, "'");
  const clauses = [];
  for (const rawLine of text.split('\n')) {
    const line = rawLine.replace(BULLET, '').trim();
    if (!line) continue;
    for (const sentence of splitSentences(line)) {
      for (const piece of sentence.split(';')) {
        for (const clause of splitImperatives(piece)) {
          const c = clause.trim().replace(LEADING_CONNECTOR, '').replace(TRAILING_PUNCT, '').trim();
          if (c.length >= 3) clauses.push(c);
        }
      }
    }
  }
  return clauses;
}

/** Embeds text: Float32Array(384), mean-pooled and L2-normalised. */
export async function embed(text) {
  await loadModel();
  const out = await extractor(String(text), { pooling: 'mean', normalize: true });
  const v = new Float32Array(out.data);
  out.dispose?.();
  return v;
}

/** Scores a prompt: where each clause lands, which parts it touches, and where the whole prompt points. */
export async function score(prompt) {
  await init();
  await Promise.all([loadMap(), loadModel()]); // in case init() was called with { map: false }
  const touch = map.thresholds.touch;
  const clauses = [];
  for (const text of splitClauses(prompt)) {
    const q = await embed(text);
    const { part, xyz } = locate(q);
    const sim = round(dotCentroid(q, partIds.indexOf(part)));
    clauses.push({ text, part, sim, xyz, xy: xyz.slice(0, 2), touched: sim >= touch });
  }
  const partsTouched = [...new Set(clauses.filter((c) => c.touched).map((c) => c.part))];

  const whole = { sims: {}, top: { part: null, sim: 0 }, xyz: null, xy: null };
  const trimmed = typeof prompt === 'string' ? prompt.trim() : '';
  if (trimmed) {
    const q = await embed(trimmed);
    let best = -1;
    partIds.forEach((id, p) => {
      const s = dotCentroid(q, p);
      whole.sims[id] = round(s);
      if (s > best) { best = s; whole.top = { part: id, sim: round(s) }; }
    });
    whole.xyz = locate(q).xyz;
    whole.xy = whole.xyz.slice(0, 2);
  } else {
    for (const id of partIds) whole.sims[id] = 0;
  }
  return { clauses, partsTouched, whole };
}

// ---------------------------------------------------------------------------
// Clause splitting

const BULLET = /^\s*(?:[-*\u2022\u00B7\u2013\u2014>]+|\d{1,2}[.)]|[a-zA-Z][)])\s+/;
const LEADING_CONNECTOR = /^(?:(?:and|also|plus|then|but)\b[\s,]*)+/i;
const TRAILING_PUNCT = /[\s,;:.!\u2026]+$/;
const ABBREVIATIONS = /\b(e\.g|i\.e|etc|vs|approx|mr|mrs|ms|dr|st|no|fig)\.(?=\s)/gi;
const DOT = '\u0000';

function splitSentences(line) {
  const protectedLine = line.replace(ABBREVIATIONS, (m) => m.replace(/\./g, DOT));
  return protectedLine.split(/(?<=[.!?])\s+(?=\S)/).map((s) => s.replaceAll(DOT, '.'));
}

// Connectors that may separate two imperative clauses: ", " / ", and " / " and " / " also " / " plus " / " then ".
const CONNECTOR = /\s*,\s*(?:(?:and|also|plus|then)\s+)*|\s+(?:(?:and|also|plus|then)\s+)+/gi;

// Verbs that start an imperative clause in a coding-agent prompt. A connector only splits when the
// next clause starts with one of these (or with a negation), so "Postgres and Redis" stays together
// and "Don't use React" keeps its negation.
const VERBS = new Set(`add allow append avoid build bump call change check clean cleanup compute configure connect
convert copy create debug delete deploy design document download drop email enable disable ensure explain export
extract fetch find fix format generate give handle hide host implement import improve include install integrate
keep let list load lock make merge migrate move optimise optimize paginate parse persist port print prevent protect
publish push put read rebuild redesign redirect refactor reject remove rename render replace reply report rerun
reset restrict retry return reuse review rewrite run save scale schedule secure send serve set setup ship show
sign sort split store stream style support switch sync test throttle track train translate trigger try turn
update upgrade upload use validate verify wire wrap write ask tell text cook bake buy pay invest hire quit
email message`.split(/\s+/));
const NEGATION = /^(?:don't|dont|do not|never|no need to|avoid|stop)\b/i;

function startsImperative(rest) {
  const s = rest.trim().replace(/^please\s+/i, '');
  if (NEGATION.test(s)) return true;
  const word = s.match(/^[a-z]+/i)?.[0]?.toLowerCase();
  return !!word && VERBS.has(word);
}

function splitImperatives(piece) {
  const out = [];
  let start = 0;
  for (const m of piece.matchAll(CONNECTOR)) {
    if (m.index === 0 || !m[0]) continue;
    const next = m.index + m[0].length;
    if (startsImperative(piece.slice(next))) {
      out.push(piece.slice(start, m.index));
      start = next;
    }
  }
  out.push(piece.slice(start));
  return out;
}

// ---------------------------------------------------------------------------
// Geometry

function dotCentroid(q, p) {
  let s = 0;
  const o = p * DIMS;
  for (let d = 0; d < DIMS; d++) s += q[d] * centroids[o + d];
  return s;
}

// kNN over the corpus: the part with the most similarity-weighted votes among the K nearest
// points, and the landing spot = softmax(sim / T)-weighted mean of their 3-D positions.
function locate(q) {
  const n = map.points.length;
  const topIdx = new Int32Array(K).fill(-1);
  const topSim = new Float64Array(K).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * DIMS;
    for (let d = 0; d < DIMS; d++) s += q[d] * vectors[o + d];
    if (s <= topSim[K - 1]) continue;
    let j = K - 1;
    while (j > 0 && topSim[j - 1] < s) { topSim[j] = topSim[j - 1]; topIdx[j] = topIdx[j - 1]; j--; }
    topSim[j] = s; topIdx[j] = i;
  }

  const votes = new Float64Array(partIds.length);
  for (let j = 0; j < K; j++) votes[pointPart[topIdx[j]]] += Math.max(topSim[j], 0);
  let part = pointPart[topIdx[0]];
  for (let p = 0; p < votes.length; p++) if (votes[p] > votes[part]) part = p;

  let wsum = 0, x = 0, y = 0, z = 0;
  for (let j = 0; j < K; j++) {
    const w = Math.exp((topSim[j] - topSim[0]) / XYZ_TEMPERATURE);
    const pt = map.points[topIdx[j]];
    wsum += w; x += w * pt.x; y += w * pt.y; z += w * (pt.z ?? 0);
  }
  return { part: partIds[part], xyz: [round(x / wsum), round(y / wsum), round(z / wsum)] };
}

const round = (v) => Math.round(v * 1e4) / 1e4;

// ---------------------------------------------------------------------------
// Loading

function loadMap() {
  mapPromise ??= (async () => {
    const progress = tracker('map');
    const mapUrl = new URL('data/map.json', base);
    const m = JSON.parse(new TextDecoder().decode(await readBytes(mapUrl, progress)));
    const vecUrl = new URL(m.vectors.file, mapUrl);
    const bytes = await readBytes(vecUrl, progress);
    const count = m.vectors.count;
    if (m.dims !== DIMS || count !== m.points.length || bytes.byteLength !== count * DIMS) {
      throw new Error(`engine: ${m.vectors.file} does not match map.json (count ${count}, points ${m.points.length}, bytes ${bytes.byteLength})`);
    }
    const i8 = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const v = new Float32Array(count * DIMS);
    for (let i = 0; i < count; i++) {
      const o = i * DIMS;
      let norm = 0;
      for (let d = 0; d < DIMS; d++) { const x = i8[o + d] / m.vectors.scale; v[o + d] = x; norm += x * x; }
      norm = Math.sqrt(norm) || 1;
      for (let d = 0; d < DIMS; d++) v[o + d] /= norm;
    }
    partIds = m.parts.map((p) => p.id);
    const index = new Map(partIds.map((id, i) => [id, i]));
    pointPart = Int32Array.from(m.points, (pt) => index.get(pt.part));
    centroids = new Float32Array(partIds.length * DIMS);
    m.parts.forEach((p, i) => centroids.set(p.centroid, i * DIMS));
    vectors = v;
    map = m;
    return m;
  })().catch((err) => { mapPromise = null; throw err; });
  return mapPromise;
}

// Sums byte progress over several files and reports it for one stage.
function tracker(stage) {
  const files = new Map();
  return (key, loaded, total) => {
    files.set(key, { loaded, total });
    let l = 0, t = 0;
    for (const f of files.values()) { l += f.loaded; t += f.total; }
    emit(stage, l, t);
  };
}

async function readBytes(url, progress) {
  if (IS_NODE) {
    const { readFile } = await import('node:fs/promises');
    const b = await readFile(url);
    progress(url.href, b.byteLength, b.byteLength);
    return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`engine: ${url} -> HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    progress(url.href, buf.byteLength, buf.byteLength);
    return buf;
  }
  const chunks = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    progress(url.href, loaded, Math.max(total, loaded));
  }
  progress(url.href, loaded, loaded);
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.byteLength; }
  return buf;
}

function loadModel() {
  modelPromise ??= (async () => {
    const vendor = new URL('vendor/', base);
    let T;
    const wasmPaths = {
      mjs: new URL('ort-wasm-simd-threaded.mjs', vendor).href,
      wasm: new URL('ort-wasm-simd-threaded.wasm', vendor).href,
    };
    if (IS_NODE) {
      // Run the browser build of transformers.js in Node, on its web code path, so Node uses the very
      // same JS, onnxruntime-web runtime and wasm binary as the browser (one shared vector space).
      // The build decides "am I in Node?" once, at import time, from process.release.name.
      const desc = Object.getOwnPropertyDescriptor(process, 'release');
      Object.defineProperty(process, 'release', { ...desc, value: { ...process.release, name: 'engine-web-path' } });
      try {
        T = await import(new URL('transformers.min.js', vendor).href);
      } finally {
        Object.defineProperty(process, 'release', desc);
      }
      T.env.fetch = nodeFileFetch;
    } else {
      T = await import(new URL('transformers.min.js', vendor).href);
      const sameOrigin = new URL(base).origin;
      T.env.fetch = (input, init) => {
        const url = new URL(input instanceof Request ? input.url : String(input), globalThis.location?.href);
        if (url.origin !== sameOrigin) return Promise.reject(new Error(`engine: offline mode, refusing ${url}`));
        return fetch(input, init);
      };
    }
    T.env.backends.onnx.wasm.wasmPaths = wasmPaths;
    T.env.backends.onnx.wasm.numThreads = 1; // no cross-origin isolation on a plain static server
    T.env.backends.onnx.wasm.proxy = false;
    T.env.allowRemoteModels = false;
    T.env.allowLocalModels = true;
    // Browser: a same-origin path, not a full URL. transformers.js 4.3 only probes optional files
    // (tokenizer_config.json) locally when localModelPath is not an http(s) URL.
    T.env.localModelPath = IS_NODE ? new URL('models/', base).href : new URL('models/', base).pathname;

    const files = new Map();
    const progress = tracker('model');
    const progress_callback = (e) => {
      if (e.status === 'progress' && e.file) files.set(e.file, { loaded: e.loaded ?? 0, total: e.total ?? 0 });
      else if (e.status === 'done' && e.file && files.has(e.file)) { const f = files.get(e.file); f.loaded = f.total; }
      else return;
      const f = files.get(e.file);
      progress(e.file, f.loaded, f.total);
    };
    const options = { dtype: 'q8', device: 'wasm', progress_callback };
    extractor = await T.pipeline('feature-extraction', MODEL_ID, options);
    await extractor('warm up', { pooling: 'mean', normalize: true });
    let total = 0;
    for (const f of files.values()) total += f.total;
    emit('model', total || 1, total || 1);
    return extractor;
  })().catch((err) => { modelPromise = null; throw err; });
  return modelPromise;
}

// Node only: serve file:// URLs from disk and refuse everything else (the demo is offline).
async function nodeFileFetch(input) {
  const url = new URL(input instanceof URL ? input.href : typeof input === 'string' ? input : input.url);
  if (url.protocol !== 'file:') throw new Error(`engine: offline mode, refusing ${url}`);
  const { readFile } = await import('node:fs/promises');
  try {
    const buf = await readFile(url);
    return new Response(buf, { status: 200, headers: { 'content-length': String(buf.byteLength) } });
  } catch (err) {
    if (err.code === 'ENOENT') return new Response(null, { status: 404, statusText: 'Not Found' });
    throw err;
  }
}

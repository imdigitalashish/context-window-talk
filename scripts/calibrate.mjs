// Calibrates map.json thresholds against demo/presets.json.
//
// touch:  a clause "touches" its part when sim >= touch. Each preset expectation turns into a bound:
//         parts that must light up give upper bounds, parts that must stay dark give lower bounds.
//         Generic chit-chat ("hello", "thanks", ...) should not touch anything (soft lower bound).
//         touch = the midpoint of the feasible interval, rounded to 0.01.
// strong: the midpoint between the scattered preset's Focus and the focused preset's Focus, so a
//         focused prompt's top bar clears it and a scattered prompt's doesn't.
//
// Then run `node demo/test-engine.mjs --write` to regenerate demo/data/presets.results.json.
import { readFile, writeFile } from 'node:fs/promises';
import * as engine from '../demo/engine.js';

const DEMO = new URL('../demo/', import.meta.url);
const MAP_FILE = new URL('data/map.json', DEMO);
const CHITCHAT = ['hello', 'hi', 'thanks', 'ok do it', 'can you help me', 'what is the capital of France'];

const presets = JSON.parse(await readFile(new URL('presets.json', DEMO), 'utf8'));
await engine.init();

let lo = 0, hi = 1;
const why = { lo: 'none', hi: 'none' };
const raise = (v, reason) => { if (v > lo) { lo = v; why.lo = reason; } };
const lower = (v, reason) => { if (v < hi) { hi = v; why.hi = reason; } };
const maxSimOf = (clauses, part) => Math.max(-1, ...clauses.filter((c) => c.part === part).map((c) => c.sim));

const results = {};
for (const p of presets) {
  const r = await engine.score(p.prompt);
  results[p.key] = r;
  const e = p.expect;
  for (const part of [...(e.touchedExactly ?? []), ...(e.touchedIncludes ?? [])]) {
    lower(maxSimOf(r.clauses, part), `preset ${p.key}: ${part} must light up`);
  }
  if (e.touchedExactly) {
    for (const c of r.clauses) if (!e.touchedExactly.includes(c.part)) raise(c.sim, `preset ${p.key}: "${c.text}" (${c.part}) must stay dark`);
  }
  if (e.minTouched) {
    const perPart = [...new Set(r.clauses.map((c) => c.part))].map((part) => maxSimOf(r.clauses, part)).sort((a, b) => b - a);
    lower(perPart[e.minTouched - 1] ?? -1, `preset ${p.key}: needs ${e.minTouched} parts`);
  }
}
const hardLo = lo;
for (const text of CHITCHAT) {
  const r = await engine.score(text);
  for (const c of r.clauses) raise(c.sim, `chit-chat "${c.text}" (${c.part}) should stay dark`);
}
if (lo >= hi) {
  console.warn(`soft bound ${lo.toFixed(4)} (${why.lo}) conflicts with ${hi.toFixed(4)}; using hard bounds only`);
  lo = hardLo;
}
if (lo >= hi) throw new Error(`no feasible touch threshold: need > ${lo} and <= ${hi}`);

let touch = Math.round(((lo + hi) / 2) * 100) / 100;
if (!(touch > lo && touch <= hi)) touch = Math.round(((lo + hi) / 2) * 1e4) / 1e4;

const focused = presets.find((p) => p.expect.touchedExactly);
const scattered = presets.find((p) => p.expect.focusBelow);
const strong = Math.round(((results[focused.key].whole.top.sim + results[scattered.key].whole.top.sim) / 2) * 100) / 100;

console.log(`touch:  feasible (${lo.toFixed(4)}, ${hi.toFixed(4)}]  -> ${touch}`);
console.log(`        lower bound: ${why.lo}`);
console.log(`        upper bound: ${why.hi}`);
console.log(`strong: between scattered Focus ${results[scattered.key].whole.top.sim} and focused Focus ${results[focused.key].whole.top.sim} -> ${strong}`);

const map = JSON.parse(await readFile(MAP_FILE, 'utf8'));
map.thresholds = { touch, strong };
await writeFile(MAP_FILE, JSON.stringify(map));
console.log(`wrote thresholds to ${MAP_FILE.pathname}`);

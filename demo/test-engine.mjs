#!/usr/bin/env node
// Engine checks + preset calibration table. Exits non-zero if anything fails.
//
//   node demo/test-engine.mjs           run the checks
//   node demo/test-engine.mjs --write   also (re)write demo/data/presets.results.json, the UI's offline fallback
//
// Needs no npm install: the engine runs the vendored transformers.js build on its wasm runtime.
import { readFile, writeFile } from 'node:fs/promises';
import * as engine from './engine.js';

const WRITE = process.argv.includes('--write');
const here = (p) => new URL(p, import.meta.url);
const RESULTS_FILE = here('./data/presets.results.json');

const failures = [];
const check = (ok, msg) => { if (!ok) failures.push(msg); return ok; };
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const pad = (s, n) => String(s).padEnd(n);

const t0 = Date.now();
await engine.init();
const map = engine.getMap();
const presets = JSON.parse(await readFile(here('./presets.json'), 'utf8'));
console.log(`engine ready in ${Date.now() - t0} ms: ${map.points.length} points, ${map.parts.length} parts, touch ${map.thresholds.touch}, strong ${map.thresholds.strong}\n`);

// --- map integrity
const ids = map.parts.map((p) => p.id);
check(map.points.length === map.vectors.count, 'points.length !== vectors.count');
check(map.parts.length >= 14 && map.parts.length <= 16, `expected 14-16 parts, got ${map.parts.length}`);
check(new Set(map.parts.map((p) => p.color.toLowerCase())).size === map.parts.length, 'part colours are not distinct');
check(map.points.every((p) => ids.includes(p.part)), 'a point has an unknown part');
check([...map.points, ...map.parts].every((p) => ['x', 'y', 'z'].every((k) => p[k] >= 0 && p[k] <= 1)), 'a coordinate is outside [0,1]');
check(map.parts.every((p) => Math.abs(Math.hypot(...p.centroid) - 1) < 1e-3 && p.centroid.length === map.dims), 'a centroid is not L2-normalised');
check(map.parts.every((p) => p.examples.length === 3), 'a part does not have 3 examples');
check(map.thresholds.touch > 0 && map.thresholds.strong > map.thresholds.touch, 'thresholds not calibrated');

// --- splitClauses
const split = (s) => JSON.stringify(engine.splitClauses(s));
check(split("Build a simple admin dashboard. Don't use React.") === JSON.stringify(['Build a simple admin dashboard', "Don't use React"]), `negation split: ${split("Build a simple admin dashboard. Don't use React.")}`);
check(engine.splitClauses(presets.find((p) => p.key === '2').prompt).length === 7, 'scattered preset should split into 7 clauses');
check(split('Use Postgres and Redis') === JSON.stringify(['Use Postgres and Redis']), 'noun lists must not split');
check(split('- add tests\n- fix it; ok') === JSON.stringify(['add tests', 'fix it']), `bullets/; split: ${split('- add tests\n- fix it; ok')}`);
check(engine.splitClauses('').length === 0 && engine.splitClauses('  ').length === 0, 'empty prompt must give no clauses');

// --- embed
const a = await engine.embed('Use React for the UI');
const b = await engine.embed("Don't use React for the UI");
check(a instanceof Float32Array && a.length === 384, 'embed must return Float32Array(384)');
check(Math.abs(Math.hypot(...a) - 1) < 1e-4, 'embed must be L2-normalised');
const negationCos = dot(a, b);

// --- presets
const results = {};
for (const p of presets) results[p.key] = await engine.score(p.prompt);
const rows = [];
for (const p of presets) {
  const r = results[p.key];
  const e = p.expect;
  const fails = [];
  const has = (part) => r.partsTouched.includes(part);
  if (e.touchedExactly && !(r.partsTouched.length === e.touchedExactly.length && e.touchedExactly.every(has))) fails.push(`touched ${r.partsTouched.join(',') || 'nothing'}, want exactly ${e.touchedExactly.join(',')}`);
  if (e.touchedIncludes && !e.touchedIncludes.every(has)) fails.push(`touched ${r.partsTouched.join(',') || 'nothing'}, want ${e.touchedIncludes.join(',')}`);
  if (e.minTouched && r.partsTouched.length < e.minTouched) fails.push(`touched ${r.partsTouched.length} parts, want >= ${e.minTouched}`);
  if (e.topPart && r.whole.top.part !== e.topPart) fails.push(`top part ${r.whole.top.part}, want ${e.topPart}`);
  if (e.focusBelow) {
    const ref = results[e.focusBelow.key].whole.top.sim;
    if (!(r.whole.top.sim <= ref - e.focusBelow.by)) fails.push(`Focus ${r.whole.top.sim} not at least ${e.focusBelow.by} below preset ${e.focusBelow.key}'s ${ref}`);
  }
  const shapeOk = r.clauses.every((c) => c.xyz.length === 3 && c.xy[0] === c.xyz[0] && c.xy[1] === c.xyz[1]) && r.whole.xyz?.length === 3;
  if (!shapeOk) fails.push('xyz/xy shape');
  fails.forEach((f) => failures.push(`preset ${p.key} (${p.name}): ${f}`));
  rows.push({ p, r, ok: fails.length === 0 });
}

console.log(`${pad('Preset', 14)}${pad('Clauses', 9)}${pad('Parts touched', 54)}${pad('Focus (top)', 22)}${pad('Must show', 62)}Result`);
console.log('-'.repeat(167));
for (const { p, r, ok } of rows) {
  console.log(`${pad(`${p.key} ${p.name}`, 14)}${pad(r.clauses.length, 9)}${pad(`${r.partsTouched.length}: ${r.partsTouched.join(', ')}`, 54)}${pad(`${r.whole.top.sim.toFixed(3)} ${r.whole.top.part}`, 22)}${pad(p.mustShow, 62)}${ok ? 'PASS' : 'FAIL'}`);
}
console.log('\nClauses (part, sim to that part\'s centroid, * = touched):');
for (const { p, r } of rows) {
  console.log(`  ${p.key} ${p.name}`);
  for (const c of r.clauses) console.log(`      ${c.touched ? '*' : ' '} ${pad(c.part, 14)}${c.sim.toFixed(3)}  "${c.text}"`);
}
console.log(`\nNegation check: cos("Use React for the UI", "Don't use React for the UI") = ${negationCos.toFixed(4)}`);

// --- offline fallback
const fallback = {
  model: map.model,
  thresholds: map.thresholds,
  note: 'Precomputed score() output for demo/presets.json, for when the model cannot load. Regenerate with: node demo/test-engine.mjs --write',
  results: presets.map((p) => ({ key: p.key, name: p.name, prompt: p.prompt, result: results[p.key] })),
};
if (WRITE) {
  await writeFile(RESULTS_FILE, JSON.stringify(fallback, null, 1) + '\n');
  console.log(`\nwrote ${RESULTS_FILE.pathname}`);
} else {
  let stored = null;
  try { stored = JSON.parse(await readFile(RESULTS_FILE, 'utf8')); } catch { /* reported below */ }
  check(stored && JSON.stringify(stored.results) === JSON.stringify(fallback.results) && JSON.stringify(stored.thresholds) === JSON.stringify(map.thresholds),
    'data/presets.results.json is missing or stale: run node demo/test-engine.mjs --write');
}

if (failures.length) {
  console.log(`\nFAIL (${failures.length})`);
  for (const f of failures) console.log(`  - ${f}`);
  process.exit(1);
}
console.log('\nPASS: all checks');

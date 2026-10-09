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

// --- splitClauses: [prompt, expected clauses]
const SPLIT_CASES = [
  ["Build a simple admin dashboard. Don't use React.", ['Build a simple admin dashboard', "Don't use React"]],
  ['Use Postgres and Redis', ['Use Postgres and Redis']],
  ['Set up Postgres and Redis', ['Set up Postgres and Redis']],
  ['Use React and Redux', ['Use React and Redux']],
  ['Add drag and drop to the kanban board', ['Add drag and drop to the kanban board']],
  ['Configure the Postgres and Redis containers', ['Configure the Postgres and Redis containers']],
  ['It should also support dark mode', ['It should also support dark mode']],
  ['- add tests\n- fix it; ok', ['add tests', 'fix it']],
  ['please add tests and also please fix the login bug', ['add tests', 'fix the login bug']],
  ["create me ai model and website for it and also why isn't she texting me?", ['create me ai model', 'website for it', "why isn't she texting me?"]],
  ['why is he not replying to my texts, should I text him again?', ['why is he not replying to my texts', 'should I text him again?']],
  ["build an admin dashboard but don't use react", ['build an admin dashboard', "don't use react"]],
  ['build me a website and a mobile app and an ai chatbot', ['build me a website', 'a mobile app', 'an ai chatbot']],
  ['', []],
  ['  ', []],
];
for (const [prompt, want] of SPLIT_CASES) {
  const got = engine.splitClauses(prompt);
  check(JSON.stringify(got) === JSON.stringify(want), `splitClauses(${JSON.stringify(prompt)}) = ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}
check(engine.splitClauses(presets.find((p) => p.key === '2').prompt).length === 7, 'scattered preset should split into 7 clauses');

// --- embed
const a = await engine.embed('Use React for the UI');
const b = await engine.embed("Don't use React for the UI");
check(a instanceof Float32Array && a.length === 384, 'embed must return Float32Array(384)');
check(Math.abs(Math.hypot(...a) - 1) < 1e-4, 'embed must be L2-normalised');
const negationCos = dot(a, b);

// --- presets, plus test-only cases (not UI presets, so not in presets.json)
const EXTRA_CASES = [
  {
    key: 'L1', name: 'Typed live (vague)',
    // The speaker's original live prompt. Preset 5 is the same prompt with "an" and "react" added;
    // this case documents that "website for it" is too vague to land anywhere.
    prompt: "create me ai model and website for it and also why isn't she texting me?",
    mustShow: "ml + interpersonal; \"website for it\" stays dark",
    expect: { minClauses: 3, touchedIncludes: ['ml', 'interpersonal'], untouched: ['website for it'] },
    explain: true,
  },
  {
    key: 'R1', name: 'Negation (regression)',
    prompt: "Build the dashboard. Don't use React.",
    mustShow: 'React still lights up',
    expect: { touchedIncludes: ['react'] },
  },
];

function unmet(e, r) {
  const out = [];
  const has = (part) => r.partsTouched.includes(part);
  const touched = r.partsTouched.join(',') || 'nothing';
  if (e.minClauses && r.clauses.length < e.minClauses) out.push(`${r.clauses.length} clauses, want >= ${e.minClauses}`);
  if (e.touchedExactly && !(r.partsTouched.length === e.touchedExactly.length && e.touchedExactly.every(has))) out.push(`touched ${touched}, want exactly ${e.touchedExactly.join(',')}`);
  for (const part of e.touchedIncludes ?? []) if (!has(part)) out.push(`touched ${touched}, want ${part}`);
  for (const text of e.untouched ?? []) {
    const c = r.clauses.find((cl) => cl.text === text);
    if (!c) out.push(`no clause "${text}"`);
    else if (c.touched) out.push(`"${text}" touched ${c.part} (${c.sim}), want it below touch`);
  }
  if (e.minTouched && r.partsTouched.length < e.minTouched) out.push(`touched ${r.partsTouched.length} parts, want >= ${e.minTouched}`);
  if (e.topPart && r.whole.top.part !== e.topPart) out.push(`top part ${r.whole.top.part}, want ${e.topPart}`);
  if (e.focusBelow) {
    const ref = results[e.focusBelow.key].whole.top.sim;
    if (!(r.whole.top.sim <= ref - e.focusBelow.by)) out.push(`Focus ${r.whole.top.sim} not at least ${e.focusBelow.by} below preset ${e.focusBelow.key}'s ${ref}`);
  }
  return out;
}

// Where a clause's dot lands: its nearest part label (labels sit at each part's x, y, z).
const labelDist = (xyz, p) => Math.hypot(xyz[0] - p.x, xyz[1] - p.y, xyz[2] - p.z);
const nearestLabel = (xyz) => map.parts.reduce((a, b) => (labelDist(xyz, a) <= labelDist(xyz, b) ? a : b)).id;

const results = {};
const cases = [...presets, ...EXTRA_CASES];
for (const c of cases) results[c.key] = await engine.score(c.prompt);
const rows = [];
for (const c of cases) {
  const r = results[c.key];
  const fails = unmet(c.expect, r);
  const shapeOk = r.clauses.every((cl) => cl.xyz.length === 3 && cl.xy[0] === cl.xyz[0] && cl.xy[1] === cl.xyz[1]) && r.whole.xyz?.length === 3;
  if (!shapeOk) fails.push('xyz/xy shape');
  for (const cl of r.clauses) {
    const label = nearestLabel(cl.xyz);
    if (label !== cl.part) fails.push(`"${cl.text}" (${cl.part}) lands nearest the ${label} label`);
  }
  fails.forEach((f) => failures.push(`case ${c.key} (${c.name}): ${f}`));
  rows.push({ c, r, ok: fails.length === 0 });
}

console.log(`${pad('Case', 26)}${pad('Clauses', 9)}${pad('Parts touched', 54)}${pad('Focus (top)', 22)}${pad('Must show', 60)}Result`);
console.log('-'.repeat(177));
for (const { c, r, ok } of rows) {
  console.log(`${pad(`${c.key} ${c.name}`, 26)}${pad(r.clauses.length, 9)}${pad(`${r.partsTouched.length}: ${r.partsTouched.join(', ')}`, 54)}${pad(`${r.whole.top.sim.toFixed(3)} ${r.whole.top.part}`, 22)}${pad(c.mustShow, 60)}${ok ? 'PASS' : 'FAIL'}`);
}
console.log('\nClauses (part, sim to that part\'s centroid, * = touched):');
for (const { c, r } of rows) {
  console.log(`  ${c.key} ${c.name}`);
  for (const cl of r.clauses) console.log(`      ${cl.touched ? '*' : ' '} ${pad(cl.part, 14)}${cl.sim.toFixed(3)}  "${cl.text}"`);
}
for (const { c, r } of rows.filter((row) => row.c.explain)) {
  console.log(`\nCase ${c.key}, nearest part centroids per clause:`);
  for (const cl of r.clauses) {
    const q = await engine.embed(cl.text);
    const near = map.parts.map((p) => [p.id, dot(q, p.centroid)]).sort((x, y) => y[1] - x[1]).slice(0, 4);
    console.log(`      "${cl.text}": ${near.map(([id, v]) => `${id} ${v.toFixed(3)}`).join(', ')}`);
  }
}
const far = rows.flatMap(({ r }) => r.clauses).map((cl) => ({ cl, d: labelDist(cl.xyz, map.parts.find((p) => p.id === cl.part)) }))
  .sort((a, b) => b.d - a.d)[0];
console.log(`\nLanding: each clause lands among its own part's nearest titles, so nearest its own part's label (checked);`);
console.log(`  farthest from its label: "${far.cl.text}" (${far.cl.part}) at ${far.d.toFixed(3)}. The whole-prompt dot averages across parts.`);
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

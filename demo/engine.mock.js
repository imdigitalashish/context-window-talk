// Mock of demo/engine.js: the exact API from docs/demo-spec.md, with fake but plausible data.
// 16 parts, 2,000 points in islands, keyword-driven "embeddings" with the same kNN scoring rules.
// Deterministic: the same prompt always gives the same result.

const DIMS = 384;
const PER_PART = 125;

const PARTS = [
  ['react', 'React & frontend', '#149ECA', 0.22, 0.27, ['How do I update state inside useEffect without an infinite loop?', 'Why does my React component render twice?', 'Passing props to a child component in React', 'useState not updating immediately after setState', 'How to conditionally render a component in JSX', 'React Router redirect after login']],
  ['css', 'CSS & styling', '#9B59B6', 0.09, 0.46, ['How do I vertically center a div?', 'Flexbox: make the last item stick to the right', 'Why is my z-index not working?', 'CSS grid with equal height rows', 'Make a background image cover the whole page', 'Smooth hover transition on a button']],
  ['ux', 'UX & design', '#E377C2', 0.18, 0.67, ['Should the primary button be on the left or the right?', 'Is infinite scroll bad for usability?', 'How many fields should a signup form have?', 'Best way to show an empty state on a dashboard', 'Writing microcopy for error messages', 'Hamburger menu or tab bar on mobile?']],
  ['android', 'Android', '#2CA02C', 0.37, 0.11, ['RecyclerView not showing items', 'How to request location permission on Android 13', 'Kotlin coroutine vs AsyncTask', 'App crashes on screen rotation', 'Change the status bar color in Android', 'Gradle build fails after updating Android Studio']],
  ['testing', 'Unit testing', '#9DA61D', 0.41, 0.33, ['How to mock a module in Jest?', 'Unit test a function that calls an API', 'What should I assert in a unit test?', 'pytest fixture scope explained', 'How do I measure code coverage?', 'Testing private methods: yes or no?']],
  ['auth', 'Auth & login', '#E5242B', 0.33, 0.56, ['How to limit login attempts per IP?', 'Store JWT in localStorage or cookies?', 'Implementing Google OAuth login', 'How to hash passwords with bcrypt', 'Session expires too soon after login', 'Refresh tokens: how do they work?']],
  ['security', 'Security', '#7F1D1D', 0.48, 0.76, ['How to prevent brute-force attacks?', 'Is it safe to put an API key in frontend code?', 'What is a CSRF token for?', 'Preventing SQL injection in a login form', 'How does HTTPS protect my data?', 'Should I rate limit by IP or by account?']],
  ['postgres', 'Postgres & SQL', '#336791', 0.56, 0.40, ['How to index a JSONB column in PostgreSQL', 'Postgres query is slow with a LEFT JOIN', 'Difference between serial and identity columns', 'How do I run a database migration safely?', 'Find duplicate rows in a Postgres table', 'Connection pool exhausted in Postgres']],
  ['docker', 'Docker & deploy', '#0E7C86', 0.63, 0.17, ['Docker container exits immediately', 'How to pass environment variables to docker compose', 'Reduce the size of a Docker image', 'Deploy a Node app with Docker on a VPS', 'Docker volume permissions denied', 'Difference between CMD and ENTRYPOINT']],
  ['payments', 'Stripe & payments', '#635BFF', 0.62, 0.62, ['Stripe webhook signature verification fails', 'How to handle subscription upgrades in Stripe', 'Stripe Checkout in test mode', 'Refund a payment with the Stripe API', 'Save a card for future payments', 'Payment intent requires_action status']],
  ['ml', 'Machine learning', '#FF7F0E', 0.83, 0.14, ['Why is my validation loss increasing?', 'How many epochs should I train for?', 'Normalise features before training?', 'What is overfitting in machine learning?', 'Choosing a learning rate', 'Train/test split for time series data']],
  ['money', 'Personal finance', '#C99A06', 0.80, 0.55, ['Should I pay off my loan or invest?', 'How much of my salary should I save?', 'Is a credit card EMI a bad idea?', 'How do index funds work?', 'Building an emergency fund', 'Tax on freelance income']],
  ['law', 'Law', '#6B4F3A', 0.92, 0.79, ['Can my landlord keep my deposit?', 'Is it legal to record a phone call?', 'Do I own the code I write at work?', 'Can I use a song in my YouTube video?', 'What happens if I break a contract?', 'Is a verbal agreement binding?']],
  ['workplace', 'Workplace', '#4E7A3A', 0.89, 0.35, ['My manager takes credit for my work', 'How to ask for a raise', 'Coworker keeps interrupting me in meetings', 'How do I resign politely?', 'Is it okay to decline a meeting?', 'Negotiating a job offer']],
  ['interpersonal', 'Relationships', '#D6336C', 0.71, 0.88, ['He stopped replying to my messages. What should I do?', 'How do I tell a friend they hurt me?', 'Should I text her first?', 'My friend always cancels plans', 'How to apologise after an argument', 'Is it normal to feel ignored in a relationship?']],
  ['cooking', 'Cooking', '#B5541C', 0.29, 0.88, ['Why is my rice always sticky?', 'How do I stop dal from boiling over?', 'Substitute for eggs in baking', 'How long to marinate chicken?', 'Why does my paneer turn rubbery?', 'Best oil for deep frying']],
];

// Keyword -> part weights. Negation is ignored on purpose: "don't use React" still says "React".
const KEYWORDS = [
  [/\breact(js)?\b|\bjsx\b|\bhooks?\b|\bredux\b|\bnext\.?js\b/, { react: 1 }],
  [/\bfront-?end\b/, { react: 0.5, css: 0.2 }],
  [/\bcomponents?\b|\bvue\b|\bangular\b|\bsvelte\b/, { react: 0.6 }],
  [/\bdashboard\b/, { ux: 0.6, react: 0.45 }],
  [/\badmin\b/, { ux: 0.3, react: 0.1 }],
  [/\bcss\b|\bflexbox\b|\btailwind\b|\bstyl(e|es|ing)\b/, { css: 1 }],
  [/\bbeautiful\b|\bpretty\b|\bgorgeous\b/, { css: 0.6, ux: 0.25 }],
  [/\blook\b/, { css: 0.3, ux: 0.15 }],
  [/\bresponsive\b|\bfonts?\b|\bcolou?rs?\b|\blayout\b|\banimations?\b/, { css: 0.7 }],
  [/\bux\b|\busability\b|\bonboarding\b|\buser experience\b|\bwireframes?\b/, { ux: 1 }],
  [/\blanding page\b/, { ux: 0.7, css: 0.15 }],
  [/\bcopy(writing)?\b|\bheadline\b|\btagline\b/, { ux: 0.55 }],
  [/\bdesign\b/, { ux: 0.5, css: 0.3 }],
  [/\blog ?in\b|\bsign ?in\b|\bsign ?up\b|\blogout\b/, { auth: 1 }],
  [/\boauth\b|\bjwt\b|\bpasswords?\b|\bsessions?\b|\bauth(entication|orization)?\b|\b2fa\b|\btokens?\b/, { auth: 1 }],
  [/\bgoogle\b/, { auth: 0.3 }],
  [/\brate[- ]?limit(ing|s)?\b/, { auth: 0.6, security: 0.35 }],
  [/\battempts?\b/, { auth: 0.5, security: 0.15 }],
  [/\b429\b|\bhttp\b|\bendpoints?\b/, { auth: 0.55, security: 0.2 }],
  [/\bip\b/, { security: 0.35, auth: 0.1 }],
  [/\bsecur(e|ity)\b|\bxss\b|\bcsrf\b|\bvulnerab\w*|\bencrypt\w*|\bhack\w*|\bbrute[- ]force\b|\bmalware\b/, { security: 1 }],
  [/\bpostgres(ql)?\b|\bsql\b|\bdatabases?\b|\bqueries\b|\bquery\b|\bschema\b|\bmigrations?\b/, { postgres: 0.9 }],
  [/\bdocker\b|\bcontainers?\b|\bkubernetes\b|\bdockerfile\b|\bcompose\b/, { docker: 1 }],
  [/\bdeploy\w*|\bhosting\b|\bci\/?cd\b|\bservers?\b/, { docker: 0.75 }],
  [/\bstripe\b|\bpayments?\b|\bcheckout\b|\bsubscriptions?\b|\bbilling\b|\binvoices?\b|\brefunds?\b/, { payments: 1 }],
  [/\bmachine learning\b|\bml\b|\bmodels?\b|\bneural\b|\btrain(ing)?\b|\bdatasets?\b|\bllms?\b|\bgpt\b|\bai\b/, { ml: 0.9 }],
  [/\bandroid\b|\bkotlin\b|\bapk\b|\bplay store\b/, { android: 1 }],
  [/\bmobile\b|\bapps?\b/, { android: 0.3 }],
  [/\bunit tests?\b|\btests?\b|\btesting\b|\bjest\b|\bpytest\b|\bmocks?\b|\bcoverage\b/, { testing: 1 }],
  [/\blaw\b|\blegal\b|\bcontracts?\b|\bsue\b|\blawsuit\b|\bcopyright\b|\blandlord\b|\bcourt\b|\bgdpr\b/, { law: 1 }],
  [/\bmoney\b|\bsalary\b|\binvest\w*|\btax\w*|\bloans?\b|\bsavings?\b|\bbudget\b|\bemi\b|\brupees?\b/, { money: 1 }],
  [/\brepl(y|ying|ied|ies)\b|\bghost\w*|\bcrush\b|\bboyfriend\b|\bgirlfriend\b|\brelationships?\b|\bfriends?\b|\bignor\w*|\blove\b|\bbreak ?up\b/, { interpersonal: 1 }],
  [/\bmessages?\b|\btext(s|ing|ed)?\b|\bchat\b/, { interpersonal: 0.55 }],
  [/\bhe\b|\bshe\b|\bhim\b|\bher\b/, { interpersonal: 0.3 }],
  [/\bboss\b|\bmanager\b|\bcoworkers?\b|\bcolleagues?\b|\bjobs?\b|\boffice\b|\binterviews?\b|\bresign\w*|\bpromotion\b|\bappraisal\b/, { workplace: 1 }],
  [/\bcook\w*|\brecipes?\b|\bchicken\b|\brice\b|\bbak(e|ing)\b|\boven\b|\bcurry\b|\bspices?\b|\bdal\b|\bpaneer\b|\bbiryani\b/, { cooking: 1 }],
];

const IMPERATIVE = new Set(('add build create make set setup write deploy integrate use implement fix return configure install design update ' +
  'remove delete refactor test run ship host connect send show display handle generate optimize optimise improve migrate translate explain ' +
  'give put keep allow enable store save include support do dont don\'t never avoid please style log track cache validate check ensure ' +
  'change move rename convert paginate sort filter upload download let help tell find get call fetch compute apply prevent limit block ' +
  'protect encrypt hash secure document review plan scale monitor notify email hook wire publish launch').split(/\s+/));

const LIFE = new Set(['money', 'law', 'workplace', 'interpersonal', 'cooking']);
// Island depth (3rd UMAP axis): tech parts and everyday-life parts sit in different layers, with spread inside each.
const DEPTH = { react: 0.32, css: 0.18, ux: 0.42, android: 0.62, testing: 0.7, auth: 0.5, security: 0.78, postgres: 0.38,
  docker: 0.58, payments: 0.24, ml: 0.82, money: 0.36, law: 0.62, workplace: 0.2, interpersonal: 0.74, cooking: 0.5 };
const THRESHOLDS = { touch: 0.33, strong: 0.5 };

const sleep = ms => new Promise(r => setTimeout(r, ms));
const round3 = x => Math.round(x * 1000) / 1000;
const clamp01 = x => Math.min(0.97, Math.max(0.03, x));

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function gauss(rand) { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); }
function normalize(v) {
  let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1; for (let i = 0; i < v.length; i++) v[i] /= s;
  return v;
}
function randomUnit(rand) { const v = new Float32Array(DIMS); for (let i = 0; i < DIMS; i++) v[i] = gauss(rand); return normalize(v); }
function combine(terms) {
  const v = new Float32Array(DIMS);
  for (const [w, u] of terms) for (let i = 0; i < DIMS; i++) v[i] += w * u[i];
  return normalize(v);
}
function dot(a, b, off = 0) { let s = 0; for (let i = 0; i < DIMS; i++) s += a[i] * b[off + i]; return s; }

let world = null;
function build() {
  if (world) return world;
  const rand = mulberry32(20261010);
  const generic = randomUnit(rand);
  const domain = { tech: randomUnit(rand), life: randomUnit(rand) };
  const parts = PARTS.map(([id, label, color, x, y, titles]) => ({
    id, label, color, x, y, z: DEPTH[id] ?? 0.5, titles,
    seed: combine([[0.3, generic], [0.2, domain[LIFE.has(id) ? 'life' : 'tech']], [0.93, randomUnit(rand)]]),
  }));
  const points = [];
  const vectors = new Float32Array(parts.length * PER_PART * DIMS);
  for (const p of parts) {
    const blobs = [0, 1, 2].map(() => ({ dx: gauss(rand) * 0.028, dy: gauss(rand) * 0.024, dz: gauss(rand) * 0.03,
      sx: 0.016 + rand() * 0.014, sy: 0.014 + rand() * 0.012, sz: 0.016 + rand() * 0.014 }));
    const sum = new Float32Array(DIMS);
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < PER_PART; i++) {
      const b = blobs[i % 3], stray = rand() < 0.03;
      const x = clamp01(stray ? p.x + gauss(rand) * 0.09 : p.x + b.dx + gauss(rand) * b.sx);
      const y = clamp01(stray ? p.y + gauss(rand) * 0.09 : p.y + b.dy + gauss(rand) * b.sy);
      const z = clamp01(stray ? p.z + gauss(rand) * 0.09 : p.z + b.dz + gauss(rand) * b.sz);
      points.push({ x: round3(x), y: round3(y), z: round3(z), part: p.id, t: p.titles[i % p.titles.length] });
      const v = combine([[1, p.seed], [1, randomUnit(rand)]]);
      vectors.set(v, (points.length - 1) * DIMS);
      for (let d = 0; d < DIMS; d++) sum[d] += v[d];
      sx += x; sy += y; sz += z;
    }
    p.centroid = normalize(sum);
    p.x = round3(sx / PER_PART); p.y = round3(sy / PER_PART); p.z = round3(sz / PER_PART);
  }
  const map = {
    model: 'Xenova/all-MiniLM-L6-v2',
    dims: DIMS,
    source: 'synthetic (engine.mock.js)',
    thresholds: { ...THRESHOLDS },
    parts: parts.map(p => ({ id: p.id, label: p.label, color: p.color, x: p.x, y: p.y, z: p.z, count: PER_PART,
      centroid: Array.from(p.centroid, round3), examples: p.titles.slice(0, 3) })),
    points,
    vectors: { file: 'vectors.i8', dtype: 'int8', scale: 127, count: points.length },
  };
  world = { parts, points, vectors, generic, index: new Map(parts.map((p, i) => [p.id, i])), map };
  return world;
}

function knn(v, k = 12) {
  const { points, vectors } = build();
  const best = []; // [sim, index], descending
  for (let i = 0; i < points.length; i++) {
    const s = dot(v, vectors, i * DIMS);
    if (best.length < k || s > best[best.length - 1][0]) {
      let j = best.length < k ? best.length : k - 1;
      while (j > 0 && best[j - 1][0] < s) { best[j] = best[j - 1]; j--; }
      best[j] = [s, i];
    }
  }
  const vote = new Map();
  for (const [s, i] of best) vote.set(points[i].part, (vote.get(points[i].part) || 0) + Math.max(0, s));
  const part = [...vote].sort((a, b) => b[1] - a[1])[0][0];
  let wx = 0, wy = 0, wz = 0, ws = 0;
  for (const [s, i] of best) {
    const w = Math.exp((s - best[0][0]) / 0.05);
    wx += w * points[i].x; wy += w * points[i].y; wz += w * points[i].z; ws += w;
  }
  const xyz = [round3(wx / ws), round3(wy / ws), round3(wz / ws)];
  return { part, xyz, xy: xyz.slice(0, 2) };
}

const listeners = new Set();
let ready = null, latency = 1;

export function onProgress(cb) { listeners.add(cb); return () => listeners.delete(cb); }

async function fakeDownload(stage, total, ms) {
  const steps = ms > 0 ? Math.max(4, Math.round(ms / 70)) : 1;
  for (let i = 1; i <= steps; i++) {
    if (ms > 0) await sleep(ms / steps);
    for (const cb of listeners) { try { cb({ stage, loaded: Math.round((total * i) / steps), total }); } catch { /* ignore */ } }
  }
}

export async function init(opts = {}) {
  latency = opts.delayScale ?? 1;
  ready ??= (async () => {
    await fakeDownload('model', 22_900_000, 1400 * latency);
    build();
    await fakeDownload('map', 2_400_000, 450 * latency);
  })();
  return ready;
}

export function getMap() {
  if (!world) throw new Error('engine.mock: call init() first');
  return world.map;
}

export function splitClauses(prompt) {
  const text = String(prompt ?? '').replace(/\r/g, '').replace(/^[ \t]*(?:[-*•·]|\d+[.)])[ \t]+/gm, '\n');
  const out = [];
  for (const piece of text.split(/\n+|;\s*|(?<=[.!?])\s+(?=\S)/)) {
    const re = /(,\s*|\s+)((?:(?:and|then|also|plus)\s+)*)([a-z'’]+)/gi;
    let m, start = 0;
    while ((m = re.exec(piece))) {
      const word = m[3].toLowerCase().replace(/’/g, "'");
      const cut = (m[1].includes(',') || m[2].trim()) && IMPERATIVE.has(word);
      const wordStart = m.index + m[1].length + m[2].length;
      if (cut) { out.push(piece.slice(start, m.index)); start = wordStart; }
      re.lastIndex = wordStart;
    }
    out.push(piece.slice(start));
  }
  return out
    .map(s => s.trim().replace(/^(?:and|then|also|plus)\s+/i, '').replace(/[\s.,;:]+$/, '').trim())
    .filter(s => s.length >= 3);
}

export async function embed(text) {
  const { parts, index, generic } = build();
  const s = String(text ?? '').toLowerCase().trim();
  const weights = new Map();
  for (const [re, w] of KEYWORDS) if (re.test(s)) for (const [id, x] of Object.entries(w)) weights.set(id, (weights.get(id) || 0) + x);
  const total = [...weights.values()].reduce((a, b) => a + b, 0);
  const peak = Math.max(0, ...weights.values());
  // More keyword mass = stronger pull; spreading it over many parts averages the vector out.
  const alpha = total > 0 ? 0.66 * (1 - Math.exp(-total)) * (peak / total) ** 0.4 : 0;
  const terms = [[0.25, generic], [Math.sqrt(Math.max(0, 1 - alpha * alpha - 0.0625)), randomUnit(mulberry32(hash(s)))]];
  if (alpha) terms.push([alpha, combine([...weights].map(([id, x]) => [x, parts[index.get(id)].centroid]))]);
  return combine(terms);
}

export async function score(prompt) {
  await init();
  const { parts, index } = build();
  const texts = splitClauses(prompt);
  await sleep(latency * (60 + 25 * texts.length));
  const clauses = [];
  for (const text of texts) {
    const v = await embed(text);
    const { part, xyz, xy } = knn(v);
    const sim = round3(dot(v, parts[index.get(part)].centroid));
    clauses.push({ text, part, sim, xyz, xy, touched: sim >= THRESHOLDS.touch });
  }
  const partsTouched = [...new Set(clauses.filter(c => c.touched).map(c => c.part))];
  const wv = await embed(prompt);
  const sims = {};
  let top = null;
  for (const p of parts) {
    const s = round3(dot(wv, p.centroid));
    sims[p.id] = s;
    if (!top || s > top.sim) top = { part: p.id, sim: s };
  }
  const { xyz, xy } = knn(wv);
  return { clauses, partsTouched, whole: { sims, top, xyz, xy } };
}

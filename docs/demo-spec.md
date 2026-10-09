# Prompt footprint demo — shared spec (Clustering law)

This file is the contract between the **Engine** session and the **UI** session. Read all of it before starting.

## Why this demo exists

Talk: *"AI is like a girlfriend that ghosts you in your context window"* by Ashish Kumar Verma. 25 minutes, 10 Oct 2026. The audience is Indian college students and developers who use coding agents but don't know LLM internals. The talk rests on three laws: the **dumb zone**, the **clustering law**, and **U-shaped attention**.

**The clustering law (speaker's framing).** An LLM's knowledge sits in regions, or "parts." The words in your prompt decide which parts get pulled in.

- A focused prompt pulls one part.
- A scattered prompt ("build my frontend, backend, auth, payments and the landing page copy") pulls many, and the answer becomes averaged and generic.
- Sub-points:
  - **Negation blindness:** "don't use React" still pulls React.
  - **Instruction overload:** in IFScale (2025), the best model followed only 68% of 500 instructions.

This demo makes the spread visible with a real embedding model. Keep the on-screen framing honest: the embedding model (all-MiniLM-L6-v2) is a small cousin of an LLM's internal representations. It shows **how spread out** a prompt is; the **cost** of that spread comes from research such as IFScale.

## What the audience sees

The page uses a 1920×1080 stage, scaled to fit the window exactly as `deck/index.html` does.

- **Left (~62%):** the knowledge map, a **3-D point cloud rendered with Three.js**. About 2,000 points, each a real question, coloured by part, with each part labelled at its centroid.
- **Right (~38%):** the prompt panel.

The speaker picks a preset (keys 1–4) or types a prompt and presses Enter. Then:

1. The prompt splits into **clause chips**.
2. Each clause **travels to the spot on the map where it lands**, drawn as a line from the chip to the map. Touched parts light up and the others dim.
3. A huge counter shows **"Parts touched: N"**.
4. A bar chart, **"Where the whole prompt points,"** shows the cosine similarity of the whole prompt to each part's centroid. A focused prompt gives one tall bar; a scattered prompt gives many short bars and no tall one.
5. **Focus** is the whole prompt's top similarity; for example 0.62 for a focused prompt versus 0.31 for a scattered one.

### Presets

You may adjust the wording slightly while calibrating, but keep each preset's intent.

| Key | Name | Prompt | Must show |
|---|---|---|---|
| 1 | Focused | `Add rate limiting to the /login endpoint: max 5 attempts per IP per minute, then return HTTP 429.` | 1 part (auth) |
| 2 | Scattered | `Build my React frontend, set up the Postgres database, add Google login, integrate Stripe payments, write the landing page copy, deploy it with Docker and make it look beautiful.` | ≥ 5 parts, and a visibly lower Focus than preset 1 |
| 3 | Negation | `Build a simple admin dashboard. Don't use React.` | The React part still lights up |
| 4 | 3 AM | `why is he not replying to my messages, should I text him again?` | The relationships (interpersonal) part |

## Files and ownership (both sessions share one checkout)

| Path | Owner |
|---|---|
| `deck/**` | **Nobody: do not modify.** The speaker drives the deck. |
| `demo/engine.js`, `demo/data/**`, `demo/models/**`, `demo/vendor/**`, `demo/presets.json`, `demo/test-engine.mjs`, `scripts/**` | Engine session |
| `demo/index.html`, `demo/ui/**`, `demo/engine.mock.js`, `demo/screens/**` | UI session |
| `docs/demo-spec.md` | Shared. Only the Engine session appends the **Calibration** section. Changing the API requires messaging the coordinator first. |

Don't run git commands that create or switch branches, commit, stash or reset; the coordinator commits. Don't install anything globally. Keep project dependencies inside the repo (for example `scripts/package.json` and `node_modules/`, which are gitignored).

## Engine API: the contract

`demo/engine.js` is an ES module. It must work in the browser (served over http) and in Node 18 or later.

```js
export async function init(opts = {})            // loads the model from demo/models/, demo/data/map.json and the vectors; idempotent
export function splitClauses(prompt)              // -> string[]
export async function embed(text)                 // -> Float32Array(384), mean-pooled, L2-normalised
export async function score(prompt)               // -> ScoreResult (below)
export function getMap()                          // -> parsed map.json (after init)
export function onProgress(cb)                    // optional: cb({ stage: 'model'|'map', loaded, total })

// ScoreResult
{
  clauses: [{ text, part, sim, xyz: [x, y, z], xy: [x, y], touched }], // part = nearest part id; touched = sim >= thresholds.touch; xy = first two of xyz
  partsTouched: ['auth', ...],                          // unique part ids where touched
  whole: { sims: { react: 0.21, auth: 0.55, ... },      // whole prompt vs every part centroid
           top: { part: 'auth', sim: 0.55 },
           xyz: [x, y, z], xy: [x, y] }
}
```

- `splitClauses` splits on sentence ends, newlines, bullets, `;`, and on `, and` / ` and ` / ` also ` / ` plus ` / ` then ` between imperative clauses. Keep a negation attached to its object: "Don't use React" stays one clause. Drop empty clauses and clauses of fewer than three characters.
- **Clause part.** Use the nearest part by kNN vote: take the 12 nearest corpus points by cosine similarity, weight each by its similarity, and pick the part with the most weight. `sim` is the clause's cosine similarity to that part's centroid. Use kNN rather than a raw centroid argmax because it is more robust.
- **`xyz`.** The weighted average of the 3-D positions of the clause's 12 nearest points, with weights of softmax(sim / 0.05). All coordinates are in [0,1], the same space as `points`. `xy` is kept for compatibility and is simply the first two coordinates of `xyz`.

## `demo/data/map.json`

```json
{
  "model": "Xenova/all-MiniLM-L6-v2",
  "dims": 384,
  "source": "Stack Exchange question titles, CC BY-SA 4.0",
  "thresholds": { "touch": 0.0, "strong": 0.0 },
  "parts": [
    { "id": "react", "label": "React & frontend", "color": "#hex", "x": 0.0, "y": 0.0, "z": 0.0, "count": 0,
      "centroid": [384 floats, L2-normalised], "examples": ["...", "...", "..."] }
  ],
  "points": [ { "x": 0.0, "y": 0.0, "z": 0.0, "part": "react", "t": "question title" } ],
  "vectors": { "file": "vectors.i8", "dtype": "int8", "scale": 127, "count": 0 }
}
```

- The vectors are stored in `demo/data/vectors.i8`, in the same order as `points`.
- Keep `map.json` and the vectors together under about 3 MB.
- `demo/data/presets.results.json` holds the precomputed `score()` output for the four presets. The UI falls back to it if the model can't load.

## Engine requirements

- **Offline at the venue.** Put the model files (quantized ONNX plus tokenizer and config) in `demo/models/`, and transformers.js plus the onnxruntime-web wasm files in `demo/vendor/`. Nothing may come from a CDN or Hugging Face at runtime.
- **One shared vector space.** Compute the corpus embeddings with the same model and the same transformers.js build that the browser uses (run transformers.js in Node), so the clause and corpus vectors are comparable.
- **Corpus.**
  - Use roughly 120–180 top-voted question titles per part from the Stack Exchange API v2.3. No API key is needed; there's a limit of about 300 requests per day, so respect `backoff`.
  - Aim for 14–16 parts. Suggestions:
    - Stack Overflow tags `reactjs`, `css`, `authentication`, `postgresql`, `docker`, `stripe-payments`, `machine-learning`, `android`, `unit-testing`.
    - Other sites: `ux`, `security`, `law`, `money`, `interpersonal`, `workplace`, `cooking`.
  - Decode HTML entities, dedupe, and keep English only.
  - If the API is unavailable, generate synthetic titles, set `source` to "synthetic", and tell the coordinator.
- **3-D layout.** Use UMAP with `n_components=3` and a cosine metric, or similar. Normalise each axis to [0,1] with padding. The parts should form readable islands in 3-D.
- **Calibration.**
  - Pick `thresholds.touch` so that all four presets behave as the table above requires.
  - `node demo/test-engine.mjs` prints a results table and exits non-zero if any preset fails.
  - Append a **Calibration** section to this file with the thresholds, the preset results table, and a cosine-similarity check of "Use React for the UI" against "Don't use React for the UI" (the negation evidence).

## UI requirements

- **Visual language: professional, not cartoony** (the speaker's direction). This page doesn't follow the deck's playful slide-1 style.
  - Aim for a polished data-visualisation or research-lab look: no Baloo 2 or Kalam, no comic offset shadows, no Doraemon-blue drench.
  - Fonts: the system UI stack (SF Pro, then Inter, then system-ui), with tabular figures for numbers.
  - Colour: a neutral background chosen for projector legibility, a restrained categorical palette for the parts, and one accent for highlights. Use thin, crisp lines and subtle motion.
  - The page will be projected, so no text smaller than about 24px at 1920 width.
- **Three.js for the embedding map.**
  - Render the points in 3-D (instanced points or sprites, about 2,000 of them), with slow auto-rotate and OrbitControls. Show part labels as billboards or CSS2D labels at the centroids.
  - On run, touched parts brighten or grow while the others dim. Each clause travels in 3-D from the prompt panel to its `xyz`, with a trail line, and the whole-prompt point is shown too.
  - Vendor three.js locally in `demo/ui/vendor/` so it works offline with no CDN. `demo/vendor/` belongs to the engine.
- **Stage and keys.** Use the same 1920×1080 stage scaling as the deck. Keys: `1`–`4` presets, `Enter` run, `Esc` clear, `F` fullscreen.
- **Engine.** Build against `demo/engine.mock.js` first: the same API, with fake but plausible data and parts. When the coordinator says the engine is ready, switch to `demo/engine.js`.
- **States:**
  - model loading (show progress)
  - ready
  - running
  - error (fall back to `demo/data/presets.results.json` and say so on screen)
- **Motion.** One authored moment: the clauses travelling in 3-D to where they land. Respect `prefers-reduced-motion`.
- **Footnote:** "Embedding model: all-MiniLM-L6-v2 · dots: real Stack Exchange questions."

## Running it

Run `./serve.sh`, then open http://localhost:8000/demo/. The model files need http; they won't load from `file://`.

## Calibration

Engine-generated on 9 Oct 2026 from the shipped `demo/data/map.json`. `node demo/test-engine.mjs` re-checks all of this and exits non-zero on any failure (Node 18.20 and 25; Chrome gives identical numbers).

**Thresholds** (`map.json` → `thresholds`)

| Name | Value | Meaning and how it was set |
|---|---|---|
| `touch` | **0.21** | A clause touches its part when `sim >= 0.21`. The feasible window was (0.180, 0.237]: the 3 AM clause (0.237) must light up; chit-chat ("can you help me", 0.180) should not. The value is the midpoint. |
| `strong` | **0.43** | A strong pull: the focused preset's Focus (0.497) clears it, the scattered preset's (0.362) doesn't. The value is their midpoint. Use it, for example, as a reference line on the "Where the whole prompt points" chart. |

**Preset results** (Focus = `whole.top.sim`)

| Key | Preset | Clauses | Parts touched | Focus | Must show | Result |
|---|---|---|---|---|---|---|
| 1 | Focused | 2 | 1: auth | 0.497 (auth) | 1 part (auth) | PASS |
| 2 | Scattered | 7 | 6: react, postgres, auth, payments, docker, css | 0.362 (react) | ≥ 5 parts, visibly lower Focus than preset 1 | PASS (0.135 lower) |
| 3 | Negation | 2 | 1: react | 0.544 (react) | React still lights up | PASS ("Don't use React" → react, 0.767) |
| 4 | 3 AM | 1 | 1: interpersonal | 0.237 (interpersonal) | Relationships part | PASS |

Clause detail: preset 1 is "Add rate limiting … per minute" → auth 0.480, "return HTTP 429" → auth 0.225. Preset 2 is React frontend → react 0.652, Postgres database → postgres 0.703, Google login → auth 0.427, Stripe payments → payments 0.821, landing page copy → writing 0.203 (below `touch`), Docker → docker 0.714, look beautiful → css 0.284. Preset 3 is admin dashboard → ux 0.140 (below `touch`), "Don't use React" → react 0.767.

**Negation evidence:** cos("Use React for the UI", "Don't use React for the UI") = **0.853**. To the model the two instructions are nearly the same point.

**Preset wording adjusted while calibrating** (intent kept; `demo/presets.json` holds the final text):
- Preset 1: "Add rate limiting to **user authentication on** the /login endpoint: max 5 **failed** attempts per IP per minute, then return HTTP 429." With the original wording, Focus was 0.404 against the scattered preset's 0.362, which isn't visibly lower.
- Preset 4: "why is he not replying to my **texts**, should I text him again?" The original "messages" scored 0.205, too close to chit-chat.
- Presets 2 and 3 are unchanged. Focus values are lower than the spec's illustrative 0.62 / 0.31: with all-MiniLM-L6-v2, a long prompt's cosine to a whole part's centroid tops out around 0.5.

**Parts** (16 parts, 125 top-voted titles each, 2,000 points): react (SO [reactjs]), css (SO [css]), auth (SO [authentication]), postgres (SO [postgresql]), docker (SO [docker]), payments (SO [stripe-payments], payment titles only), ml (SO [machine-learning]), android (SO [android]), testing (SO [unit-testing]), ux, security, writing, law, money, interpersonal (label "Relationships"), cooking (Stack Exchange sites). `workplace` was dropped, because it competed with Relationships on 3 AM-style prompts. `writing` was added so that "write the landing page copy" has a home; without it, the clause landed on law ("copy" ≈ copyright). In the 3-D layout, 90% of the points are nearer their own part's centroid than any other.

**Engine notes**
- `splitClauses` also splits on a plain comma, but only when the next clause starts with an imperative verb or a negation ("Build X, set up Y, add Z"). Noun lists such as "Postgres and Redis" stay together. This is what splits preset 2 into its 7 asks.
- `init()` loads the map first. `getMap()` returns it as soon as it has loaded, even if the model then fails, so the UI can still draw the points in its fallback state. `onProgress` reports bytes (`loaded`/`total`) per stage and returns an unsubscribe function. `score('')` returns no clauses, `whole.top.part = null` and `whole.xyz = xy = null`.
- `demo/data/presets.results.json` has the shape `{ model, thresholds, note, results: [{ key, name, prompt, result }] }`, where `result` is the exact `score()` output.
- Offline: the model is loaded with `allowRemoteModels = false`, local `localModelPath` and local wasm paths, and fetches are guarded so that any off-origin request is refused. The browser check (`scripts/browser-check.mjs`, headless Chrome) saw 18 requests, all to the local server.
- One vector space: Node runs the same vendored `transformers.min.js` (4.3.1), onnxruntime-web wasm (1.31.0-dev) and quantized model as the browser. Browser and Node scores are identical, and browser embeddings match the stored int8 vectors with cos ≥ 0.9989.
- `demo/package.json` (`"type": "module"`) lets Node 18–22.6 load `engine.js` and the vendored build as ES modules. Browsers ignore it.

**Rebuild** (needs network; not needed at the venue): `cd scripts && npm install`, then `node fetch-corpus.mjs` (responses cached in `scripts/.cache/`; 41 API requests used in total, against a quota of 300 per day), `node fetch-model.mjs`, `node vendor.mjs`, `node build-map.mjs` (deterministic, seeded), `node calibrate.mjs`, and finally `node ../demo/test-engine.mjs --write`.

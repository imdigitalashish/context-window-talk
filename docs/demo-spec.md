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

- **Left (~62%):** the knowledge map. About 2,000 dots, each a real question, coloured by part, with each part labelled at its centroid.
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
  clauses: [{ text, part, sim, xy: [x, y], touched }], // part = nearest part id; touched = sim >= thresholds.touch
  partsTouched: ['auth', ...],                          // unique part ids where touched
  whole: { sims: { react: 0.21, auth: 0.55, ... },      // whole prompt vs every part centroid
           top: { part: 'auth', sim: 0.55 },
           xy: [x, y] }
}
```

- `splitClauses` splits on sentence ends, newlines, bullets, `;`, and on `, and` / ` and ` / ` also ` / ` plus ` / ` then ` between imperative clauses. Keep a negation attached to its object: "Don't use React" stays one clause. Drop empty clauses and clauses of fewer than three characters.
- **Clause part.** Use the nearest part by kNN vote: take the 12 nearest corpus points by cosine similarity, weight each by its similarity, and pick the part with the most weight. `sim` is the clause's cosine similarity to that part's centroid. Use kNN rather than a raw centroid argmax because it is more robust.
- **`xy`.** The weighted average of the 2-D positions of the clause's 12 nearest points, with weights of softmax(sim / 0.05). Both coordinates are in [0,1], the same space as `points`.

## `demo/data/map.json`

```json
{
  "model": "Xenova/all-MiniLM-L6-v2",
  "dims": 384,
  "source": "Stack Exchange question titles, CC BY-SA 4.0",
  "thresholds": { "touch": 0.0, "strong": 0.0 },
  "parts": [
    { "id": "react", "label": "React & frontend", "color": "#hex", "x": 0.0, "y": 0.0, "count": 0,
      "centroid": [384 floats, L2-normalised], "examples": ["...", "...", "..."] }
  ],
  "points": [ { "x": 0.0, "y": 0.0, "part": "react", "t": "question title" } ],
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
- **2-D layout.** Use UMAP with a cosine metric, or similar. Normalise to [0,1] with padding. The parts should form readable islands.
- **Calibration.**
  - Pick `thresholds.touch` so that all four presets behave as the table above requires.
  - `node demo/test-engine.mjs` prints a results table and exits non-zero if any preset fails.
  - Append a **Calibration** section to this file with the thresholds, the preset results table, and a cosine-similarity check of "Use React for the UI" against "Don't use React for the UI" (the negation evidence).

## UI requirements

- **Visual language: match `deck/index.html`.**
  - Fonts: load `../deck/assets/fonts.css`. Use Baloo 2 for headings and big numbers, and the system UI font for data labels.
  - Palette: `--blue #0B7DC8`, `--blue-deep #06578F`, `--red #E5242B`, `--yellow #FFD54A`, `--ink #0B2545`.
  - The page will be projected, so no text smaller than about 24px at 1920 width.
- **Stage and keys.** Use the same 1920×1080 stage scaling as the deck. Keys: `1`–`4` presets, `Enter` run, `Esc` clear, `F` fullscreen.
- **Engine.** Build against `demo/engine.mock.js` first: the same API, with fake but plausible data and parts. When the coordinator says the engine is ready, switch to `demo/engine.js`.
- **States:**
  - model loading (show progress)
  - ready
  - running
  - error (fall back to `demo/data/presets.results.json` and say so on screen)
- **Motion.** One authored moment: the clauses travelling to their parts. Respect `prefers-reduced-motion`.
- **Footnote:** "Embedding model: all-MiniLM-L6-v2 · dots: real Stack Exchange questions."

## Running it

Run `./serve.sh`, then open http://localhost:8000/demo/. The model files need http; they won't load from `file://`.

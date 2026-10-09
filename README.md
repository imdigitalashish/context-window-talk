# AI is like a girlfriend that ghosts you in your context window

A 25-minute talk by Ashish Kumar Verma, 10 Oct 2026, with its slide deck and live demos.

**The goal:** by the end, everyone can predict whether their AI will fail before they hit Enter, and knows the fix for each reason it fails.

## The three laws

| Law | Concepts under it |
|---|---|
| Dumb zone | Context rot as the window fills; self-conditioning, where errors in the context breed more errors |
| Clustering | Prompts pull "parts" of the model's knowledge; negation blindness; instruction overload; literal matching |
| U-shaped attention | Lost in the middle; attention sinks |

## Layout

| Path | What |
|---|---|
| `deck/` | Slide deck: one self-contained HTML file with fonts embedded, so it works from `file://`. Use →/Space to advance, ← to go back, F for fullscreen. |
| `demo/` | Live demos. They need `./serve.sh`, because the model files can't load from `file://`. |
| `docs/demo-spec.md` | Spec for the prompt-footprint demo (clustering law) |
| `notes/last-talk-transcript.txt` | Transcript of the previous talk, "Why your AI is like a Ghajini?" |

## Run

```sh
./serve.sh   # deck: http://localhost:8000/deck/   demo: http://localhost:8000/demo/
```

"""Recall by position: hide a needle at 11 depths in Gutenberg text, ask about it at the end, score exact match.

    nice -n 15 ../.venv/bin/python recall.py --cond lookalikes --mode after --lengths 8192 2048

Writes one JSON line per trial to .cache/runs/<cond>-<mode>-<length>.jsonl (resumable); aggregate.py turns
those into data/recall.json.
"""
import argparse
import gc
import json
import re
import subprocess
import time

import mlx.core as mx

from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--cond", choices=["single", "lookalikes"], required=True)
ap.add_argument("--mode", choices=["after", "both"], default="after")
ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 2048])
ap.add_argument("--trials", type=int, default=len(NEEDLES))
ap.add_argument("--min-free", type=int, default=25, help="pause while system free memory %% is below this")
args = ap.parse_args()


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else 100


def wait_for_memory():
    while (f := free_pct()) < args.min_free:
        print(f"  memory free {f}% < {args.min_free}%: waiting 30 s", flush=True)
        time.sleep(30)


model, tok = load_model()
EOS = eos_ids(tok)
runs = CACHE / "runs"
runs.mkdir(exist_ok=True)
t_all = time.time()

for L in args.lengths:
    path = runs / f"{args.cond}-{args.mode}-{L}.jsonl"
    done = set()
    if path.exists():
        done = {(r["trial"], r["depth"]) for r in map(json.loads, path.read_text().splitlines())}
    t_len = time.time()
    with path.open("a") as fh:
        for i in range(args.trials):
            if all((i, d) in done for d in DEPTHS):
                continue
            wait_for_memory()
            book_id, title = BOOKS[i]
            p = Prompt(tok, book_id, i, L, mode=args.mode, lookalikes=args.cond == "lookalikes")
            seqs = [p.ids(d) for d in DEPTHS]
            cache, cached = make_prompt_cache(model), []
            t_trial, row = time.time(), []
            for j, d in enumerate(DEPTHS):
                if (i, d) in done:
                    continue
                ids = seqs[j]
                keep = lcp(cached, ids)  # the text before the needle is shared with the previous depth
                trim_to(cache, keep)
                t0 = time.time()
                reply = tok.decode(greedy(model, cache, prefill(model, cache, ids[keep:], want_logits=True), EOS))
                keep = lcp(ids, seqs[j + 1]) if j + 1 < len(seqs) else 0
                trim_to(cache, keep)
                cached = ids[:keep]
                ok = is_correct(reply, p.answer)
                row.append("✓" if ok else "·")
                fh.write(json.dumps({
                    "trial": i, "depth": d, "depth_actual": p.depth_actual(d), "book": title, "book_id": book_id,
                    "answer": p.answer, "question": p.question, "reply": reply, "correct": ok,
                    "prompt_tokens": len(ids), "lookalike_pos": p.lookalike_pos, "seconds": round(time.time() - t0, 2),
                }, ensure_ascii=False) + "\n")
                fh.flush()
            del cache
            gc.collect()
            mx.clear_cache()
            print(f"{args.cond}/{args.mode} L={L} trial {i} {''.join(row)}  {time.time() - t_trial:5.1f}s "
                  f"peak {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)
    gc.collect()
    mx.clear_cache()
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    curve = [sum(r["correct"] for r in rows if r["depth"] == d) for d in DEPTHS]
    print(f"== {args.cond}/{args.mode} L={L}: correct by depth {curve} of {args.trials}  ({time.time() - t_len:.0f}s)", flush=True)

print(f"total {time.time() - t_all:.0f}s", flush=True)

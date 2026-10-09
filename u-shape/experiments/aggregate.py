"""Collect .cache/runs/*.jsonl into data/recall.json (no model needed)."""
import datetime as dt
import json
import re

from common import *

LABELS = {
    "single-after": "One needle, question after the document",
    "lookalikes-after": "Needle + 9 look-alike facts, question after the document",
    "lookalikes-both": "Needle + 9 look-alike facts, question before and after the document (the fix)",
    "single-both": "One needle, question before and after the document",
}


def said_value(reply, i):
    """The value part of a reply: strips the needle's own wording ("The Wi-Fi password at the venue is ")."""
    stmt, _, ans, _ = needle_spec(i)
    stem = stmt[: stmt.index(ans)].lower()
    r = re.sub(r"\s+", " ", reply).strip()
    for v in lookalike_values(i):
        if v.lower() in r.lower():
            return v
    if r.lower().startswith(stem):
        r = r[len(stem):]
    return r.rstrip(".").strip()


def miss_kind(reply, i, answer):
    r = reply.lower()
    vals = [v.lower() for v in lookalike_values(i)]
    if any(v in r for v in vals):
        return "lookalike"  # answered with a look-alike fact's value
    if re.search(r"not (mentioned|provided|found|specified|stated|given|available)|no (information|mention)|does not|doesn't|cannot|unknown", r):
        return "not_found"
    return "garbled"  # e.g. a blend of the target and a look-alike


def main():
    out = {
        "what": "Recall by needle position. One fact (the needle) is hidden at 11 depths in Project Gutenberg text; "
                "the question comes at the end. Each point = 10 needles, each in its own book.",
        "model": MODEL_NAME,
        "model_short": "Qwen2.5-3B-Instruct",
        "model_source": "Qwen/Qwen2.5-3B-Instruct from the local Hugging Face cache, converted with "
                        "mlx_lm.convert -q --q-bits 4 --q-group-size 64 (4.5 bits/weight)",
        "decoding": "greedy (argmax), at most 24 new tokens, stops at <|im_end|>",
        "scoring": "exact match: correct if the answer string appears in the reply (case-insensitive), as in Liu et al. 2023",
        "prompt": {"system": "Qwen2.5 default system prompt", "instruction": INSTRUCTION, "answer_rule": ANSWER_RULE},
        "depths": DEPTHS,
        "needles": [{"statement": needle_spec(i)[0], "question": needle_spec(i)[1], "answer": needle_spec(i)[2],
                     "lookalikes": needle_spec(i)[3], "book": BOOKS[i][1], "gutenberg_id": BOOKS[i][0]}
                    for i in range(len(NEEDLES))],
        "conditions": {},
        "env": env_info(),
    }
    total_s = 0.0
    for path in sorted((CACHE / "runs").glob("*.jsonl")):
        cond, mode, length = path.stem.rsplit("-", 2)
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        grid = {(r["trial"], r["depth"]): r for r in rows}
        trials = sorted(t for t in {r["trial"] for r in rows} if all((t, d) in grid for d in DEPTHS))
        if len(trials) < len(NEEDLES):
            print(f"incomplete: {path.name} has {len(trials)} complete trials; skipping it")
            continue
        total_s += sum(r["seconds"] for r in rows)
        key = f"{cond}-{mode}"
        c = out["conditions"].setdefault(key, {"label": LABELS.get(key, key), "by_length": {}})
        pts = []
        for d in DEPTHS:
            rs = [grid[(t, d)] for t in trials if (t, d) in grid]
            misses = [{"trial": r["trial"], "reply": r["reply"], "kind": miss_kind(r["reply"], r["trial"], r["answer"]),
                       "said": said_value(r["reply"], r["trial"]), "answer": r["answer"]}
                      for r in rs if not r["correct"]]
            pts.append({"depth": d, "correct": sum(r["correct"] for r in rs), "n": len(rs),
                        "depth_actual_mean": round(sum(r["depth_actual"] for r in rs) / len(rs), 4), "misses": misses})
        toks = [r["prompt_tokens"] for r in rows]
        c["by_length"][length] = {
            "points": pts,
            "prompt_tokens": [min(toks), max(toks)],
            "trials": [[grid[(t, d)]["correct"] for d in DEPTHS] for t in trials],
            "lookalike_pos": [grid[(t, DEPTHS[0])]["lookalike_pos"] for t in trials],
            "run_finished": dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "model_seconds": round(sum(r["seconds"] for r in rows), 1),
        }
        print(f"{key:18s} L={length:>5s} " + " ".join(f"{p['correct']:2d}" for p in pts) + f"   (n={pts[0]['n']})")
    out["model_seconds_total"] = round(total_s, 1)
    print("saved", save_json("recall.json", out))


if __name__ == "__main__":
    main()

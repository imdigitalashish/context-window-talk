"""Self-conditioning, measured locally: a small replication of the running-sum task
from Sinha et al. 2025, "The Illusion of Diminishing Returns" (arXiv:2509.09677).

Task: a dictionary maps 100 common five-letter words to integers in [-99, 99].
Each turn gives K keys; the model must add their values to a running sum and reply
<answer>N</answer>. Turn accuracy follows the paper: a turn is correct when the state
update from turn t-1 to t is correct, whatever the state at t-1 was.

Conditions (same dictionaries and keys in each rollout):
  clean     same chat, no injected mistake; the model's own replies stay in the history
  mistake   same chat; turn 2's reply in the history is a wrong sum we injected
  fresh     every turn is a new chat holding only a one-line summary: "running sum so far: S"

Run (from the repo root):
  dumb-zone/.venv/bin/python dumb-zone/experiment/self_conditioning.py --pilot
  dumb-zone/.venv/bin/python dumb-zone/experiment/self_conditioning.py
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MODEL = ROOT / ".cache" / "Qwen2.5-3B-Instruct-4bit"

WORDS = """apple beach brain bread brush chair chest chord click clock cloud coast couch
crane cream crowd dance diary dream drink earth field flame floor flute frame fruit
ghost glass globe grape grass guard heart horse hotel house image juice knife lemon
light lunch magic metal money month mouse music nurse ocean olive onion paint paper
party peach pearl phone piano pilot plane plant plate press queen radio river robot
salad scale scarf sheep shelf shirt shoes skirt smile snake space spoon sport stage
stone storm sugar table tiger toast tooth tower train truck uncle voice water whale
bench cabin wheel""".split()
assert len(WORDS) == 100 and len(set(WORDS)) == 100

SYSTEM = """You are a careful calculator. Here is a dictionary that maps words to integers:

{dictionary}

The running sum starts at 0. In each turn I will give you {k} word(s). Look up the value of each word, add the values to the running sum, and reply with ONLY the new running sum as a single integer inside answer tags. Do not write the calculation.

Example with a different dictionary (cat: 5, dog: -3). Running sum 10, turn "cat, dog": 10 + 5 + (-3) = 12, so you reply <answer>12</answer>."""

FRESH_SUMMARY = "Summary of our chat so far: the running sum is {state}."


def make_rollout(seed, turns, k, n_words):
    rng = random.Random(seed)
    vocab = rng.sample(WORDS, n_words)
    values = {w: rng.randint(-99, 99) for w in vocab}
    keys = [[rng.choice(vocab) for _ in range(k)] for _ in range(turns)]
    offset = rng.choice([v for v in range(-60, 61) if abs(v) >= 15])
    return values, keys, offset


def system_prompt(values, k):
    dictionary = "\n".join(f"{w}: {v}" for w, v in values.items())
    return SYSTEM.format(dictionary=dictionary, k=k)


def user_turn(keys):
    return ", ".join(keys)


def parse(text):
    m = re.search(r"<answer>\s*([-+]?\d+)\s*</answer>", text)
    if m:
        return int(m.group(1)), True
    nums = re.findall(r"[-+]?\d+", text)
    return (int(nums[-1]), False) if nums else (None, False)


class Model:
    def __init__(self, path):
        self.model, self.tok = load(str(path))
        self.sampler = make_sampler(temp=0.0)
        self.calls = 0
        self.prompt_tokens = 0

    def reply(self, messages):
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.calls += 1
        self.prompt_tokens += len(self.tok.encode(prompt))
        return generate(self.model, self.tok, prompt=prompt, max_tokens=24, sampler=self.sampler, verbose=False).strip()


def run_same_chat(m, values, keys, offset, inject):
    """inject=False: the model's own replies form the history from turn 1.
    inject=True: turn 1 is a correct reply, turn 2 a wrong one we inject, then the model takes over."""
    msgs = [{"role": "system", "content": system_prompt(values, len(keys[0]))}]
    state, cells = 0, []
    for t, ks in enumerate(keys, start=1):
        msgs.append({"role": "user", "content": user_turn(ks)})
        delta = sum(values[x] for x in ks)
        if inject and t <= 2:
            shown = state + delta + (offset if t == 2 else 0)
            msgs.append({"role": "assistant", "content": f"<answer>{shown}</answer>"})
            cells.append({"t": t, "injected": True, "correct": t != 2})
            state = shown
            continue
        out = m.reply(msgs)
        got, fmt = parse(out)
        ok = got is not None and got - state == delta
        cells.append({"t": t, "correct": ok, "format_ok": fmt, "reply": out[:40]})
        msgs.append({"role": "assistant", "content": out})
        if got is not None:
            state = got
    return cells


def run_fresh(m, values, keys):
    sys_msg = {"role": "system", "content": system_prompt(values, len(keys[0]))}
    state, cells = 0, []
    for t, ks in enumerate(keys, start=1):
        msgs = [sys_msg, {"role": "user", "content": FRESH_SUMMARY.format(state=state) + "\n\n" + user_turn(ks)}]
        out = m.reply(msgs)
        got, fmt = parse(out)
        delta = sum(values[x] for x in ks)
        ok = got is not None and got - state == delta
        cells.append({"t": t, "correct": ok, "format_ok": fmt, "reply": out[:40]})
        if got is not None:
            state = got
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--words", type=int, default=100, help="dictionary size (the paper uses 100)")
    ap.add_argument("--seed", type=int, default=2509)
    ap.add_argument("--pilot", action="store_true", help="1 rollout, 12 turns, print and exit")
    ap.add_argument("--out", default=str(ROOT / "data" / "self-conditioning.json"))
    a = ap.parse_args()
    if a.pilot:
        a.rollouts, a.turns = 1, 16

    mx.set_cache_limit(256 * 1024 * 1024)
    t0 = time.time()
    m = Model(a.model)
    runs = []
    for r in range(a.rollouts):
        values, keys, offset = make_rollout(a.seed + r, a.turns, a.k, a.words)
        row = {"rollout": r, "seed": a.seed + r, "injected_offset": offset,
               "clean": run_same_chat(m, values, keys, offset, inject=False),
               "mistake": run_same_chat(m, values, keys, offset, inject=True),
               "fresh": run_fresh(m, values, keys)}
        runs.append(row)
        acc = {c: sum(x["correct"] for x in row[c] if not x.get("injected")) for c in ("clean", "mistake", "fresh")}
        print(f"rollout {r}: correct turns {acc}  ({time.time() - t0:.0f}s)", flush=True)
        if a.pilot:
            for c in ("clean", "mistake", "fresh"):
                print(c, "".join("i" if x.get("injected") else ("." if x["correct"] else "X") for x in row[c]),
                      [x.get("reply") for x in row[c]][:6])

    result = {
        "what": "Turn accuracy on the running-sum task of Sinha et al. 2025 (arXiv:2509.09677), measured locally",
        "model": "Qwen2.5-3B-Instruct, MLX 4-bit (converted locally with mlx_lm.convert -q)",
        "decoding": "greedy (temperature 0), max 24 new tokens",
        "dictionary": f"{a.words} five-letter words -> integers uniform in [-99, 99], resampled per rollout",
        "keys_per_turn": a.k, "turns": a.turns, "rollouts": a.rollouts, "seed": a.seed,
        "injected_mistake": "same-chat 'mistake' condition: turn 1 reply correct, turn 2 reply = true sum + offset (|offset| 15-60), both injected; model replies from turn 3",
        "metric": "a turn is correct when (reply - previous state in context) equals the sum of this turn's values",
        "seconds": round(time.time() - t0, 1),
        "model_calls": m.calls,
        "prompt_tokens_total": m.prompt_tokens,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "mlx_lm": __import__("mlx_lm").__version__,
        "runs": runs,
    }
    if not a.pilot:
        Path(a.out).write_text(json.dumps(result, indent=1))
        print("wrote", a.out)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

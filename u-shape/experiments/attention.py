"""Attention by position: how much attention the final question tokens pay to each earlier position.

Runs on the small model (Qwen2.5-0.5B-Instruct, MLX 4-bit) at 2K tokens. The prompt minus its question is
prefilled normally; then the question tokens go through as one chunk with mlx-lm's attention hooked, computing
softmax(QK^T) for just those query rows against every key (never a full T x T matrix). Probabilities are
averaged over query tokens, heads and layers, then bucketed into 60 position bins; token 0 is reported alone.

    nice -n 15 ../.venv/bin/python attention.py
"""
import gc
import glob
import os
import time

import mlx.core as mx
import mlx_lm.models.qwen2 as qwen2
import numpy as np
from mlx_lm import load

from common import *

SMALL_DIR = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/*"))[0]
SMALL_NAME = "Qwen2.5-0.5B-Instruct (mlx-community, MLX 4-bit)"
L = 2048
BINS = 60
REP = 0  # representative trial: needle 0 (Wi-Fi password) in book 0
CAP = {"on": False, "layers": []}
_orig = qwen2.scaled_dot_product_attention


def _hooked(queries, keys, values, cache, scale, mask, **kw):
    if CAP["on"]:
        B, H, Q, D = queries.shape
        KV, T = keys.shape[1], keys.shape[2]
        q = queries.astype(mx.float32).reshape(B, KV, H // KV, Q, D)
        k = keys.astype(mx.float32)[:, :, None]
        s = (q * scale) @ k.swapaxes(-1, -2)  # [B, KV, rep, Q, T]
        causal = mx.arange(T)[None, :] <= mx.arange(T - Q, T)[:, None]
        p = mx.softmax(mx.where(causal, s, -mx.inf), axis=-1).reshape(H, Q, T)
        CAP["layers"].append(mx.stack([p.mean(axis=1), p[:, -1, :]]))  # [2, H, T]: mean over question, last token
        mx.eval(CAP["layers"][-1])
    return _orig(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)


qwen2.scaled_dot_product_attention = _hooked


def measure(model, tok, p, depth):
    ids = p.ids(depth)
    text = p.text(depth)
    n_ctx = len(encode(tok, text[: text.rindex("Question:")]))
    assert lcp(ids, encode(tok, text[: text.rindex("Question:")])) >= n_ctx - 1
    cache = make_prompt_cache(model)
    prefill(model, cache, ids[:n_ctx])
    CAP["on"], CAP["layers"] = True, []
    prefill(model, cache, ids[n_ctx:], want_logits=True)
    CAP["on"] = False
    att = np.array(mx.stack(CAP["layers"]))  # [layers, 2, H, T]
    del cache
    # where the document and needle sit, in token positions
    doc_start = len(encode(tok, p.pre))
    doc_end = len(encode(tok, text[: text.rindex("\n</document>")]))
    needle = None
    if depth is not None:
        at = text.index(p.statement)
        n0 = len(encode(tok, text[:at]))
        needle = [n0, n0 + len(encode(tok, " " + p.statement))]
    return ids, att, n_ctx, doc_start, doc_end, needle


def summarise(att_q, T, n_ctx, doc_start, doc_end, needle):
    """att_q: [layers, heads, T] attention from the question tokens.

    bins: 60 equal-width bins over the document tokens. The prompt before the document (token 0 = the sink,
    then the rest of the template + instruction) and the question after it are reported separately.
    """
    prof = att_q.mean(axis=(0, 1))  # mean over layers and heads -> sums to 1 over positions
    edges = np.linspace(doc_start, doc_end, BINS + 1).round().astype(int)
    bins = [float(prof[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
    doc = prof[doc_start:doc_end]
    n = len(doc)
    q_edges = np.linspace(doc_end, T, 5).round().astype(int)
    out = {
        "tokens": int(T),
        "sink": float(prof[0]),
        "sink_by_layer": [round(float(x), 4) for x in att_q[:, :, 0].mean(axis=1)],
        "prefix_rest_per_token": float(prof[1:doc_start].mean()),
        "bin_edges": edges.tolist(),
        "bins": bins,
        "tail_bins": [float(prof[a:b].mean()) for a, b in zip(q_edges[:-1], q_edges[1:])],
        "tail_edges": q_edges.tolist(),
        "uniform": 1.0 / T,
        "share": {
            "token0": float(prof[0]),
            "prompt_before_doc": float(prof[1:doc_start].sum()),
            "doc_first10": float(doc[: n // 10].sum()),
            "doc_middle80": float(doc[n // 10: n - n // 10].sum()),
            "doc_last10": float(doc[n - n // 10:].sum()),
            "after_doc": float(prof[doc_end:].sum()),
        },
        "per_token": {
            "doc_first10": float(doc[: n // 10].mean()),
            "doc_middle80": float(doc[n // 10: n - n // 10].mean()),
            "doc_last10": float(doc[n - n // 10:].mean()),
            "after_doc": float(prof[doc_end:].mean()),
        },
        "doc_span": [int(doc_start), int(doc_end)],
        "question_span": [int(n_ctx), int(T)],
    }
    if needle:
        a, b = needle
        out["needle_span"] = needle
        out["needle_mass"] = float(prof[a:b].sum())
        out["needle_per_token_vs_doc"] = float(prof[a:b].mean() / doc.mean())
    return out


def main():
    mx.set_cache_limit(1 * 2**30)
    model, tok = load(SMALL_DIR)
    t0 = time.time()
    result = {
        "what": "Attention from the final question tokens to every earlier position, averaged over the question "
                "tokens, all heads and all layers. bins = mean attention per token in 60 equal-width position bins "
                "over tokens 1..T-1 (token 0, the attention sink, reported separately as 'sink').",
        "model": SMALL_NAME, "model_short": "Qwen2.5-0.5B-Instruct", "model_path": SMALL_DIR.split("hub/")[-1], "context_tokens_target": L,
        "layers": model.args.num_hidden_layers, "heads": model.args.num_attention_heads,
        "kv_heads": model.args.num_key_value_heads, "bins": BINS, "env": env_info(),
        "representative": {"trial": REP, "book": BOOKS[REP][1], "book_id": BOOKS[REP][0],
                           "needle": needle_spec(REP)[0], "question": needle_spec(REP)[1]},
        "runs": {},
    }
    for cond, mode in [("single", "after"), ("lookalikes", "after"), ("lookalikes", "both"), ("single", "both")]:
        key = f"{cond}-{mode}"
        p = Prompt(tok, BOOKS[REP][0], REP, L, mode=mode, lookalikes=cond == "lookalikes")
        run = {"lookalike_pos": p.lookalike_pos, "no_needle": None, "by_depth": [], "last_token_no_needle": None}
        for depth in [None] + DEPTHS:
            ids, att, n_ctx, ds, de, needle = measure(model, tok, p, depth)
            s = summarise(att[:, 0], len(ids), n_ctx, ds, de, needle)
            if depth is None:
                run["no_needle"] = s
                run["last_token_no_needle"] = summarise(att[:, 1], len(ids), n_ctx, ds, de, None)
                run["token0"] = tok.convert_ids_to_tokens(ids[0])
            else:
                s["depth"], s["depth_actual"] = depth, p.depth_actual(depth)
                run["by_depth"].append(s)
            gc.collect()
        # robustness: the no-needle profile averaged over all 10 haystacks
        if mode == "after":
            profs = []
            for i in range(len(BOOKS)):
                q = Prompt(tok, BOOKS[i][0], i, L, mode=mode, lookalikes=cond == "lookalikes")
                ids, att, n_ctx, ds, de, _ = measure(model, tok, q, None)
                profs.append(summarise(att[:, 0], len(ids), n_ctx, ds, de, None))
            run["no_needle_10_haystacks"] = {
                "sink_mean": float(np.mean([x["sink"] for x in profs])),
                "sink_min": float(np.min([x["sink"] for x in profs])),
                "sink_max": float(np.max([x["sink"] for x in profs])),
                "bins_mean": np.mean([x["bins"] for x in profs], axis=0).tolist(),
                "per_token_mean": {k: float(np.mean([x["per_token"][k] for x in profs])) for k in profs[0]["per_token"]},
                "share_mean": {k: float(np.mean([x["share"][k] for x in profs])) for k in profs[0]["share"]},
            }
        result["runs"][key] = run
        nn = run["no_needle"]
        print(f"{key}: T={nn['tokens']} token0={run['token0']!r} sink={nn['sink']:.3f} per-token first10/mid80/last10 = "
              f"{nn['per_token']['doc_first10']:.2e}/{nn['per_token']['doc_middle80']:.2e}/{nn['per_token']['doc_last10']:.2e} "
              f"after-doc share={nn['share']['after_doc']:.3f}", flush=True)
        print("   needle mass by depth:", [round(x["needle_mass"], 4) for x in run["by_depth"]], flush=True)
    result["seconds"] = round(time.time() - t0, 1)
    print("saved", save_json("attention.json", result), f"{result['seconds']}s, peak {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()

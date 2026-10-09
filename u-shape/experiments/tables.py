"""Print the README's Results section (markdown), straight from data/*.json. No model needed."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
rec = json.loads((DATA / "recall.json").read_text())
att = json.loads((DATA / "attention.json").read_text())
LABEL = {
    "lookalikes-after": "Needle + 9 look-alikes, question after the document",
    "lookalikes-both": "Needle + 9 look-alikes, question before and after (the fix)",
    "single-after": "Lone needle, question after the document",
    "single-both": "Lone needle, question before and after",
}
depths = rec["depths"]
head = " | ".join(f"{round(d * 100)}%" for d in depths)

print(f"**Recall by needle depth.** {rec['model_short']}, MLX 4-bit. Each cell is the number of the 10 documents "
      "whose needle was recalled (exact match).\n")
print(f"| Condition | Context | {head} |")
print("|---|---|" + "---|" * len(depths))
for key in LABEL:
    c = rec["conditions"].get(key)
    for L in sorted(c["by_length"], key=int) if c else []:
        b = c["by_length"][L]
        cells = " | ".join(f"{p['correct']}/{p['n']}" for p in b["points"])
        print(f"| {LABEL[key]} | {round(int(L) / 1024)}K ({b['prompt_tokens'][0]:,}–{b['prompt_tokens'][1]:,} tokens) | {cells} |")

print(f"\n**Where the question's attention goes.** {att['model_short']}, MLX 4-bit, document 1 at "
      f"{att['context_tokens_target'] // 1024}K tokens with no needle in it, averaged over the question tokens, "
      f"{att['heads']} heads and {att['layers']} layers. Token 0 is given as a share of all attention; the rest is "
      "attention per token, in multiples of an even share (1/T).\n")
print("| Prompt | Token 0 (sink) | Rest of template + instruction | Document: first 10% | Middle 80% | Last 10% | After the document (question) |")
print("|---|---|---|---|---|---|---|")
for key in ["lookalikes-after", "single-after", "lookalikes-both"]:
    r = att["runs"].get(key)
    if not r:
        continue
    n = r["no_needle"]
    u, pt = n["uniform"], n["per_token"]
    print(f"| {LABEL[key]} | {n['sink'] * 100:.1f}% ({n['sink'] / u:,.0f}× even) | {n['prefix_rest_per_token'] / u:.1f}× | "
          f"{pt['doc_first10'] / u:.2f}× | {pt['doc_middle80'] / u:.2f}× | {pt['doc_last10'] / u:.2f}× | {pt['after_doc'] / u:.1f}× |")
for key in ["lookalikes-after", "single-after"]:
    h = att["runs"].get(key, {}).get("no_needle_10_haystacks")
    if h:
        u = att["runs"][key]["no_needle"]["uniform"]
        pm = h["per_token_mean"]
        print(f"\nAcross all 10 haystacks ({LABEL[key].lower()}): token 0 got {h['sink_min'] * 100:.1f}–{h['sink_max'] * 100:.1f}% "
              f"(mean {h['sink_mean'] * 100:.1f}%). Per token, the first 10% of the document got {pm['doc_first10'] / u:.2f}×, "
              f"the middle 80% {pm['doc_middle80'] / u:.2f}× and the last 10% {pm['doc_last10'] / u:.2f}× an even share.")

print("\n**Attention on the needle itself**, per needle token, in multiples of an even share, by needle depth "
      "(same model and document):\n")
print(f"| Prompt | {head} |")
print("|---|" + "---|" * len(depths))
for key in ["lookalikes-after", "lookalikes-both", "single-after"]:
    r = att["runs"].get(key)
    if not r:
        continue
    vals = [d["needle_mass"] / (d["needle_span"][1] - d["needle_span"][0]) / d["uniform"] for d in r["by_depth"]]
    print(f"| {LABEL[key]} | " + " | ".join(f"{v:.2f}×" for v in vals) + " |")

import json, re, types, glob as _glob, sys, random
from pathlib import Path
R = Path("/Users/imdigitalashish/Projects/AshishKumarVerm/context-window-talk/kaggle")
nb = json.loads((R / "dumb-zone/dumb-zone-measured.ipynb").read_text())
cells = {c["source"].split("\n")[0]: c["source"] for c in nb["cells"] if c["cell_type"] == "code"}
from transformers import AutoTokenizer
snap = _glob.glob("/Users/imdigitalashish/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/")[0]
tok = AutoTokenizer.from_pretrained(snap)
fake_glob = types.SimpleNamespace(glob=lambda p, recursive=False: _glob.glob(p.replace("/kaggle/input/**", str(R / "haystack"))))
fake_torch = types.SimpleNamespace(no_grad=lambda: (lambda f: f))
ns = dict(tok=tok, glob=fake_glob, torch=fake_torch, re=re, random=random, json=json)
for key in cells:
    if key.startswith(("# The haystack", "# The facts we hide", "# Build a prompt")):
        print(f"--- exec: {key}"); exec(cells[key], ns)
TOKS, BOOKS, WORDS, MEANING = ns["TOKS"], ns["BOOKS"], ns["WORDS"], ns["MEANING"]
haystack, insert, prompt, right = ns["haystack"], ns["insert"], ns["prompt"], ns["right"]
assert min(len(TOKS[n]) for n in BOOKS) > 33000, "a book is too short"
rng = random.Random(7); DEPTH = [round(rng.uniform(0.2, 0.8), 3) for _ in BOOKS]
print("depths:", DEPTH)
worst = 0
for L in [1024, 2048, 4096, 8192, 16384, 31744]:
    ns_tokens = []
    for kind, needles in (("words", WORDS), ("meaning", MEANING)):
        for i, nd in enumerate(needles):
            doc = insert(haystack(BOOKS[i], L), nd["fact"], DEPTH[i])
            p = prompt(doc, nd)
            n = len(tok(p, add_special_tokens=False).input_ids)
            ns_tokens.append(n); worst = max(worst, n)
            assert nd["fact"] in p and p.count(nd["q"]) == 1
            actual = doc.find(nd["fact"]) / len(doc)
            assert abs(actual - DEPTH[i]) < (0.1 if L < 4096 else 0.03), (L, i, actual, DEPTH[i])
    print(f"L={L:>6}: prompt tokens {min(ns_tokens)}..{max(ns_tokens)}")
print("max prompt tokens:", worst, "(limit 32768)")
assert worst < 32768 - 32
# U-shape: depth 0 and 1 placement, 'before' variant, attention offsets
for d in (0.0, 0.5, 1.0):
    doc = insert(haystack(BOOKS[1], 4096), MEANING[1]["fact"], d)
    print(f"depth {d}: fact at {doc.find(MEANING[1]['fact']) / len(doc):.3f}")
p = prompt(insert(haystack(BOOKS[0], 2048), MEANING[0]["fact"], 0.5), MEANING[0], before=True)
assert p.count(MEANING[0]["q"]) == 2
text = prompt(haystack(BOOKS[0], 8192), MEANING[0])
enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
offs = enc.offset_mapping
q0 = next(i for i, (a, b) in enumerate(offs) if a >= text.rfind("Question:"))
d0 = next(i for i, (a, b) in enumerate(offs) if a >= text.find("<story>") + 8)
d1 = next(i for i, (a, b) in enumerate(offs) if a >= text.find("</story>"))
print("attention split: total", len(enc.input_ids), "doc", d0, "→", d1, "question from", q0, "→", repr(tok.decode(enc.input_ids[q0:])))
print("first tokens:", repr(tok.decode(enc.input_ids[:3])), "| doc starts:", repr(tok.decode(enc.input_ids[d0:d0+8])))
for reply, ans, exp in [("Kabir", "Kabir", True), ("The character is Kabir.", "Kabir", True), ("Kabirs", "Kabir", False),
                        ("blue-dorayaki-42", "blue-dorayaki-42", True), ("blue-dorayaki-421", "blue-dorayaki-42", False),
                        ("9:40 pm", "9:40", True), ("Room 4C-17", "4C-17", True), ("Elizabeth", "Tara", False)]:
    assert right(reply, ans) == exp, (reply, ans)
print("scoring ok")
print(text[:420])

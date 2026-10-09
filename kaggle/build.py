"""Builds the Kaggle notebooks for the talk.

Run:  python kaggle/build.py [machine_shape]
Writes kaggle/{dumb-zone,u-shape,smoke}/<slug>.ipynb plus kernel-metadata.json, and
kaggle/haystack/ (the Project Gutenberg novels, as a private Kaggle dataset).
"""
import json
import shutil
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
USER = "imdigitalashish"
MODEL_SOURCE = "qwen-lm/qwen2.5/transformers/7b-instruct/1"
DATASET = f"{USER}/gutenberg-novels-haystack"
BOOKS = [1342, 1661, 84, 2701, 98, 35, 345, 1400, 36, 120]

LITERAL = [
    {k: n[k] for k in ("statement", "question", "answer")}
    for n in json.loads((REPO / "u-shape/data/recall.json").read_text())["needles"]
]
MEANING = [
    ("Kabir's flat looks straight out at the Charminar.", "Hyderabad", "Kabir"),
    ("Ananya got married in a hall right beside the Hawa Mahal.", "Jaipur", "Ananya"),
    ("Rohan runs a flower stall at the east gate of the Meenakshi Temple.", "Madurai", "Rohan"),
    ("Tara's office window faces the Burj Khalifa.", "Dubai", "Tara"),
    ("Arjun proposed to his wife on the rooftop of the Marina Bay Sands.", "Singapore", "Arjun"),
    ("Zoya grew up a short walk from the Golden Temple.", "Amritsar", "Zoya"),
    ("Vikram jogs around the Qutub Minar every morning.", "Delhi", "Vikram"),
    ("Meera lives right next to the Gateway of India.", "Mumbai", "Meera"),
    ("Ishaan sketches the domes of the Hagia Sophia every weekend.", "Istanbul", "Ishaan"),
    ("Neha hikes up Table Mountain before work.", "Cape Town", "Neha"),
]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n")}


def code(*parts):
    src = "\n\n".join(textwrap.dedent(p).strip("\n") for p in parts)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


# ── Cells shared by every notebook ──────────────────────────────────────────

SETUP = code("""
    # Setup: libraries and the GPU
    import os, re, glob, json, time, random, datetime, warnings
    import numpy as np, pandas as pd, torch
    import matplotlib.pyplot as plt
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    warnings.filterwarnings("ignore")

    GPU = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    print("GPU:", GPU or "none", "| torch", torch.__version__, "| transformers", transformers.__version__)
    assert GPU, "Turn on a GPU: Settings → Accelerator"
    TODAY = datetime.date.today().isoformat()
""")

STYLE = code("""
    # Plot style: dark, big, readable from the back of a hall
    BG, FG, MUTED, GRID = "#0f0f10", "#f5f5f2", "#9a9aa0", "#2b2b30"
    RED, BLUE, GREEN, AMBER = "#ff5a5f", "#4c8dff", "#3ecf8e", "#ffb020"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 16, "axes.titlesize": 24,
        "axes.titleweight": "bold", "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })
    LABEL = {1024: "1K", 2048: "2K", 4096: "4K", 8192: "8K", 16384: "16K", 31744: "32K"}

    def spread(ys, gap, lo=-np.inf, hi=np.inf):
        # Nudge label heights apart, keeping their order, so line labels never sit on top of each other
        ys = np.asarray(ys, float)
        order = np.argsort(ys, kind="stable")
        out = ys[order].copy()
        for _ in range(200):
            for k in range(1, len(out)):
                if out[k] - out[k - 1] < gap:
                    mid = (out[k] + out[k - 1]) / 2
                    out[k - 1], out[k] = mid - gap / 2, mid + gap / 2
        out += min(0, hi - out[-1]) + max(0, lo - out[0])
        res = np.empty_like(out)
        res[order] = out
        return res
""")

MODEL = code("""
    # The model: Qwen2.5-7B-Instruct, open weights, 32K-token window
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    found = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    found = [os.path.dirname(p) for p in found if "7b-instruct" in p.lower()]
    MODEL_PATH = found[0] if found else "Qwen/Qwen2.5-7B-Instruct"
    # bf16 needs Ampere or newer (L4); T4 only emulates it, so use fp16 there
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    if torch.cuda.get_device_capability(0)[0] < 8:
        # T4 has no flash kernel, and SDPA's memory-efficient kernel can't do grouped-query attention, so
        # by default a long prompt falls back to the math kernel and builds the full tokens × tokens matrix
        # (7 GB at 8K). Repeat the key/value heads instead, and never allow the math kernel.
        import transformers.integrations.sdpa_attention as sdpa_impl
        sdpa_impl.use_gqa_in_sdpa = lambda *args, **kwargs: False
        torch.backends.cuda.enable_math_sdp(False)
    v = tuple(int(x) for x in re.findall(r"\\d+", transformers.__version__)[:2])
    dtype_arg = {"dtype" if v >= (4, 56) else "torch_dtype": DTYPE}

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", attn_implementation="sdpa", **dtype_arg).eval()
    BASE, HEAD = model.get_decoder(), model.get_output_embeddings()
    STOP = {t for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                        tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(t, int) and t >= 0}
    print(f"{MODEL_NAME} · {DTYPE} · from {MODEL_PATH}")
""")

BOOKS_CELL = code(f"""
    # The haystack: ten public-domain novels from Project Gutenberg
    BOOKS = {BOOKS}
    TITLES = {{1342: "Pride and Prejudice", 1661: "The Adventures of Sherlock Holmes", 84: "Frankenstein",
              2701: "Moby Dick", 98: "A Tale of Two Cities", 35: "The Time Machine", 345: "Dracula",
              1400: "Great Expectations", 36: "The War of the Worlds", 120: "Treasure Island"}}

    def load_book(n):
        hit = glob.glob(f"/kaggle/input/**/pg{{n}}.txt", recursive=True)
        if hit:
            raw = open(hit[0], encoding="utf-8").read()
        else:
            import urllib.request
            url = f"https://www.gutenberg.org/cache/epub/{{n}}/pg{{n}}.txt"
            raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
        raw = raw.replace("\\r\\n", "\\n")
        s = re.search(r"\\*\\*\\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\\n]*\\n", raw)
        e = re.search(r"\\*\\*\\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", raw)
        body = raw[s.end() if s else 0: e.start() if e else None]
        body = re.sub(r"\\[Illustration[^\\]]*\\]", "", body)
        hits = re.finditer(r"\\n\\s*(?:CHAPTER|Chapter) (?:1|I|ONE)\\b", body[: len(body) // 6])
        # Skip table-of-contents entries: the real heading is followed by prose, not another heading
        ch = next((m for m in hits if not re.search(r"(?:CHAPTER|Chapter) ", body[m.end(): m.end() + 300])), None)
        body = body[ch.start():] if ch else body[len(body) // 50:]
        return re.sub(r"[ \\t]+", " ", re.sub(r"\\n{{3,}}", "\\n\\n", body)).strip()

    TEXT = {{n: load_book(n) for n in BOOKS}}
    TOKS = {{n: tok(TEXT[n], add_special_tokens=False).input_ids for n in BOOKS}}
    for n in BOOKS:
        print(f"{{TITLES[n]:<36}} {{len(TOKS[n]):>8,}} tokens")
""")

NEEDLES = code(
    "# The facts we hide (\"needles\"). Fact i always lives in book i.\n"
    "# Find the words: the question repeats the fact's own words.\n"
    "WORDS = [dict(fact=n[\"statement\"], q=n[\"question\"], a=n[\"answer\"],\n"
    "              rule=\"Answer with just the answer from the story.\") for n in "
    + json.dumps(LITERAL, ensure_ascii=False, indent=1) + "]\n\n"
    "# Find the meaning: no shared words. You have to know where the landmark is.\n"
    "MEANING = [dict(fact=f, q=f\"Which character has been to {city}?\", a=name,\n"
    "                rule=\"Answer with just the character's name.\") for f, city, name in "
    + json.dumps(MEANING, ensure_ascii=False, indent=1) + "]\n\n"
    "for w, m in zip(WORDS[:3], MEANING[:3]):\n"
    "    print(f\"words:   {w['q']:<55} ← {w['fact']}\")\n"
    "    print(f\"meaning: {m['q']:<55} ← {m['fact']}\\n\")"
)

PROMPTS = code("""
    # Build a prompt: the first N tokens of a novel, the fact at a chosen depth, the question at the end
    def haystack(book, n_tokens):
        text = tok.decode(TOKS[book][:n_tokens])
        cut = max(text.rfind(". "), text.rfind(".\\n"), text.rfind('." '))
        return text[: cut + 1] if cut > 0 else text

    def insert(text, fact, depth):
        spots = [0] + [m.end() for m in re.finditer(r'[.!?]["”’]?\\s+', text)] + [len(text)]
        at = min(spots, key=lambda p: abs(p - depth * len(text)))
        pad = " " if at and not text[at - 1].isspace() else ""
        return text[:at] + pad + fact + " " + text[at:]

    def prompt(doc, needle, before=False):
        q = f"Question: {needle['q']}\\n"
        user = ((q + "\\n") if before else "") + (
            "Read the story below and answer the question after it.\\n\\n"
            f"<story>\\n{doc}\\n</story>\\n\\n{q}{needle['rule']}")
        return tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def ask(text, max_new=24):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        out = BASE(input_ids=ids, use_cache=True)
        past, got = out.past_key_values, []
        nxt = HEAD(out.last_hidden_state[:, -1:]).argmax(-1)
        for _ in range(max_new):
            if nxt.item() in STOP:
                break
            got.append(nxt.item())
            out = BASE(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = HEAD(out.last_hidden_state[:, -1:]).argmax(-1)
        del out, past
        return tok.decode(got, skip_special_tokens=True).strip(), ids.shape[1]

    def right(reply, answer):
        return re.search(r"(?<![\\w-])" + re.escape(answer.lower()) + r"(?![\\w-])", reply.lower()) is not None

    example = prompt(insert(haystack(BOOKS[0], 300), MEANING[0]["fact"], 0.5), MEANING[0])
    print(example[example.find("<story>"):])
""")

SET_ATTN = """
    import transformers.models.qwen2.modeling_qwen2 as qwen2_impl

    def eager_fp32(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        # transformers' eager attention with the scores and softmax in float32. In float16 (T4) some of
        # Qwen's query·key products overflow, which zeroes or NaNs exactly the weights we want to read.
        n = module.num_key_value_groups
        k, v = key.repeat_interleave(n, dim=1), value.repeat_interleave(n, dim=1)
        w = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            m = attention_mask[..., : k.shape[-2]]
            w = w.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else w + m.float()
        w = torch.softmax(w, dim=-1)
        return torch.matmul(w.to(v.dtype), v).transpose(1, 2).contiguous(), w

    qwen2_impl.eager_attention_forward = eager_fp32

    def set_attn(impl):
        try:
            model.set_attn_implementation(impl)
        except Exception:
            model.config._attn_implementation = impl
"""


def metadata(slug, title, gpu):
    return {
        "id": f"{USER}/{slug}", "title": title, "code_file": f"{slug}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true", "machine_shape": gpu,
        "dataset_sources": [DATASET], "competition_sources": [],
        "kernel_sources": [], "model_sources": [MODEL_SOURCE],
    }


# ── Dumb zone ───────────────────────────────────────────────────────────────

DZ_RUN = code("""
    # Run: the same 20 questions at six prompt lengths (fact i keeps the same depth at every length)
    LENGTHS = [1024, 2048, 4096, 8192, 16384, 31744]
    rng = random.Random(7)
    DEPTH = [round(rng.uniform(0.2, 0.8), 3) for _ in BOOKS]

    rows = []
    for L in LENGTHS:
        t0 = time.time()
        for kind, needles in (("words", WORDS), ("meaning", MEANING)):
            for i, nd in enumerate(needles):
                doc = insert(haystack(BOOKS[i], L), nd["fact"], DEPTH[i])
                reply, n = ask(prompt(doc, nd))
                rows.append(dict(kind=kind, length=L, tokens=n, needle=i, depth=DEPTH[i],
                                 q=nd["q"], answer=nd["a"], reply=reply, ok=right(reply, nd["a"])))
        torch.cuda.empty_cache()
        done = [r for r in rows if r["length"] == L]
        w = np.mean([r["ok"] for r in done if r["kind"] == "words"]) * 100
        m = np.mean([r["ok"] for r in done if r["kind"] == "meaning"]) * 100
        print(f"{LABEL[L]:>4} tokens · words {w:5.0f}% · meaning {m:5.0f}% · {time.time() - t0:5.0f}s")

    df = pd.DataFrame(rows)
    df.to_json("dumb_zone_results.json", orient="records", indent=1)
""")

DZ_PLOT = code("""
    # The chart
    acc = df.groupby(["kind", "length"]).ok.mean().unstack(0) * 100
    xs = np.arange(len(LENGTHS))
    lines = (("words", BLUE, "Find the words"), ("meaning", RED, "Find the meaning"))
    starts = spread([acc[k].iloc[0] for k, _, _ in lines], 9, lo=0, hi=104)
    ends = spread([acc[k].iloc[-1] for k, _, _ in lines], 9, lo=0, hi=104)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    for (kind, color, name), y0, y1 in zip(lines, starts, ends):
        y = acc[kind].values
        ax.plot(xs, y, color=color, lw=5, marker="o", ms=13, zorder=3)
        ax.annotate(f"{y[0]:.0f}%", (xs[0], y0), xytext=(-18, 0), textcoords="offset points", ha="right",
                    va="center", color=color, fontsize=20, fontweight="bold", annotation_clip=False)
        ax.annotate(f"{y[-1]:.0f}%  {name}", (xs[-1], y1), xytext=(20, 0), textcoords="offset points",
                    va="center", color=color, fontsize=21, fontweight="bold", annotation_clip=False)
    ax.set_xticks(xs, [LABEL[L] for L in LENGTHS])
    ax.set_xlim(-0.8, len(xs) + 1.9)
    ax.set_ylim(-4, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", alpha=0.7)
    ax.set_xlabel("Tokens of novel around the fact")
    ax.set_ylabel("Answered correctly")
    ax.set_title("Same fact. Same question. Only more text around it.", loc="left", pad=22)
    fig.text(0.01, 0.01, f"{MODEL_NAME} · {GPU} · {df.needle.nunique()} facts per point, each in its own novel · "
             f"greedy decoding · measured {TODAY}", color=MUTED, fontsize=12)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig("dumb_zone.png", dpi=160)
    plt.show()

    w, m = acc["words"], acc["meaning"]
    print(f"Find the words:   {w.iloc[0]:.0f}% at {LABEL[LENGTHS[0]]} → {w.iloc[-1]:.0f}% at {LABEL[LENGTHS[-1]]}")
    print(f"Find the meaning: {m.iloc[0]:.0f}% at {LABEL[LENGTHS[0]]} → {m.iloc[-1]:.0f}% at {LABEL[LENGTHS[-1]]}")
""")

DZ_WRONG = code("""
    # What it said at 32K when it had to find the meaning
    last = df[(df.length == LENGTHS[-1]) & (df.kind == "meaning")]
    for _, r in last.iterrows():
        print(f"{'✓' if r.ok else '✗'}  {r.q:<44} expected {r.answer:<7} → {r.reply[:70]!r}")
""")

DUMB_ZONE = [
    md("""
        # The dumb zone, measured

        **Same fact. Same question. We only add more text around it.**

        An open 7B model (Qwen2.5-7B-Instruct, 32K-token window) reads a public-domain novel with one fact
        hidden inside. We grow the novel from 1K to 32K tokens and ask the same question every time.

        Two kinds of question:
        - **Find the words.** The question repeats the fact's words.
          *"What is the Wi-Fi password at the venue?"* ← *"The Wi-Fi password at the venue is blue-dorayaki-42."*
        - **Find the meaning.** No shared words; you need to know where the landmark is.
          *"Which character has been to Hyderabad?"* ← *"Kabir's flat looks straight out at the Charminar."*

        The second kind is how NoLiMa (Adobe, ICML 2025) tests long context, and it's closer to real work:
        your agent has to connect *"the payment bug"* to `checkout.ts`.
    """),
    SETUP, STYLE, MODEL, BOOKS_CELL, NEEDLES, PROMPTS,
    md("## Run it\n10 facts × 2 kinds × 6 lengths = 120 questions."),
    DZ_RUN,
    md("## The result"),
    DZ_PLOT,
    md("## What it said instead"),
    DZ_WRONG,
    md("""
        ## Read the chart
        - **Blue** is what most "needle in a haystack" scores measure: can it find the exact words?
        - **Red** is real work: can it connect two things that don't share words?
        - The window never filled up: 32K is the model's limit, so every point on this chart is *inside* the window.

        Sources: NoLiMa, Modarressi et al., ICML 2025 · Context Rot, Chroma, 2025.
    """),
]

# ── U-shape ─────────────────────────────────────────────────────────────────

# The grid. On Kaggle's 2× T4 one 32K question takes ~50 s (smoke test), so the full 10 facts × 9 depths
# × 4 lengths would run for 2+ hours. Keep all four lengths; use the first 5 facts at 7 depths (~1 h).
U_FACTS = 5
U_DEPTHS = [round(k / 6, 4) for k in range(7)]
U_LENGTHS = [4096, 8192, 16384, 31744]
FIX_LEN = 16384
U_QUESTIONS = U_FACTS * len(U_DEPTHS) * len(U_LENGTHS)

U_SETTINGS = code(
    f"# The grid. One 32K-token question takes ~50 s on Kaggle's 2× T4, so the full 10 facts × 9 depths\n"
    f"# × 4 lengths would run for over two hours. We keep all four lengths and use the first {U_FACTS} facts\n"
    f"# at {len(U_DEPTHS)} depths, from the very start (0) to the very end (1).\n"
    f"N_FACTS = {U_FACTS}\n"
    f"DEPTHS = {U_DEPTHS}\n"
    f"U_LENGTHS = {U_LENGTHS}\n"
    f"FIX_LEN = {FIX_LEN}\n\n"
    "def depth_label(d):\n"
    "    return {0: \"start\", 0.5: \"middle\", 1: \"end\"}.get(d, f\"{d:.0%}\")"
)

U_RUN = code("""
    # Run: every depth × every length, with the "find the meaning" questions
    def run_grid(lengths, depths, before=False, tag=""):
        out = []
        for L in lengths:
            t0 = time.time()
            for d in depths:
                for i, nd in enumerate(MEANING[:N_FACTS]):
                    reply, n = ask(prompt(insert(haystack(BOOKS[i], L), nd["fact"], d), nd, before=before))
                    out.append(dict(length=L, depth=d, needle=i, tokens=n, answer=nd["a"], reply=reply,
                                    ok=right(reply, nd["a"])))
            torch.cuda.empty_cache()
            row = [np.mean([r["ok"] for r in out if r["length"] == L and r["depth"] == d]) * 100 for d in depths]
            print(f"{tag}{LABEL[L]:>4}: " + " ".join(f"{v:4.0f}" for v in row) + f"   ({time.time() - t0:.0f}s)")
        return pd.DataFrame(out)

    print("depth:  " + " ".join(f"{d:4.0%}" for d in DEPTHS))
    grid = run_grid(U_LENGTHS, DEPTHS)
    grid.to_json("u_shape_grid.json", orient="records", indent=1)
""")

U_HEAT = code("""
    # Chart 1: where the fact sits × how long the prompt is
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ok", ["#c93b3b", "#e8a33d", "#3ecf8e"])
    mat = grid.groupby(["depth", "length"]).ok.mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (r, c), val in np.ndenumerate(mat.values):
        ax.text(c, r, f"{val:.0f}%", ha="center", va="center", fontsize=22, fontweight="bold", color=BG)
    ax.set_xticks(range(mat.shape[1]), [LABEL[L] for L in mat.columns])
    ax.set_yticks(range(mat.shape[0]), [depth_label(d) for d in mat.index])
    ax.tick_params(length=0)
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Where the fact sits")
    ax.set_title("Where the fact sits × how long the prompt is", loc="left", pad=20)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.01, 0.01, f"{MODEL_NAME} · {GPU} · {grid.needle.nunique()} facts per cell · "
             f"'find the meaning' questions · measured {TODAY}", color=MUTED, fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig("u_shape_heatmap.png", dpi=160)
    plt.show()
""")

U_CURVE = code("""
    # Chart 2: start, middle and end, at each length
    col = mat[U_LENGTHS[-1]]
    fig, ax = plt.subplots(figsize=(15, 7.5))
    xs = np.array(DEPTHS) * 100
    styles = {4096: ("#8a96a3", ":"), 8192: ("#aab4be", "--"), 16384: ("#d0d6dc", "-.")}
    for L in U_LENGTHS:
        last = L == U_LENGTHS[-1]
        color, ls = (RED, "-") if last else styles.get(L, (MUTED, "-"))
        ax.plot(xs, mat[L].values, color=color, ls=ls, lw=5 if last else 3, marker="o", ms=12 if last else 7,
                zorder=3 if last else 2, label=LABEL[L])
    ax.set_xticks(xs, [depth_label(d) for d in DEPTHS])
    ax.set_xlim(-5, 105)
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", alpha=0.7)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(U_LENGTHS), fontsize=18, borderaxespad=0.3,
              handlelength=2.6, title="prompt length", title_fontsize=15, alignment="left")
    ax.set_xlabel("Where the fact sits in the prompt")
    ax.set_ylabel("Found it")
    ax.set_title(f"At {LABEL[U_LENGTHS[-1]]}: start {col.iloc[0]:.0f}% · middle {col.loc[0.5]:.0f}% · "
                 f"end {col.iloc[-1]:.0f}%", loc="left", pad=80)
    fig.text(0.01, 0.01, f"{MODEL_NAME} · {GPU} · {grid.needle.nunique()} facts per point · "
             f"'find the meaning' questions · measured {TODAY}", color=MUTED, fontsize=12)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig("u_shape_curve.png", dpi=160)
    plt.show()
    for L in U_LENGTHS:
        c = mat[L]
        print(f"{LABEL[L]:>4}: start {c.iloc[0]:3.0f}% · middle {c.loc[0.5]:3.0f}% · end {c.iloc[-1]:3.0f}%")
""")

U_ATTN = code(SET_ATTN, """
    # Chart 3: inside the model. Where the question's attention goes, position by position.
    # We read the attention of the final question tokens over every earlier token,
    # averaged over all layers and heads, on 10 novels with no fact inside.
    ATT_LEN = 8192

    @torch.no_grad()
    def attention_profile(text):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.offset_mapping
        q0 = next(i for i, (a, b) in enumerate(offs) if a >= text.rfind("Question:"))
        d0 = next(i for i, (a, b) in enumerate(offs) if a >= text.find("<story>") + 8)
        d1 = next(i for i, (a, b) in enumerate(offs) if a >= text.find("</story>"))
        ids = torch.tensor([enc.input_ids], device=model.device)
        out = BASE(input_ids=ids[:, :q0], use_cache=True)
        set_attn("eager")
        try:
            out2 = BASE(input_ids=ids[:, q0:], past_key_values=out.past_key_values,
                        use_cache=True, output_attentions=True)
        finally:
            set_attn("sdpa")
        per_layer = torch.stack([a[0].float().mean(0).mean(0).cpu() for a in out2.attentions])
        return per_layer.numpy(), d0, d1, enc.input_ids

    profiles, sinks = [], []
    for i in range(len(BOOKS)):
        att, d0, d1, ids = attention_profile(prompt(haystack(BOOKS[i], ATT_LEN), MEANING[i]))
        mean = att.mean(0)
        T = len(mean)
        profiles.append(dict(token0=mean[0], template=mean[:d0].sum(), story=mean[d0:d1].sum(),
                             question=mean[d1:].sum(), head=mean[:d0], head_ids=ids[:d0],
                             bins=[b.mean() * T for b in np.array_split(mean[d0:d1], 60)]))
        sinks.append(att[:, 0])
    token0 = np.mean([p["token0"] for p in profiles])
    share = {k: float(np.mean([p[k] for p in profiles])) for k in ("template", "story", "question")}
    n0 = min(len(p["head"]) for p in profiles)
    head = np.mean([p["head"][:n0] for p in profiles], axis=0)
    top = [(int(j), tok.decode([profiles[0]["head_ids"][j]]), float(head[j])) for j in np.argsort(head)[::-1][:3]]
    bins = np.mean([p["bins"] for p in profiles], axis=0)
    sink_by_layer = np.mean(sinks, axis=0)
    assert np.isfinite(bins).all() and np.isfinite(sink_by_layer).all(), "attention weights are not finite"
    first, mid, tail = bins[:3].mean(), bins[len(bins) // 3: 2 * len(bins) // 3].mean(), bins[-3:].mean()
    torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(15, 7.5))
    xs = np.linspace(1, 99, len(bins))
    ax.plot(xs, bins, color=BLUE, lw=4, zorder=3)
    ax.axhline(1, color=MUTED, ls="--", lw=1.5)
    ax.text(101, 1, "even\\nshare", color=MUTED, fontsize=14, va="center")
    for x0, x1, y, name, dx, dy, ha in ((0, 5, first, "first 5%", 12, 36, "left"),
                                        (100 / 3, 200 / 3, mid, "middle third", 0, -46, "center"),
                                        (95, 100, tail, "last 5%", -12, 50, "right")):
        ax.hlines(y, x0, x1, color=AMBER, lw=5, zorder=4)
        ax.annotate(f"{name}: {y:.2f}×", ((x0 + x1) / 2, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", color=AMBER, fontsize=19, fontweight="bold")
    ax.set_yscale("log")
    lo, hi = min(bins.min(), 1), max(bins.max(), 1)
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo / 1.5 <= t <= hi * 1.5]
    ax.set_yticks(ticks, [f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(lo / 2, hi * 2.5)
    ax.set_xticks([0, 25, 50, 75, 100], ["start", "25%", "middle", "75%", "end"])
    ax.set_xlim(-3, 112)
    ax.set_xlabel("Position in the story")
    ax.set_ylabel("Attention per token  (× even share)")
    ax.set_title(f"Per token, the end gets {tail / mid:.1f}× the attention of the middle", loc="left", pad=20)
    ax.text(0.01, 0.97, f"Of the question's attention: story {share['story']:.0%} · chat template "
            f"{share['template']:.0%} (one {top[0][1]!r} token: {top[0][2]:.0%}) · the question itself "
            f"{share['question']:.0%}", transform=ax.transAxes, ha="left", va="top", color=FG, fontsize=15)
    ax.grid(axis="y", alpha=0.5)
    fig.text(0.01, 0.01, f"{MODEL_NAME} · {LABEL[ATT_LEN]} tokens · mean of {att.shape[0]} layers and all heads · "
             f"{len(profiles)} novels · story tokens only · measured {TODAY}", color=MUTED, fontsize=12)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig("u_shape_attention.png", dpi=160)
    plt.show()
    json.dump(dict(length=ATT_LEN, novels=len(profiles), token0=float(token0), share=share,
                   top_template_tokens=top, first=float(first), middle=float(mid), end=float(tail),
                   bins=[float(b) for b in bins], sink_by_layer=[float(s) for s in sink_by_layer]),
              open("u_shape_attention.json", "w"), indent=1)
    print(f"where the question's attention goes: chat template {share['template']:.0%} (token 0: {token0:.2%}) · "
          f"story {share['story']:.0%} · the question itself {share['question']:.0%}")
    print("most-attended template tokens:", " · ".join(f"#{j} {t!r} {s:.1%}" for j, t, s in top))
    print(f"story, per token: first 5% {first:.2f}× · middle third {mid:.2f}× · last 5% {tail:.2f}× an even share")
    print(f"per token, the end of the story gets {tail / mid:.1f}× the attention of the middle")
    print("token-0 share by layer:", " ".join(f"{s:.1%}" for s in sink_by_layer))
""")

U_FIX_RUN = code("""
    # Same facts, same depths, but the question comes before AND after the story
    fixed = run_grid([FIX_LEN], DEPTHS, before=True, tag="ask twice ")
    fixed.to_json("u_shape_fix.json", orient="records", indent=1)
""")

U_FIX_PLOT = code("""
    # Chart 4: question at the end only vs. question before and after
    base = mat[FIX_LEN].values
    after = fixed.groupby("depth").ok.mean().reindex(mat.index).values * 100
    m = DEPTHS.index(0.5)
    xs = np.array(DEPTHS) * 100
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.plot(xs, base, color=BLUE, lw=4, marker="o", ms=11, label="question at the end only", zorder=3)
    ax.plot(xs, after, color=AMBER, lw=4, marker="o", ms=11, label="question before and after", zorder=3)
    ax.set_xticks(xs, [depth_label(d) for d in DEPTHS])
    ax.set_xlim(-5, 105)
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", alpha=0.7)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=18, borderaxespad=0.3)
    ax.set_xlabel("Where the fact sits in the prompt")
    ax.set_ylabel("Found it")
    ax.set_title(f"Ask twice at {LABEL[FIX_LEN]}: middle {base[m]:.0f}% → {after[m]:.0f}% · "
                 f"average {base.mean():.0f}% → {after.mean():.0f}%", loc="left", pad=52)
    fig.text(0.01, 0.01, f"{MODEL_NAME} · {GPU} · {fixed.needle.nunique()} facts per point · "
             f"'find the meaning' questions · measured {TODAY}", color=MUTED, fontsize=12)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig("u_shape_fix.png", dpi=160)
    plt.show()
    print(f"middle: {base[m]:.0f}% → {after[m]:.0f}% · average over all depths: {base.mean():.0f}% → {after.mean():.0f}%")
""")

U_SUMMARY = code("""
    # What this run shows, printed from the numbers above
    col = mat[U_LENGTHS[-1]]
    s, mi, e = col.iloc[0], col.loc[0.5], col.iloc[-1]
    where = f"At {LABEL[U_LENGTHS[-1]]}: start {s:.0f}%, middle {mi:.0f}%, end {e:.0f}%"
    if mi < s and mi < e:
        print(f"{where}. The middle is the weakest spot: lost in the middle.")
    elif e > s and e > mi:
        print(f"{where}. Only the end holds up: the start is lost as well as the middle.")
    else:
        print(f"{where}.")
    change = after.mean() - base.mean()
    print(f"Asking the question twice at {LABEL[FIX_LEN]}: {base.mean():.0f}% → {after.mean():.0f}% found on average "
          f"({'better' if change > 0 else 'worse' if change < 0 else 'no change'}).")
    print(f"The question's attention: story {share['story']:.0%} · chat template {share['template']:.0%} · "
          f"token 0 {token0:.2%} · biggest single token {top[0][1]!r} {top[0][2]:.0%} · "
          f"per story token, the end gets {tail / mid:.1f}× the middle.")
""")

U_SHAPE = [
    md(f"""
        # Lost in the middle, measured

        **Hide one fact in a long prompt. Move it from the top to the bottom. Where can the model still find it?**

        An open 7B model (Qwen2.5-7B-Instruct) reads a public-domain novel with one fact hidden at
        {len(U_DEPTHS)} depths, from the very start to the very end, at {len(U_LENGTHS)} prompt lengths
        (4K to 32K tokens). The questions share no words with the fact
        (*"Which character has been to Hyderabad?"* ← *"Kabir's flat looks straight out at the Charminar."*),
        so the model has to understand the story, not match words.

        Then we look inside: where does the question's attention go, position by position?
    """),
    SETUP, STYLE, MODEL, BOOKS_CELL, NEEDLES, PROMPTS,
    md(f"""
        ## Run it
        {U_FACTS} facts × {len(U_DEPTHS)} depths × {len(U_LENGTHS)} lengths = {U_QUESTIONS} questions.

        Why not all 10 facts at more depths? On the 2× T4 GPUs this ran on, one 32K-token question takes
        about 50 seconds, so the full 10 × 9 × 4 grid would take over two hours. Each cell below is
        {U_FACTS} questions, so one answer moves it by {100 // U_FACTS} points: read small differences as noise.
    """),
    U_SETTINGS, U_RUN,
    md("## Where the fact sits × how long the prompt is"),
    U_HEAT,
    md("## Start, middle, end"),
    U_CURVE,
    md("""
        ## Inside the model: where the attention goes
        Attention is a budget that adds up to 100%. Here is where the question's budget lands, position by
        position, in novels with no fact hidden in them.
    """),
    U_ATTN,
    md("""
        - The dashed line is an even share: what each story token would get if attention were spread evenly.
          Above it, a token gets more than its share; below it, less.
        - The story is only part of the prompt. The printout splits the question's attention between the chat
          template, the story and the question itself, and shows how much lands on token 0, where many models
          park spare attention (the *attention sink*, Xiao et al., StreamingLLM, 2023).
    """),
    md(f"## Ask the question twice: before and after the story\nSame {U_FACTS} facts and "
       f"{len(U_DEPTHS)} depths, at {FIX_LEN // 1024}K tokens."),
    U_FIX_RUN, U_FIX_PLOT,
    md("## What this run shows"),
    U_SUMMARY,
    md("""
        ## Use it
        - Where a model loses facts depends on the model, the prompt and the length, so measure yours like this.
        - Common advice (put key facts at the top, ask the question twice) is worth testing, not assuming:
          the printout above shows what it did for this model.

        Sources: Liu et al. 2023, *Lost in the Middle* · Xiao et al. 2023, StreamingLLM ·
        Hsieh et al. 2024, *Found in the Middle* · NoLiMa, ICML 2025.
    """),
]

# ── Smoke test: proves the pipeline end to end in a few minutes ─────────────

SMOKE = [
    md("# Smoke test\nPrompt checks, speed from 1K to 32K, the attention pass on real novels, and a chart layout "
       "preview with made-up numbers."),
    SETUP, STYLE, MODEL, BOOKS_CELL, NEEDLES, PROMPTS, U_SETTINGS,
    code("""
        # Prompt checks: every book is long enough, every prompt fits the window, every fact sits where we asked
        assert min(len(TOKS[n]) for n in BOOKS) > 33000, "a book is too short"
        rng = random.Random(7)
        DEPTH = [round(rng.uniform(0.2, 0.8), 3) for _ in BOOKS]
        worst = 0
        for L in (1024, 2048, 4096, 8192, 16384, 31744):
            errs, ntok = [], []
            for needles in (WORDS, MEANING):
                for i, nd in enumerate(needles):
                    doc = insert(haystack(BOOKS[i], L), nd["fact"], DEPTH[i])
                    p = prompt(doc, nd)
                    assert p.count(nd["fact"]) == 1 and p.count(nd["q"]) == 1
                    ntok.append(len(tok(p, add_special_tokens=False).input_ids))
                    errs.append(abs(doc.find(nd["fact"]) / len(doc) - DEPTH[i]))
            worst = max(worst, max(ntok))
            print(f"dumb zone {LABEL[L]:>4}: prompts {min(ntok):>6,}–{max(ntok):>6,} tokens · depth error ≤ {max(errs):.3f}")
            # facts go in at sentence ends, so short prompts (few sentences) can't hit the depth exactly
            assert max(errs) < (0.1 if L < 4096 else 0.03), L
        assert worst < 32768 - 32, worst
        for L in U_LENGTHS:
            err = 0
            for i, nd in enumerate(MEANING):
                hs = haystack(BOOKS[i], L)
                for d in DEPTHS:
                    doc = insert(hs, nd["fact"], d)
                    err = max(err, abs(doc.find(nd["fact"]) / len(doc) - d))
            print(f"u-shape   {LABEL[L]:>4}: depth error ≤ {err:.3f}")
            assert err < 0.03, L
        assert prompt("x", MEANING[0], before=True).count(MEANING[0]["q"]) == 2
        for reply, ans, exp in [("Kabir", "Kabir", True), ("The character is Kabir.", "Kabir", True),
                                ("Kabirs", "Kabir", False), ("blue-dorayaki-42", "blue-dorayaki-42", True),
                                ("blue-dorayaki-421", "blue-dorayaki-42", False), ("9:40 pm", "9:40", True),
                                ("Room 4C-17", "4C-17", True), ("Elizabeth", "Tara", False)]:
            assert right(reply, ans) == exp, (reply, ans)
        print(f"max prompt {worst:,} tokens (window 32,768) · scoring ok")
    """),
    code("""
        # Speed and memory: three questions per length
        timing = []
        for L in (1024, 4096, 8192, 16384, 31744):
            for kind, i in (("words", 0), ("meaning", 0), ("meaning", 7)):
                nd = (WORDS if kind == "words" else MEANING)[i]
                t0 = time.time()
                reply, n = ask(prompt(insert(haystack(BOOKS[i], L), nd["fact"], 0.5), nd))
                dt = time.time() - t0
                ok = right(reply, nd["a"])
                timing.append(dict(length=L, kind=kind, needle=i, tokens=n, seconds=round(dt, 2), reply=reply, ok=ok))
                print(f"{L:>6} {kind:<8} {n:>6} tok {dt:6.1f}s ok={ok!s:<5} {reply[:60]!r}")
        print("peak GPU memory:", [f"{torch.cuda.max_memory_allocated(i) / 2**30:.1f} GB"
                                   for i in range(torch.cuda.device_count())])
        json.dump(timing, open("smoke.json", "w"), indent=1)
    """),
    code(SET_ATTN, """
        # One attention pass: every row of weights must be finite and sum to 1
        print("attention check: start", flush=True)
        text = prompt(haystack(BOOKS[0], 8192), MEANING[0])
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        q0 = next(i for i, (a, b) in enumerate(enc.offset_mapping) if a >= text.rfind("Question:"))
        ids = torch.tensor([enc.input_ids], device=model.device)
        with torch.no_grad():
            out = BASE(input_ids=ids[:, :q0], use_cache=True)
            set_attn("eager")
            try:
                out2 = BASE(input_ids=ids[:, q0:], past_key_values=out.past_key_values,
                            use_cache=True, output_attentions=True)
            finally:
                set_attn("sdpa")
        a = out2.attentions
        sums = torch.stack([x[0].float().sum(-1).cpu() for x in a])
        check = dict(layers=len(a), shape=list(a[0].shape), finite=bool(all(torch.isfinite(x).all() for x in a)),
                     row_sum_error=float((sums - 1).abs().max()),
                     token0=float(torch.stack([x[0].float().mean(0).mean(0)[0].cpu() for x in a]).mean()),
                     peak_gb=[round(torch.cuda.max_memory_allocated(i) / 2**30, 1)
                              for i in range(torch.cuda.device_count())])
        json.dump(check, open("smoke_attention.json", "w"), indent=1)
        print(check, flush=True)
        assert check["finite"] and check["row_sum_error"] < 1e-3, check
        del out, out2, a
        torch.cuda.empty_cache()
    """),
    md("## The real attention profile (the same cell as in the U-shape notebook)"),
    U_ATTN,
    md("## Chart layout preview: MADE-UP numbers, not results\n"
       "Worst cases for label collisions, stamped and saved as `preview_*.png`."),
    code("""
        # MADE-UP numbers, for a layout check only
        _savefig = plt.savefig
        def preview_savefig(name, **kw):
            plt.gcf().text(0.5, 0.5, "LAYOUT PREVIEW · MADE-UP NUMBERS", ha="center", va="center", rotation=10,
                           fontsize=44, fontweight="bold", color=AMBER, alpha=0.45)
            _savefig("preview_" + name, **kw)
        plt.savefig = preview_savefig

        LENGTHS = [1024, 2048, 4096, 8192, 16384, 31744]
        fake = {"words": [100, 100, 100, 100, 100, 100], "meaning": [100, 90, 90, 80, 70, 90]}
        df = pd.DataFrame([dict(kind=k, length=L, needle=i, ok=i < p // 10)
                           for k in fake for L, p in zip(LENGTHS, fake[k]) for i in range(10)])

        def fake_u(L, d):
            return 100 if d in (0, 1) else max(0, 80 - 20 * U_LENGTHS.index(L) - (40 if d == 0.5 else 0))

        grid = pd.DataFrame([dict(length=L, depth=d, needle=i, ok=i < fake_u(L, d) * N_FACTS // 100)
                             for L in U_LENGTHS for d in DEPTHS for i in range(N_FACTS)])
        fixed = pd.DataFrame([dict(length=FIX_LEN, depth=d, needle=i, ok=i < N_FACTS - 1)
                              for d in DEPTHS for i in range(N_FACTS)])
    """),
    DZ_PLOT, U_HEAT, U_CURVE, U_FIX_PLOT, U_SUMMARY,
    code("""
        json.dump(dict(done=True, gpu=GPU), open("smoke_done.json", "w"))
        print("smoke test complete", flush=True)
    """),
]


def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def write(folder, slug, title, cells, gpu):
    d = HERE / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.ipynb").write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False))
    (d / "kernel-metadata.json").write_text(json.dumps(metadata(slug, title, gpu), indent=2))


if __name__ == "__main__":
    # NvidiaL4 is accepted but this account gets 2x T4 for it, so ask for T4 explicitly
    gpu = sys.argv[1] if len(sys.argv) > 1 else "NvidiaTeslaT4"
    write("dumb-zone", "dumb-zone-measured", "Dumb Zone Measured", DUMB_ZONE, gpu)
    write("u-shape", "lost-in-the-middle-measured", "Lost in the Middle Measured", U_SHAPE, gpu)
    write("smoke", "context-window-smoke-test", "Context Window Smoke Test", SMOKE, gpu)
    hay = HERE / "haystack"
    hay.mkdir(exist_ok=True)
    src = REPO / "u-shape/.cache/gutenberg"
    for n in BOOKS:
        shutil.copy(src / f"pg{n}.txt", hay / f"pg{n}.txt")
    (hay / "dataset-metadata.json").write_text(json.dumps({
        "title": "Gutenberg Novels Haystack", "id": DATASET, "licenses": [{"name": "other"}],
        "description": "Ten public-domain novels from Project Gutenberg, used as filler text "
                       "for long-context tests."}, indent=2))
    print("built:", sorted(str(p.relative_to(HERE)) for p in HERE.rglob("*.ipynb")))

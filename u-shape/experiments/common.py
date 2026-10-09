"""Shared pieces for the U-shape experiments: model, haystack, needles, prompt assembly, cached prefill.

Prompts are assembled at the token level: [chat prefix][haystack sentences with the needle spliced in at a
sentence boundary][question suffix]. Every piece is tokenized on its own, so the haystack prefix before the
needle is identical across depths and its KV cache is computed once and reused (trimmed back after each trial).
"""
import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

ROOT = Path(__file__).resolve().parents[1]  # u-shape/
CACHE = ROOT / ".cache"
DATA = ROOT / "data"
MODEL_DIR = CACHE / "Qwen2.5-3B-Instruct-mlx-4bit"
MODEL_NAME = "Qwen2.5-3B-Instruct (MLX 4-bit, group size 64)"
BOOKS_DIR = CACHE / "gutenberg"

DEPTHS = [i / 10 for i in range(11)]
LENGTHS = [2048, 8192]

# Project Gutenberg ebook id -> title. Public domain in the USA; PG header/footer stripped.
BOOKS = [
    (1342, "Pride and Prejudice"),
    (1661, "The Adventures of Sherlock Holmes"),
    (84, "Frankenstein"),
    (2701, "Moby Dick"),
    (98, "A Tale of Two Cities"),
    (35, "The Time Machine"),
    (345, "Dracula"),
    (1400, "Great Expectations"),
    (36, "The War of the Worlds"),
    (120, "Treasure Island"),
]

_COLOURS = ["red", "green", "pink", "orange", "purple", "yellow", "silver", "golden", "brown", "grey"]
_SNACKS = ["samosa", "jalebi", "vada", "ladoo", "momo", "pakora", "idli", "kachori", "dhokla", "rasgulla"]
_TITLES = ["Major", "Doctor", "Professor", "Colonel", "Sir", "Lady", "General", "Admiral", "Baron"]
_SNACKS_CAP = ["Jalebi", "Samosa", "Ladoo", "Momo", "Idli", "Vada", "Dhokla", "Kachori", "Barfi"]
_ADJ = ["roasted", "crushed", "toasted", "pickled", "dried", "candied", "burnt", "fresh", "salted"]
_SPICE = ["fennel", "nutmeg", "mace", "ajwain", "star anise", "amchur", "curry leaf", "saffron", "jaggery"]
_CHAI = ["ADRAK", "ELAICHI", "TULSI", "KESAR", "IRANI", "CUTTING", "SULEMANI", "NOON", "KULHAD"]

# (statement template, question template, target key, target answer, look-alike keys, look-alike value maker)
# Target answers are arbitrary strings, so the model can't guess them from prior knowledge. The look-alikes share
# the target's sentence template with a different key and value (none contains the target key or answer).
NEEDLES = [
    ("The Wi-Fi password at the {k} is {v}.", "What is the Wi-Fi password at the {k}?", "venue", "blue-dorayaki-42",
     ["library", "hostel", "canteen", "gym", "auditorium", "guest house", "staff room", "admin block", "sports complex"],
     lambda r: f"{r.choice(_COLOURS)}-{r.choice(_SNACKS)}-{r.randint(10, 99)}"),
    ("The secret code for free chai at the {k} is {v}.", "What is the secret code for free chai at the {k}?", "canteen", "MASALA-7781",
     ["library cafe", "hostel mess", "staff room", "food court", "night stall", "station stall", "juice centre", "tuck shop", "guest house"],
     lambda r: f"{r.choice(_CHAI)}-{r.randint(1000, 9999)}"),
    ("The {k} demo room is Room {v}.", "Which room is the {k} demo room?", "hackathon", "4C-17",
     ["robotics club", "design team", "startup cell", "quiz club", "coding club", "drone team", "music society", "film club", "AI lab"],
     lambda r: f"{r.randint(1, 9)}{r.choice('ABDEFGH')}-{r.randint(10, 99)}"),
    ("The locker combination for the {k} is {v}.", "What is the locker combination for the {k}?", "robotics lab", "38-11-52",
     ["chemistry lab", "physics lab", "music room", "gym", "library", "drone lab", "art studio", "sports room", "IoT lab"],
     lambda r: f"{r.randint(10, 99)}-{r.randint(10, 99)}-{r.randint(10, 99)}"),
    ("{k}'s pet tortoise is named {v}.", "What is the name of {k}'s pet tortoise?", "Priya", "Captain Pakora",
     ["Rahul", "Ananya", "Arjun", "Meera", "Kabir", "Sneha", "Rohan", "Isha", "Vikram"],
     lambda r: f"{r.choice(_TITLES)} {r.choice(_SNACKS_CAP)}"),
    ("The admin port of the {k} is {v}.", "What is the admin port of the {k}?", "demo server", "47291",
     ["build server", "test server", "mail server", "backup server", "staging server", "game server", "chat server", "file server", "proxy server"],
     lambda r: str(r.randint(10000, 65000))),
    ("The {k}'s lucky jersey number this season is {v}.", "What is the {k}'s lucky jersey number this season?", "cricket team", "863",
     ["football team", "hockey team", "kabaddi team", "chess team", "basketball team", "volleyball team", "badminton team", "tennis team", "athletics team"],
     lambda r: str(r.randint(100, 999))),
    ("The secret ingredient in {k}'s biryani is {v}.", "What is the secret ingredient in {k}'s biryani?", "Ravi", "smoked kokum",
     ["Asha", "Imran", "Deepa", "Farhan", "Lakshmi", "Manoj", "Nisha", "Suresh", "Zoya"],
     lambda r: f"{r.choice(_ADJ)} {r.choice(_SPICE)}"),
    ("The {k} leaves from Gate {v}.", "Which gate does the {k} leave from?", "campus shuttle", "14B",
     ["airport bus", "city bus", "night bus", "metro feeder", "hostel van", "college bus", "staff bus", "sports bus", "library van"],
     lambda r: f"{r.randint(1, 29)}{r.choice('ACDEFGH')}"),
    ("The DJ night at the {k} starts at {v} pm.", "What time does the DJ night at the {k} start?", "fest", "9:40",
     ["freshers party", "farewell", "alumni meet", "hostel day", "cultural week", "tech week", "sports day", "Diwali party", "Holi party"],
     lambda r: f"{r.randint(6, 11)}:{r.choice(['05', '10', '15', '20', '25', '30', '35', '45', '50', '55'])}"),
]


def needle_spec(i, seed=0):
    """Target statement/question/answer for needle i, plus its 9 look-alike statements (deterministic)."""
    import random
    stmt, ques, key, ans, keys, make = NEEDLES[i]
    r = random.Random(1000 * seed + i)
    looks, used = [], {ans.lower()}
    for k in keys:
        v = make(r)
        while v.lower() in used or ans.lower() in v.lower() or v.lower() in ans.lower():
            v = make(r)
        used.add(v.lower())
        looks.append(stmt.format(k=k, v=v))
    return stmt.format(k=key, v=ans), ques.format(k=key), ans, looks


def lookalike_values(i, seed=0):
    """The 9 look-alike answers for needle i, in the same order needle_spec builds them."""
    import random
    stmt, ques, key, ans, keys, make = NEEDLES[i]
    r = random.Random(1000 * seed + i)
    vals, used = [], {ans.lower()}
    for _ in keys:
        v = make(r)
        while v.lower() in used or ans.lower() in v.lower() or v.lower() in ans.lower():
            v = make(r)
        used.add(v.lower())
        vals.append(v)
    return vals

INSTRUCTION = "Read the document below and answer the question after it."
ANSWER_RULE = "Answer with just the answer from the document."
EOS_TEXT = ("<|im_end|>", "<|endoftext|>")


def env_info():
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip())
    except Exception:
        chip, mem = platform.processor(), 0
    return {
        "machine": f"{chip}, {round(mem / 2**30)} GB",
        "os": platform.platform(),
        "mlx": _md.version("mlx"),
        "mlx_lm": _md.version("mlx-lm"),
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def load_model():
    mx.set_cache_limit(2 * 2**30)
    model, tok = load(str(MODEL_DIR))
    return model, tok


# ───────── haystack ─────────

def book_paragraphs(book_id):
    raw = (BOOKS_DIR / f"pg{book_id}.txt").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG E[Bb]OOK[^\n]*", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG E[Bb]OOK", raw)
    body = raw[start.end(): end.start()]
    paras = []
    for p in re.split(r"\n\s*\n", body):
        p = re.sub(r"\[Illustration[^\]]*\]", " ", p)
        p = re.sub(r"\s+", " ", p.replace("_", "")).strip()
        if p:
            paras.append(p)
    # Skip front matter (title page, contents): start at the first run of three prose paragraphs.
    for i in range(len(paras) - 2):
        if all(len(paras[j]) >= 200 for j in (i, i + 1, i + 2)):
            return paras[i:]
    return paras


_SENT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019]))\s+(?=[\"'\u201c\u2018(\[A-Z0-9])")
_ABBR = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Mt|Jr|Sr|Messrs|Mme|Mlle|Capt|Col|Gen|Lt|Rev|Prof|Hon|No|Vol|viz|etc)|\b[A-Z])\.$")


def book_sentences(book_id):
    """Haystack pieces, each with its leading separator: '' (first), ' ' (same paragraph) or '\n\n' (new one)."""
    out = []
    for k, p in enumerate(book_paragraphs(book_id)):
        sents = []
        for s in _SENT.split(p):
            if sents and _ABBR.search(sents[-1]):
                sents[-1] += " " + s
            else:
                sents.append(s)
        out += [("" if k == 0 and j == 0 else "\n\n" if j == 0 else " ") + s for j, s in enumerate(sents)]
    return out


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


class Prompt:
    """One (needle, book, length, mode) trial. Builds the prompt text for any depth; tokenize with ids().

    The needle goes in at the sentence boundary nearest the requested depth (by token position).
    lookalikes=True also puts the needle's 9 look-alike statements into the haystack, one at a random sentence
    boundary in each ninth of the document (by tokens). They stay put across depths, so only the target moves.
    """

    def __init__(self, tok, book_id, needle_idx, total_len, mode="after", lookalikes=False, seed=0):
        import random
        self.tok = tok
        self.statement, self.question, self.answer, looks = needle_spec(needle_idx, seed)
        q = self.question
        if mode == "after":
            user = f"{INSTRUCTION}\n\n<document>\n@@DOC@@\n</document>\n\nQuestion: {q}\n{ANSWER_RULE}"
        elif mode == "both":  # query-aware contextualisation (Liu et al. 2023, sec. 4.3): question before and after
            user = f"{INSTRUCTION}\n\nQuestion: {q}\n\n<document>\n@@DOC@@\n</document>\n\nQuestion: {q}\n{ANSWER_RULE}"
        else:
            raise ValueError(mode)
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        self.pre, self.post = text.split("@@DOC@@")
        n_fixed = len(encode(tok, self.pre)) + len(encode(tok, self.post)) + len(encode(tok, " " + self.statement))
        look_txt = [" " + s for s in looks] if lookalikes else []
        look_len = [len(encode(tok, s)) for s in look_txt]
        budget = total_len - n_fixed - sum(look_len)
        pieces, lens = [], []
        for s in book_sentences(book_id):
            c = len(encode(tok, s))  # per-piece counts slightly overestimate the joined text, so we stay under
            if sum(lens) + c > budget:
                break
            pieces.append(s)
            lens.append(c)
        look_idx = []
        if look_txt:
            r = random.Random(f"{needle_idx}-{total_len}-{seed}")
            order = list(range(len(look_txt)))
            r.shuffle(order)
            cum = [0]
            for c in lens:
                cum.append(cum[-1] + c)
            k, picks = len(look_txt), []
            for j in range(k):
                target = (j + r.random()) / k * cum[-1]
                cands = sorted(range(1, len(pieces)), key=lambda i: abs(cum[i] - target))
                picks.append(next(i for i in cands if i not in picks))
            for at, o in sorted(zip(picks, order), reverse=True):
                pieces.insert(at, look_txt[o])
                lens.insert(at, look_len[o])
            look_idx = sorted(i + n for n, i in enumerate(sorted(picks)))
        self.pieces = pieces
        self.cum = [0]
        for c in lens:
            self.cum.append(self.cum[-1] + c)
        self.lookalike_pos = [round(self.cum[i] / self.cum[-1], 4) for i in look_idx]

    def boundary(self, depth):
        """Index of the piece boundary nearest to depth (0 = before the first sentence, len = after the last)."""
        target = depth * self.cum[-1]
        return min(range(len(self.cum)), key=lambda i: (abs(self.cum[i] - target), i))

    def depth_actual(self, depth):
        return round(self.cum[self.boundary(depth)] / self.cum[-1], 4)

    def document(self, depth=None):
        if depth is None:
            return "".join(self.pieces)
        i = self.boundary(depth)
        if i == 0:
            return self.statement + " " + "".join(self.pieces)
        return "".join(self.pieces[:i]) + " " + self.statement + "".join(self.pieces[i:])

    def text(self, depth=None):
        return self.pre + self.document(depth) + self.post

    def ids(self, depth=None):
        return encode(self.tok, self.text(depth))


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ───────── cached prefill + greedy decoding ─────────

def _lm_head(model, h):
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def prefill(model, cache, ids, want_logits=False, step=2048):
    """Append ids to the cache. Optionally return logits for the last position only."""
    logits = None
    for s in range(0, len(ids), step):
        chunk = mx.array(ids[s: s + step])[None]
        h = model.model(chunk, cache=cache)
        if want_logits and s + step >= len(ids):
            logits = _lm_head(model, h[:, -1:, :])[0, -1]
            mx.eval(logits, [c.state for c in cache])
        else:
            mx.eval([c.state for c in cache])
    return logits


def greedy(model, cache, logits, eos_ids, max_new=24):
    out = []
    for _ in range(max_new):
        t = int(mx.argmax(logits).item())
        if t in eos_ids:
            break
        out.append(t)
        logits = prefill(model, cache, [t], want_logits=True)
    return out


def trim_to(cache, n_tokens):
    extra = cache[0].offset - n_tokens
    if extra > 0:
        trim_prompt_cache(cache, extra)


def eos_ids(tok):
    ids = {tok.convert_tokens_to_ids(t) for t in EOS_TEXT}
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


def is_correct(reply, answer):
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    return norm(answer) in norm(reply)


def save_json(name, obj):
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return path


__all__ = [n for n in dir() if not n.startswith("_")] + ["make_prompt_cache"]

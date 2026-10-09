"""Measure how a real coding agent's context fills up, from Codex CLI session logs on this laptop.

Reads ~/.codex/sessions/**/*.jsonl and writes AGGREGATE token counts only (no log content) to
dumb-zone/data/codex-sessions.json. Context size at each model call = last_token_usage.input_tokens
from Codex's own token_count events. Text that Codex logs verbatim (base instructions, AGENTS.md,
compaction summaries) is counted exactly with tiktoken's o200k_base, the GPT-4o/GPT-5 tokenizer.

Run: dumb-zone/.venv/bin/python dumb-zone/experiment/measure_codex_sessions.py
"""
import json
import statistics as st
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent
LOGS = Path.home() / ".codex" / "sessions"
OUT = ROOT / "data" / "codex-sessions.json"
STEPS = 40
enc = tiktoken.get_encoding("o200k_base")
ntok = lambda s: len(enc.encode(s, disallowed_special=()))
READ_HINTS = ("cat ", "sed -n", "nl -ba", "head ", "tail ", "less ", "bat ", '"view"', "read_file")


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


def session(path):
    rows = []
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    meta = next((r["payload"] for r in rows if r.get("type") == "session_meta"), None)
    if not meta or meta.get("forked_from_id") or meta.get("parent_thread_id"):
        return None  # skip forks and sub-agents: they start from someone else's context

    calls, window = [], None          # calls: [ctx_tokens, {chat, tool, read} chars appended since previous call]
    pending = {"chat": 0, "tool": 0, "read": 0}
    pre = {"agents_md": 0, "dev": 0, "user": 0}
    call_kind, compactions, last_total, compacted_since = {}, [], None, False
    for r in rows:
        t, p = r.get("type"), r.get("payload")
        if not isinstance(p, dict):
            continue
        pt = p.get("type")
        if t == "compacted":
            compactions.append({"summary_tokens": ntok(str(p.get("message", ""))), "before": calls[-1][0] if calls else None})
            compacted_since = True
        elif t == "event_msg" and pt == "token_count":
            info = p.get("info") or {}
            tot = (info.get("total_token_usage") or {}).get("total_tokens")
            ctx = (info.get("last_token_usage") or {}).get("input_tokens")
            window = info.get("model_context_window") or window
            if not ctx or tot == last_total:
                continue
            last_total = tot
            if compacted_since and compactions:
                compactions[-1]["after"] = ctx
                compactions[-1]["above_first_call"] = ctx - calls[0][0] if calls else None
                compacted_since = False
                calls.append([ctx, None])  # growth across a compaction is not organic
            else:
                calls.append([ctx, pending])
            pending = {"chat": 0, "tool": 0, "read": 0}
        elif t == "response_item":
            if pt == "message":
                body = text_of(p.get("content"))
                if not calls:
                    if p.get("role") == "developer":
                        pre["dev"] += ntok(body)
                    elif "AGENTS.md" in body[:200] or "<user_instructions>" in body[:200]:
                        pre["agents_md"] += ntok(body)
                    else:
                        pre["user"] += ntok(body)
                pending["chat"] += len(body)
            elif pt in ("function_call", "custom_tool_call"):
                args = str(p.get("arguments") or p.get("input") or "")
                call_kind[p.get("call_id")] = "read" if any(h in args for h in READ_HINTS) else "tool"
                pending["chat"] += len(args)
            elif pt in ("function_call_output", "custom_tool_call_output"):
                pending[call_kind.get(p.get("call_id"), "tool")] += len(str(p.get("output", "")))
    if len(calls) < 2:
        return None
    return {
        "system_tokens": ntok((meta.get("base_instructions") or {}).get("text", "")),
        "pre": pre, "calls": calls, "window": window, "compactions": compactions,
    }


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs)) if xs else None


def main():
    sessions = [s for s in (session(p) for p in sorted(LOGS.rglob("*.jsonl"))) if s]
    first = [s["calls"][0][0] for s in sessions]
    tools = [s["calls"][0][0] - s["system_tokens"] - sum(s["pre"].values()) for s in sessions]

    steps = []
    for t in range(1, STEPS + 1):
        grown, share = [], {"chat": 0.0, "tool": 0.0, "read": 0.0}
        for s in sessions:
            c = s["calls"]
            if len(c) <= t or any(x[1] is None for x in c[1:t + 1]):
                continue
            g = c[t][0] - c[0][0]
            grown.append(g)
            chars = {k: sum(x[1][k] for x in c[1:t + 1]) for k in share}
            total = sum(chars.values())
            for k in share:
                share[k] += max(g, 0) * (chars[k] / total if total else (1.0 if k == "chat" else 0.0))
        if not grown:
            break
        m, tot = st.median(grown), sum(share.values()) or 1
        steps.append({"step": t, "sessions": len(grown), "median_growth_since_first_call": round(m),
                      **{k: round(m * v / tot) for k, v in share.items()}})

    comp = [c for s in sessions for c in s["compactions"] if c.get("after")]
    peak_frac = [max(x[0] for x in s["calls"]) / s["window"] for s in sessions if s["window"]]
    result = {
        "source": "Codex CLI session logs on the speaker's laptop (~/.codex/sessions); aggregate token counts only",
        "tokenizer": "tiktoken o200k_base for logged text; per-call context = Codex's own last_token_usage.input_tokens",
        "sessions": len(sessions),
        "first_call_context": {"median": med(first)},
        "system_prompt_tokens": {"median": med([s["system_tokens"] for s in sessions])},
        "developer_messages_tokens": {"median": med([s["pre"]["dev"] for s in sessions])},
        "agents_md_tokens": {"median": med([s["pre"]["agents_md"] for s in sessions if s["pre"]["agents_md"]]),
                              "sessions_with_agents_md": sum(1 for s in sessions if s["pre"]["agents_md"])},
        "first_user_messages_tokens": {"median": med([s["pre"]["user"] for s in sessions])},
        "tool_definitions_tokens_derived": {"median": med(tools),
                                            "how": "first-call context minus every logged text before the first call"},
        "calls_per_session": {"median": med([len(s["calls"]) for s in sessions])},
        "model_context_window": {"median": med([s["window"] for s in sessions])},
        "sessions_past_40pct_of_window": {"count": sum(1 for f in peak_frac if f >= 0.4), "of": len(peak_frac)},
        "compactions": {"count": len(comp), "summary_tokens_median": med([c["summary_tokens"] for c in comp]),
                        "context_before_median": med([c["before"] for c in comp]),
                        "context_after_median": med([c["after"] for c in comp]),
                        "after_minus_same_session_first_call_median": med([c.get("above_first_call") for c in comp]),
                        "how": "summary = the compaction message Codex logs; after = first model call after it"},
        "growth_per_step": steps,
        "growth_split": "median growth since the first call, over sessions that reached the step without compacting; split by the share of characters appended so far: chat (messages + tool-call text), tool outputs, file reads",
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "growth_per_step"}, indent=1))
    print("steps:", [(s["step"], s["median_growth_since_first_call"], s["chat"], s["tool"], s["read"], s["sessions"]) for s in steps])


if __name__ == "__main__":
    main()

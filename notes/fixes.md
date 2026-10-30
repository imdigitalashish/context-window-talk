# Fixes

All three laws share one cause: attention is a budget that adds up to 100%. Every fix spends it on fewer, better tokens.

## Dumb zone

1. **Watch the meter.** Check `/context`. Compact or `/clear` around 40% full, not at the limit (Dex Horthy, HumanLayer).
2. **Save what matters before compacting.** Write decisions and must-not-break rules to a file such as AGENTS.md or NOTES.md, because summaries drop details.
3. **Restart broken chats.** After a wrong turn, start a fresh chat with a short summary. Arguing in the same chat repeats the mistake (self-conditioning), and it costs more because every message re-sends the whole history.
4. **Send a wingman.** Sub-agents explore in their own window and return a short summary (1–2K tokens, per Anthropic).

## Clustering

5. **One ask per prompt:** Input → Outcome (a metric) → Validation.
6. **Say what to do, not what not to do.** "Use plain CSS," not "don't use Tailwind." Enforce real "never" rules with tests, linters or hooks.
7. **Keep the rules file short,** with the most important rule first. The best model followed 68% of 500 rules (IFScale, 2025).
8. **Use exact names:** file paths, function names, error strings.
9. **Work in phases:** research → plan → build, each phase's output feeding the next.

## U-shape

10. **Put what matters last.** Long material goes first, and your question and key rules go at the end. In our 7B test at 32K tokens, a fact at the end was found 80% of the time; at the start or in the middle, 0%.
11. **Repeat critical rules near the end** instead of trusting the top of a long prompt.

**Dropped:** "ask the question twice." In our Kaggle test it made results worse (average 49% → 14%).

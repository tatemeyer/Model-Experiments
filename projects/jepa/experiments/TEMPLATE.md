# <One-line question this experiment answers> (issue #NN)

<Motivation: what prior result/issue prompted this, and why this specific
question/mechanism is a genuinely different lever than what's already been
tried. Cite the paper(s) motivating the approach, with arXiv links — also
add/update the corresponding row in `../../LITERATURE.md`.>

<Implementation: what function/class was added (file + name), what stayed
fixed vs. what's the controlled variable, any non-obvious implementation
detail (a derivative trick, a new dependency with the What/Why trusted/What
it costs breakdown per `CONVENTIONS.md`, etc.).>

**Result: <one-line verdict>.**

| variant | metric (seeds ...) |
|---|---|
| ... | ... |

<Prose interpretation of the table.>

**<Mechanistic/pointwise diagnosis, if the result is surprising or
negative — don't stop at a bare aggregate number. Instrument and show a
per-chunk/per-step/pointwise table if that's what explains *why*.>**

`tests/test_<slug>.py` locks in the finding as a regression check
(<accuracy bar, or `> X` failure-signature bound if there's no bar to
clear>).

**Leads for whoever picks this up next:**
1. ...

---
Where this file goes: if this issue is a follow-up to (or targets the same
underlying question as) an existing thread folder under `experiments/`, add
it there. If it opens a genuinely new question, add it as a top-level file
here and only promote it into its own thread folder once a second
experiment actually follows up on it. Then:
- Add a row to the relevant thread table (or the "Standalone" table) in
  `../CLAUDE.md`'s experiment index, plus update the "Open leads" section.
- Append this experiment's numbers to `../results.csv` (one row per
  issue/variant/seed/metric datapoint).
- Add/update rows in `../LITERATURE.md` for any paper cited.

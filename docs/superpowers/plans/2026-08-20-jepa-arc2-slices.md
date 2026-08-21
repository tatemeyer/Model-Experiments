# jepa Arc 2 implementation plan — the slices

**Date:** 2026-08-20
**Status:** planned
**Design:** `../specs/2026-08-20-jepa-arc2-scale-and-duration-design.md` (approved, merged in #110)
**Project:** `projects/jepa`

## The question every slice moves toward

> **Are masking ratio and predictor depth inert, or is 3000 steps too few for
> them to show?**

Arc 1 cannot separate those two claims. This arc exists to separate them, and
that is the standard each slice is planned against. Every task below states
what it contributes to answering it; a task that cannot state one does not
belong in this arc.

The trap to avoid is a slice that is *interesting* but does not move the
question — "how does rank evolve with training length" is a fine study and, on
its own, still leaves the fork open. Slice 1 is scoped so its outputs are the
**inputs Slice 2 needs** (the duration `S`, and the residual sd that sets the
seed budget), not just a curve worth looking at.

## Global constraints

Carried from the design, restated here because they bind every task:

1. **`probe_r2` is not collected.** Not as a secondary metric, not "for
   completeness," not as a sanity check (#104: saturated, ~0.001 headroom).
2. **`effective_rank` is the primary metric**, measured by
   `harness.collapse_metrics` on held-out frames. Ceiling is **32**
   (per-patch embeddings, `EMBED_DIM`), not 2048.
3. **Momentum fixed at 0.0**, `use_ema=True`. Arc 1 settled momentum; this arc
   spends no compute re-testing it.
4. **Seed is a pre-registered blocking factor** in every analysis, chosen
   before seeing an F, never after.
5. **Thresholds are stated against a measured noise scale**, never an absolute
   rank — torch-version drift is ~0.06 here and a tight bound breaks
   spuriously.
6. **Convergence gates interpretation.** A cell whose runs have not converged
   is reported *not converged*, never *no effect*.

---

## Slice 1 — the duration curve, and the numbers Slice 2 needs

**Contributes to the question:** produces `S` (the duration at which
`effective_rank` saturates, or the finding that it does not within budget) and
the residual sd of the metric *at long duration*. Slice 2 cannot be sized
without both. It also settles whether the climb is duration or data, which
would otherwise confound every later claim about "training longer."

### Task 1.1 — File the Intent issue — **done, #113**

Labels `intent`, `autonomy:review`, `project:jepa`, matching #97/#107.
`autonomy:review` rather than `safe`: the slice commits several hours of
compute and lands a research conclusion, which is the class this repo reviews.

### Task 1.2 — A parallel sweep runner

Arc 1 ran its sweeps sequentially. Measured on this desktop: **63.3 ms/step**
alone, **~57 ms/step each with 6 concurrent** — near-linear to 6 workers, a
~5.2× throughput gain for no methodological cost. Arc 2's budget assumes it.

Two implementation hazards, both specific and both silent if missed:

- **Every worker must call `torch.set_num_threads(1)` itself.** It is
  per-process state and Windows uses `spawn`, so children do **not** inherit
  it. Without it, 6 workers each spin up 8 intra-op threads on 8 cores and the
  pool runs *slower* than sequential. The measured 57 ms/step above was taken
  with 6 independent processes each setting it.
- **`ProcessPoolExecutor` needs a `main()` guard** and picklable top-level
  callables, or the spawn re-imports the module and forks recursively.

Keep the pool **inside the slice's sweep module** for now. Do not promote it to
`tools/` yet — root `CLAUDE.md` principle 2 says build the tool when the need
recurs, and the precedent for the timing is Arc 1's own: `harness.py` was
promoted out of a test file at Slice 2, when a second caller actually appeared.
Slice 2 is that second caller; promote it there.

### Task 1.3 — The sweep

- momentum 0.0, `predictor_depth = 2`, default masking — the values every prior
  slice ran at, so rows join the existing record rather than sitting beside it.
- `pool_size ∈ {512, 4096}` — the confound control. At `POOL_SIZE=512` /
  `BATCH_SIZE=32`, 6000 steps is already ~375 passes over 512 frames and 48,000
  steps is ~3000. Without this axis the slice cannot say whether it measured
  "trains longer" or "revisits a small pool harder."
- seeds `0..11`.
- one run per `(pool_size, seed)` to **48,000 steps**, `checkpoint_steps =
  (300, 1000, 3000, 6000, 12000, 24000, 48000)`.
- controls at n = 12, **re-measured not inherited**: `random_init`, and
  `use_ema=False` at 48,000 steps.

`history.append` runs before `on_checkpoint(step + 1, encoder)` fires
(`train.py:141` / `:144`), so at each checkpoint the history holds exactly
`step + 1` losses and the loss slope is computable inside the callback. Record
`effective_rank`, `embedding_std`, final loss and loss slope **per checkpoint**,
not just at the end.

The checkpoint hook already saves and restores global RNG state (#97), so
measuring does not perturb training — do not re-derive that, and do not remove
the guard.

Resumable, per Arc 1's precedent: flush each row as it completes, skip any
`(pool_size, seed, checkpoint)` already recorded. At ~3.5 h a sweep that
restarts from zero on an interruption is a slice that does not land.

### Task 1.4 — Run it

~28 runs at 48,000 steps ≈ **3.5 h** wall-clock at 6-way concurrency.

### Task 1.5 — Analysis

Three outputs, all of which Slice 2 consumes:

1. **`S`** — where the curve flattens, or an explicit "still climbing at 48,000
   steps, no inflection." The second is a real answer, not a failed slice; it
   bounds every later claim.
2. **Residual sd of `effective_rank` at long duration.** May differ from the
   0.388 measured at 3000 steps. This sets Slice 2's seed budget.
3. **The random-init band at n = 12**, replacing the n = 3 band (2.44–2.93)
   that Arc 1's entire conclusion is stated against.

Also report where the saturation value sits **relative to the ceiling of 32** —
that single number decides whether Slice 3 is optional or mandatory.

### Task 1.6 — Regression test

`effective_rank` at 12,000 steps exceeds the measured random-init band by more
than the measured seed-noise scale. Stated comparatively, per Slice 3's
precedent, so torch drift cannot break it.

Plus a plumbing test that **`pool_size` actually reaches training** — a
silently ignored parameter is exactly the failure mode that makes a real effect
look like a null, which is the mistake this whole arc is correcting for.

### Task 1.7 — Write-up and records

`experiments/005-*.md`, rows appended to `results.csv`, experiment index and
open leads updated in `projects/jepa/CLAUDE.md`.

---

## Slice 2 — the load-bearing slice

**Contributes to the question:** answers it. Everything else in this arc is
either an input to this slice or a consequence of it.

### Task 2.1 — File the Intent issue

**Blocked on Slice 1** — it cannot state its own duration or seed count until
Slice 1 measures them. File it after Slice 1 lands, with `S` and the sd filled
in, rather than filing it now with placeholders.

### Task 2.2 — Promote the parallel runner to `tools/`

Second caller, so the need has recurred. Same trigger that promoted
`harness.py` at Arc 1 Slice 2. Verify behaviour-preserving by reproducing a
Slice 1 row exactly, as that promotion did.

### Task 2.3 — The grid

- depth `{1, 2, 4}` × masking `{light, default, heavy}` — **unchanged from
  Slice 3**, so the comparison is direct rather than approximate.
- seeds sized from Slice 1's measured sd for **MDD ≤ 0.20**. At the currently
  measured sd of 0.388 that is 12 seeds (n = 36 per marginal level, 108 runs).
- trained to `S`, checkpointed on the same ladder.

**The checkpointing is the mechanism that separates the two claims**, and it is
the single most important line in this plan. The same runs yield the grid at
3000 steps *and* at `S`. One compute spend, two answers:

1. a 12-seed replication of Slice 3's 3000-step null, and
2. the same grid at convergence.

Without the checkpoints this slice would answer only half the fork.

### Task 2.4 — Pre-registered analysis

Write the analysis **before** running the grid, so the decision rule cannot be
chosen after seeing the numbers:

> An axis has an effect only if the randomized-block F (seed as block) exceeds
> its α = 0.05 critical value **and** the marginal spread exceeds the slice's
> own measured MDD. Cell-table pattern-reading is not evidence.

Report the four-outcome table from the design (null/null, null/effect,
effect/null, effect/effect) and state which one landed.

Convergence is an inclusion criterion: record terminal loss slope per cell, and
report a non-converged cell as *not converged*.

### Task 2.5 — Consequences, which are not optional

If the outcome is **null at 3000 / effect at `S`**, then Arc 1's conclusion
table in `projects/jepa/CLAUDE.md` is wrong as written and must gain "at 3000
steps." Updating it is part of this slice, not a follow-up — the table is
inherited by every later arc, and leaving it unqualified while knowing better is
the failure this arc was created to prevent.

If the outcome is **null at both**, say so plainly: the levers are inert, Arc 1
stands unqualified, and the arc has bought certainty rather than a new finding.
That is a successful slice.

---

## Slice 3 — model scale, conditional

**Contributes to the question:** decides whether every "no effect" in this
project is potentially a ceiling effect. If it is, no lever conclusion —
including Arc 1's — can be trusted as stated.

**Do not file this issue until Slice 1 reports.** The gate is a measured value,
not a preference:

| Slice 1 finds | Slice 3 |
|---|---|
| saturation **well below 32** | optional — capacity is not binding, model width is the natural next axis |
| saturation **near 32** | **mandatory** — the metric is capacity-limited and every null in this project is suspect |

Unlike Slices 1–2, this one needs a code change first: `EMBED_DIM` and
`HIDDEN_DIM` are module-level constants, not `train_jepa` parameters. Budget for
that, and for the bit-for-bit no-op proof that Arc 1 Slice 3 established as the
standing rule when a swept parameter is introduced.

---

## Cost

| slice | runs | steps | wall-clock at 6-way |
|---|---|---|---|
| 1 | ~28 | 48,000 | ~3.5 h |
| 2 | 108 | 24,000 | ~7 h |
| 2 if `S` = 48,000 | 108 | 48,000 | ~14 h |

Overnight jobs. Roughly 3–5× Arc 1's total compute.

## Sequencing

```
Slice 1  ──►  S, residual sd, random-init band at n=12
              │                    │
              ▼                    ▼
         Slice 2 (sized)      Slice 3 gate (optional / mandatory)
              │
              ▼
    Arc 1's conclusion table: qualified, or confirmed
```

Slice 1 is unblocked and filed as **#113**. Slice 2 and Slice 3 are blocked on
its outputs and their issues are deliberately **not** filed yet — an Intent
issue with a placeholder where its success criterion should be is the
`needs-intent` case this repo already has a label for. Slice 2 cannot state its
duration or its seed count until Slice 1 measures them; Slice 3 cannot state
whether it is optional or mandatory until Slice 1 says where the metric tops out
against its ceiling.

## Note on the metrics feed (#112)

Arc 2's rows land in `results.csv` like every prior slice. If #112 resolves by
publishing a derived long-format JSONL projection of that file, these rows reach
the cockpit with no extra work in this arc — and the grouped-with-spread shape
is exactly what a duration curve with 12 seeds per point needs in order to be
legible. Nothing in this plan depends on that landing first.

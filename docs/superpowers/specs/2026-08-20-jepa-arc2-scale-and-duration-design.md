# jepa Arc 2: scale and duration — is Arc 1's null a property of the levers, or of the budget?

**Date:** 2026-08-20
**Status:** proposed
**Project:** `projects/jepa`
**Arc:** 2 — scale and duration (proposed; renumbers the previous Arc 2–4 backlog, see below)
**Follows:** Arc 1, closed by issue #107 (`experiments/004-masking-ratio-predictor-depth.md`)

## Why this exists

Arc 1 asked which of three architectural levers prevents representation
collapse and answered cleanly: **only stop-gradient is a lever at all.** EMA
momentum is flat across four orders of magnitude, masking ratio does nothing to
the collapse metric, predictor depth does nothing. That finding stands and this
arc does not re-litigate it.

But Arc 1 also produced a second result it was not designed to produce, and it
is the larger one:

| configuration | `effective_rank` |
|---|---|
| no stop-gradient (`use_ema=False`) | 1.25 – 1.46 |
| **Slice 3's entire 27-run grid, 3000 steps** | **1.94 – 3.44** |
| untrained random-init encoder | 2.44 – 2.93 |
| Slice 2's momentum plateau, 6000 steps | 3.15 – 4.03 |

Every cell of the grid that closed Arc 1 sits *inside the untrained band*.
Doubling the step count clears it. **Step count moved the metric further than
any lever Arc 1 set out to study**, and it was never the object of study — only
a checkpoint ladder underneath a momentum sweep.

This arc is built on that, not on the null.

## The question Arc 1 genuinely cannot answer

"Masking ratio and predictor depth are inert" and "3000 steps is too few for
masking ratio and predictor depth to show" are different claims with different
consequences, and Slice 3's design cannot separate them. Two independent
reasons, both measurable rather than rhetorical:

### 1. The runs had not converged

Slice 3's own stability table: at heavy masking the loss was **still rising in
9 of 9 runs** at the 3000-step cutoff. A null measured on models that were
still moving is a null about the cutoff as much as about the knob.

### 2. The design's detection floor was larger than any effect it could have seen

Re-analysing Slice 3's 27 recorded `effective_rank` values (no new compute —
`results.csv`, issue 107 rows):

| analysis | F | df |
|---|---|---|
| one-way over 9 cells, as published | **0.674** | 8, 18 |
| randomized-block, seed pulled out as a blocking factor | **0.804** | 8, 16 |

Seed is shared across all nine cells (`torch.manual_seed(seed)` before model
construction), so it is a genuine block, and blocking is the standard fix for
exactly the failure Slice 3 hit — seed variance swamping cell variance. It
accounts for **0.823 of the 3.227 within-cell sum of squares (26%)**, and
removing it drops the residual sd from 0.423 to **0.388**.

**The verdict does not change: F is still below 1.** That is worth stating
plainly, because it is the outcome that makes this arc necessary rather than
optional — Slice 3's null is *not* a statistics artifact, and cannot be argued
away by re-analysis. It has to be attacked with compute.

What the re-analysis does give is the number this arc needs. With residual
sd 0.388 and n = 9 per marginal level, the smallest difference between two
marginal means Slice 3 could have resolved was:

> **MDD ≈ 0.39 `effective_rank`** (α = 0.05, two-sided)

against observed marginal spreads of **0.117** (depth) and **0.267** (masking)
— both *below the detection floor*. Slice 3 could not have detected its own
largest observed effect.

And for scale: the 3000 → 6000 duration effect at momentum 0.0 is **+0.896**,
about **2.3× that floor**. So Slice 3's design had the power to see the
duration effect and not the lever effects. That asymmetry is the whole arc in
one line.

### Neither reason is a criticism of Slice 3

Slice 3 was correct to report F < 1 and to record the interaction as *not
supported* rather than as a lead. The failure would have been reading the cell
table. This arc exists because that honest null left a fork, not because the
null was wrong.

## Scope, and the renumbering

**Proposed: this becomes Arc 2, and the existing Arc 2–4 backlog shifts to
3–5.**

| was | becomes |
|---|---|
| — | **2 — Scale and duration** (this) |
| 2 — Latent vs. pixel-space prediction | 3 |
| 3 — Stochastic futures (MoP-JEPA-style) | 4 |
| 4 — Toy world-model planning | 5 |

Three reasons, in order of weight:

1. **The old Arc 2 is blocked and this one is not.** `CLAUDE.md`'s open leads
   already record it: Arc 2 is specified around downstream-probe
   sample-efficiency, and issue #104 showed the probe is saturated — an
   untrained encoder scores x/y R² = 0.9998, total headroom ~0.001. It *cannot
   produce a meaningful result* until the probe task is redesigned. Scheduling
   a blocked arc ahead of an unblocked one is how a project stalls.
2. **This arc gates the interpretation of Arc 1's conclusions.** If the levers
   do show up at convergence, the Arc 1 conclusion table in
   `projects/jepa/CLAUDE.md` needs qualifying — "no effect" becomes "no effect
   at 3000 steps." Every later arc inherits that table. Resolving it first is
   cheaper than retrofitting a caveat through three arcs of write-ups.
3. **Renumbering costs nothing here.** Arcs 2–4 are explicitly labelled backlog
   "to be detailed once Arc 1 produces a mechanism finding worth building on."
   None has an issue, a spec, or a line of code. Arc 1 has now produced that
   finding; this is the roadmap doing what it said it would.

Rejected alternative: calling it "Arc 1.5" to avoid churn. It reads as a
correction to Arc 1, which is the wrong frame — Arc 1's answer is not in
question, and duration is a first-class research axis, not an erratum.

## What Arc 1's finding contributes to the design

This arc *uses* Arc 1's result rather than re-testing it:

- **Momentum is fixed at 0.0** — the best cell measured, and pure stop-gradient
  with zero smoothing. Arc 1 earned the right to stop varying it.
- **`use_ema=True` throughout.** Stop-gradient is the mechanism; removing it is
  a known-collapse control, kept as a reference line, not an axis.
- **`effective_rank` remains the primary metric**, per Arc 1's finding that
  probe R² is not a collapse detector here (`no_ema` collapses to 1.25–1.46 and
  still probes at 0.977).

## Hard constraint carried forward: `probe_r2` stays out

Issue #104 established that `probe_r2` is saturated on this task — an untrained
encoder scores 0.9767 against a ~0.978 ceiling, ~97.8% of target variance is
position, velocity is structurally unrecoverable from single-frame samples, and
total headroom is ~0.001. **It is not evidence and no slice in this arc
collects it.** Not as a secondary metric, not "for completeness," not as a
sanity check. It would produce a column of 0.977s that invites exactly the
pattern-reading Slice 3 correctly refused.

The probe task's redesign is a prerequisite of Arc 3 (formerly 2) and belongs
there, where it blocks something.

## The headroom this arc is exploring

Worth stating because it bounds every slice: `collapse_metrics` measures
`effective_rank` over **per-patch** embeddings, shape (300 frames × 64 patches,
`EMBED_DIM`) = (19200, **32**). The ceiling is therefore **32**, not the 2048 a
flattened view would suggest.

The healthy plateau at 6000 steps is 3.15–4.03 — about **12% of the available
dimensionality**, on a curve (1.649 → 2.054 → 2.933 → 3.829 at
300/1000/3000/6000) with no sign of flattening. There is a great deal of room
above, and nothing yet measured says where the top is.

Evaluation frames are drawn held-out (`PROBE_TEST_SEED_OFFSET = 20_000`), so a
rising rank is not training-set leakage. It could still be an encoder
specialising to a 512-frame pool, which Slice 1 controls for directly.

---

# Slices

## Slice 1 — Where does `effective_rank` stop climbing, and is it duration or data?

**Question.** Trained to 8× Slice 3's budget, does `effective_rank` saturate?
At what step count, and at what value? And is the climb driven by *training
longer* or by *seeing more distinct frames*?

**Why the second half is not optional.** `POOL_SIZE = 512` and
`BATCH_SIZE = 32`: at 6000 steps the model has already drawn 192,000 samples
from 512 distinct frames — roughly 375 passes. At 48,000 steps it is ~3000
passes. "Train longer" and "revisit a small pool harder" are the same
manipulation at this pool size, and an arc named *scale and duration* that
measures only one of them cannot say which it measured. Frames are generated
procedurally, so the fix is nearly free.

**Design.**

- Configuration: momentum 0.0, `predictor_depth = 2`, default masking — the
  defaults every prior slice ran at, so numbers join the existing record.
- `pool_size` ∈ **{512, 4096}**.
- Seeds: **12** (see "Seed count" below).
- One run per (pool, seed) to **48,000 steps**, with `checkpoint_steps =
  {300, 1000, 3000, 6000, 12000, 24000, 48000}`. The hook already exists
  (`train_jepa(checkpoint_steps=, on_checkpoint=)`, issue #97) and already
  saves/restores global RNG state, so measuring does not perturb training. The
  first four checkpoints reproduce prior slices' step points exactly.
- Controls, both re-measured at **n = 12** rather than inherited at n = 3:
  random-init, and `use_ema=False` at 48,000 steps.

**Why re-measure the controls.** The random-init band (2.44–2.93) is the
reference line the whole Arc 1 conclusion leans on, and it rests on three
seeds. If Slice 1's central claim is "the trained model clears the untrained
band," the band needs the same precision as the thing being compared to it.

**Outputs Slice 2 depends on:** the saturation step count **S**, and the
residual sd of `effective_rank` **at long duration** — which may not be the
0.388 measured at 3000 steps, and determines Slice 2's seed budget.

**Falsifiable in advance.** If `effective_rank` is still climbing at 48,000
steps with no inflection, the arc says so and reports the curve. "No saturation
within budget" is a real answer that constrains every later slice; it is not a
failure of the slice.

## Slice 2 — Do the Arc 1 levers show up once the runs have converged?

This is the slice the arc exists for. It answers the fork directly.

**Design.** Slice 3's grid, re-run at duration, with power:

- depth ∈ {1, 2, 4} × masking ∈ {light, default, heavy} — **unchanged**, so
  results are directly comparable rather than approximately comparable.
- Seeds sized from Slice 1's measured residual sd to reach a **pre-registered
  MDD ≤ 0.20** on a pairwise marginal comparison. At the currently measured sd
  of 0.388 that is **12 seeds per cell** (n = 36 per marginal level), i.e. 108
  runs. If Slice 1 finds the sd larger at long duration, the seed count rises
  with it and the cost section below scales linearly.
- Trained to **S** from Slice 1 (or to the affordable maximum if there is no
  saturation), checkpointed at the same ladder.

**The checkpointing is what separates the two claims.** The same runs yield the
grid at 3000 steps *and* at S. So the slice produces, from one compute spend:

1. a **powered replication of Slice 3's 3000-step null** (12 seeds where Slice 3
   had 3), and
2. the same grid at convergence.

The four outcomes and what each means:

| 3000 steps | at S | conclusion |
|---|---|---|
| null | null | **The levers are inert.** Arc 1's table stands unqualified. |
| null | effect | **3000 steps was too few.** Arc 1's table gains "at 3000 steps"; Slice 3's null is a budget artifact. |
| effect | null | Slice 3 was underpowered *and* the effect washes out with training. Report both. |
| effect | effect | Slice 3 was underpowered, full stop. The re-analysis above argues against this, and it is the outcome that would most change how this project runs sweeps. |

**Pre-registered decision rule**, so the grid cannot be read after the fact the
way Slice 3's cell table invited:

> An axis is declared to have an effect only if the randomized-block F (seed as
> block) exceeds the α = 0.05 critical value **and** the marginal spread exceeds
> the slice's own measured MDD. Cell-table pattern-reading is not evidence, per
> Slice 3.

**Convergence is an inclusion criterion, not a footnote.** Every run records
its terminal loss slope. A cell whose runs have not converged is reported as
*not converged* rather than as *no effect* — the precise distinction Slice 3
could not make.

## Slice 3 (conditional) — model scale

`EMBED_DIM = 32` / `HIDDEN_DIM = 32` are module-level constants, not
`train_jepa` parameters, so unlike Slices 1–2 this one needs a code change
before it can run — a real cost difference worth naming up front.

**Whether it runs at all is decided by Slice 1's number, not by preference:**

- If `effective_rank` saturates **well below 32** — capacity is not what binds,
  the metric has headroom it isn't using, and model width is the natural next
  axis. Optional, worth doing.
- If it saturates **near 32** — the metric is capacity-limited, and every "no
  effect" in this project is potentially a ceiling effect rather than a null.
  Slice 3 becomes **mandatory** before any lever conclusion can be trusted,
  including Arc 1's.

Making the slice conditional on a measured value rather than scheduling it
blind is the point; the interesting case is the one where it is forced.

---

## The stability thread: inside this arc, not beside it

Slice 3 found a clean, monotone, nearly unanimous effect of masking ratio on
training stability that never reached the collapse metric:

| masking | realized fraction | mean loss slope | runs with loss rising |
|---|---|---|---|
| light | 0.258 | −0.021 | 2 / 9 |
| default | 0.493 | +0.008 | 6 / 9 |
| heavy | 0.702 | **+0.063** | **9 / 9** |

**Recommendation: fold it into Slices 1–2 as instrumentation and a gate. Do not
give it its own arc.**

The argument is that "the loss was still rising in 9/9 heavy runs at 3000
steps" and "3000 steps is too few" are *the same sentence*. The stability
finding is not a parallel thread that happens to share an arc — it is the
direct evidence for this arc's premise, and the sharpest single piece of it.
Splitting it out would mean two efforts asking one question, and the split one
would have to re-run the same long trainings to answer it.

So it becomes:

- **A recorded metric.** Terminal loss slope on every run, every checkpoint.
- **A gate on interpretation.** No cell reports a null unless its runs
  converged. This is what turns Slice 2's null into a real answer instead of a
  repeat of Slice 3's.
- **A first-class Slice 1 output.** "At what step count does heavy masking stop
  rising?" is answered by the same curve as "at what step count does
  `effective_rank` saturate," at zero marginal cost.

**What does *not* belong here, and is parked as a lead:** *why* heavy masking
destabilises — its interaction with learning rate, whether it is task difficulty
or gradient noise, whether an LR schedule removes it. That is an optimisation
question, not a collapse question. It shares only the knob, and smuggling it in
would widen the arc into "everything masking touches." If Slice 2 shows heavy
masking still failing to converge at S, that lead gets promoted on evidence.

---

## Cost — measured on the desktop, not extrapolated

Benchmarked 2026-08-20 on the target machine (i7-9700K, 8 cores,
`torch.set_num_threads(1)` per Slice 2's finding):

| condition | per-step |
|---|---|
| one run alone | 63.3 ms |
| **6 runs concurrent** | **~57 ms each** (6 × 300 steps in 22 s wall) |

Throughput scales essentially linearly to 6 workers — the model is small enough
that per-process parallelism beats intra-op threading, which is the same reason
Slice 2 pinned threads to 1. Independent runs are embarrassingly parallel, so
the sweep runner should use a process pool. **This is the single largest
practical lever in the arc: ~5.2× for no methodological cost.**

| slice | runs | steps | wall-clock at 6-way |
|---|---|---|---|
| Slice 1 (2 pools × 12 seeds + controls) | ~28 | 48,000 | **~3.5 h** |
| Slice 2 (3 × 3 × 12 seeds) | 108 | 24,000 | **~7 h** |
| Slice 2 if S = 48,000 | 108 | 48,000 | ~14 h |

Both are overnight jobs on a machine that is otherwise idle. For comparison,
Arc 1's Slice 2 cost 3.1 h and Slice 3 about 2 h, run sequentially — so this arc
is roughly 3–5× Arc 1's total compute, for the study that determines whether
Arc 1's headline table needs a qualifier.

Sweeps must stay **resumable** (per-row flush, skip completed cells) as both
prior sweeps are — at 7–14 h an interrupted run that restarts from zero is the
difference between a slice landing and not.

## Seed count

Arc 1's most useful methodological lesson was that **n = 3 cannot answer
questions about this metric** — Slice 3's within-cell spread reached 1.333,
comparable to the entire range across all nine configurations.

This arc pre-registers **12 seeds** and derives it rather than picking it:
residual sd 0.388, target MDD 0.20 on a pairwise marginal comparison, α = 0.05
⇒ ~34 observations per marginal level ⇒ 12 seeds in a 3 × 3 grid. The number is
a consequence of the target resolution, and if Slice 1 measures a different sd
at long duration, the seed count moves with it.

Seeds `0–11`, with `0, 1, 2` retained so every prior slice's runs remain a
subset and are directly comparable.

## Known confounds and how each is handled

| confound | handling |
|---|---|
| Duration vs. data-pool size | Explicit `pool_size` axis in Slice 1. |
| "Healthy rank" from never having trained (Arc 1 Slice 2's confound) | Loss and loss slope recorded on every run; a flat-loss cell is not a healthy cell. |
| Metric ceiling at 32 | Slice 1 measures where the curve tops out; Slice 3 becomes mandatory if it approaches the ceiling. |
| torch-version drift | Regression thresholds stated against the **measured seed-noise scale**, never an absolute rank — the practice Slice 3 established. Drift is ~0.06; seed spread is ~0.4–0.75. |
| Seed variance swamping the signal | Seed is a blocking factor in every analysis in this arc, and the block is pre-registered rather than chosen after seeing the F. |
| Reading the cell table | Pre-registered decision rule above. |

## What this arc does not do

- It does not re-open momentum. Arc 1 settled it.
- It does not collect `probe_r2` (#104).
- It does not touch the probe task. That is Arc 3's prerequisite.
- It does not add architecture — no new encoder, no multi-frame inputs, no
  temporal structure. The only code Slices 1–2 need is a sweep script and a
  process pool; every training parameter they vary already exists on
  `train_jepa`.

## Verification

Per this repo's loop, each slice lands with CI-checkable assertions, following
Slice 3's precedent of asserting claims **comparatively against a measured noise
scale** rather than against absolute thresholds:

- **Slice 1:** a test asserting `effective_rank` at 12,000 steps exceeds the
  random-init band by more than the measured seed-noise scale — the claim
  "training length is what clears the band," stated so a change that breaks it
  fails.
- **Slice 2:** the pre-registered decision rule applied to the recorded grid, so
  whichever of the four outcomes lands is asserted, not narrated.
- Plumbing tests in both, per Slice 3's precedent — that `pool_size` and
  `checkpoint_steps` actually reach training, since a silently ignored parameter
  is the failure mode that makes a real effect look like a null.

## Open question for the human

Slice 2's seed budget (12) and duration (S from Slice 1) set the compute at
roughly 7–14 h. The alternative is a reduced grid — corners only, depth {1, 4} ×
masking {light, heavy} plus the default centre, 5 cells — which buys back ~45%
of the cost. Arc 1's Slice 3 established there is no monotone trend to trace, so
maximum contrast may be worth more than grid resolution.

Recommendation: **keep the full 3 × 3.** The comparability with Slice 3 is the
point of the slice, and a reduced grid answers a slightly different question
than the one Arc 1 left open. But the cost is real and the call is worth making
deliberately rather than by default.

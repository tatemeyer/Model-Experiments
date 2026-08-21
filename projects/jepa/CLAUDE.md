# jepa

Research into Joint-Embedding Predictive Architecture (JEPA) — LeCun's
latent-space self-supervised/world-model architecture family (I-JEPA,
V-JEPA/V-JEPA 2, MoP-JEPA). See `LITERATURE.md` for the papers motivating
this project.

## Core mechanism under study

A JEPA trains an encoder (student) and an EMA/momentum target encoder
(teacher, stop-gradient) on the same input under different views/
masking; a predictor network maps student-encoded context to a
prediction of the teacher-encoded target in *latent* space — never
reconstructing pixels. Representation collapse (everything maps to a
constant) is the central failure mode, prevented only by architectural/
training choices (EMA momentum, predictor capacity/asymmetry, masking
strategy) — there are no contrastive negative pairs doing the work.
That prevention mechanism, not raw benchmark performance, is this
project's first research question.

Compute is constrained by `CONVENTIONS.md`'s "Compute assumption" entry
(CPU + one non-tensor-core consumer GPU + free-tier cloud only), so this
project cannot replicate V-JEPA-2-scale results — it studies the
underlying mechanisms at toy scale instead, in the same spirit as
`projects/em-piml`'s PINN ablations.

## Toy environment

A procedurally generated bouncing-ball video (small grayscale canvas,
deterministic given a seed) is this project's own choice, not drawn from
any paper — it gives an *exact* closed-form ground-truth trajectory to
evaluate against via linear probing (regressing true position/velocity
from frozen embeddings, the standard JEPA evaluation protocol), mirroring
`projects/em-piml`'s closed-form-solution approach instead of eyeballing
qualitative video predictions.

## Research Arc roadmap

Arc 1 is complete. Arc 2 is **proposed, not yet approved** — see
`docs/superpowers/specs/2026-08-20-jepa-arc2-scale-and-duration-design.md`;
it renumbers what were previously Arcs 2-4 to 3-5. Arcs 3-5 remain
backlog.

1. **Arc 1 — Collapse-avoidance mechanism study** (**complete**, issues
   #69/#97/#107 — see the Arc 1 conclusion below). What
   actually prevents representation collapse (EMA momentum, predictor
   depth/width, masking ratio/strategy) in a controlled setting where
   many seeds are cheap to run? Builds the shared encoder/predictor/eval
   infrastructure every later Arc reuses.
   - **Slice 1 — "Baseline JEPA reproduces non-collapsed, probe-able
     representations on bouncing-ball video."** Tasks: A (data
     generator, issue #65), B (encoder/EMA-target/predictor/masking
     components, issue #66), C (training loop, issue #67), D (collapse
     metrics + linear-probe harness, issue #68), E (baseline ablations +
     Slice 1 result, issue #69). A/B/D are unblocked; C blocks on A+B;
     E blocks on A-D.
   - **Slice 2 — "Where is the EMA-momentum collapse boundary?"**
     (issue #97, done). Momentum × steps only. Split out from the
     original three-axis plan because Slice 1 showed the collapse gap is
     invisible at 300 steps, so a momentum sweep at a fixed step count
     would measure that choice rather than momentum. Answer: the boundary
     is at the *high*-momentum end, and **stop-gradient — not EMA
     smoothing — is what prevents collapse**. See
     `experiments/002-momentum-collapse-boundary.md`.
   - **Slice 3 — "Do masking ratio or predictor depth move the collapse
     boundary?"** (issue #107, done). The two axes deferred out of Slice
     2. Answer: **neither, at all.** Across depth {1,2,4} × realized
     masked fraction {0.26, 0.49, 0.70} × 3 seeds, between-configuration
     variation in `effective_rank` is *smaller* than seed-to-seed
     variation (F = 0.674). Masking ratio does affect training stability
     — at the heavy end the loss was still rising in 9/9 runs — but that
     never reaches the collapse metric. See
     `experiments/004-masking-ratio-predictor-depth.md`.

   **Arc 1 conclusion — collapse avoidance here is essentially binary.**
   Arc 1 asked which of three levers prevents representation collapse.
   Having swept all three, the answer is that **only one of them is a
   lever at all**:

   - **Stop-gradient — decisive.** Removing it (`use_ema=False`, a
     target trained by ordinary gradient descent) is the only
     manipulation in the whole Arc that reliably collapses the
     representation: `effective_rank` 1.25–1.46, far below everything
     else measured, and below an untrained encoder.
   - **EMA momentum — irrelevant across four orders of magnitude**
     (0.0 → 0.999 is a flat healthy plateau). Momentum **0.0** — pure
     stop-gradient with zero smoothing — is the best cell measured. Only
     the degenerate extreme 0.9999, where the target barely moves and
     the prediction task itself degenerates, fails.
   - **Masking ratio — no effect** on collapse (affects optimization
     stability only).
   - **Predictor depth — no effect.**

   So the mechanism is the *stop-gradient asymmetry*, not the momentum
   schedule, the masking design, or predictor capacity. This
   independently reproduces SimSiam's central claim (Chen & He,
   arXiv:2011.10566 — stop-gradient is essential, a momentum encoder is
   not) on a different architecture and a different task, and it is the
   one finding Arcs 2–4 should build on.

   **The dominant variable was one Arc 1 never named: training length.**
   Every cell of Slice 3's grid at 3000 steps overlaps the *untrained*
   random-init band (2.44–2.93); Slice 2's plateau at 6000 steps clears
   it (3.15–4.03). Step count moved `effective_rank` further than any
   architectural lever the Arc set out to study — see Slice 3's leads.
2. **Arc 2 — Scale and duration** (**proposed**, design at
   `docs/superpowers/specs/2026-08-20-jepa-arc2-scale-and-duration-design.md`).
   Arc 1's dominant variable was one it never named: every cell of the
   3000-step Slice 3 grid overlaps the *untrained* random-init band,
   while Slice 2's 6000-step runs clear it. So Arc 1 leaves one question
   genuinely open — **are masking ratio and predictor depth inert, or is
   3000 steps too few for them to show?** Those are different claims and
   Slice 3's design cannot separate them: its runs had not converged
   (heavy masking, loss still rising 9/9), and its detection floor
   (MDD ≈ 0.39 `effective_rank`) was larger than either effect it could
   have observed. Re-analysing its 27 rows with seed as a blocking factor
   raises F only 0.674 → 0.804, still below 1 — the null is not a
   statistics artifact and has to be attacked with compute. Three slices:
   the duration curve (with `pool_size` controlled, since 512 frames ×
   3000 passes confounds "trains longer" with "revisits a small pool"),
   the powered re-run of Slice 3's grid at convergence, and a model-scale
   slice made conditional on where the metric tops out against its
   ceiling of 32. The masking/stability thread folds in here as a
   convergence gate rather than becoming its own arc — "loss still rising
   at 3000 steps" and "3000 steps is too few" are the same sentence.
3. **Arc 3 — Latent vs. pixel-space prediction.** Does latent-space
   prediction beat a pixel-space autoencoder baseline and a contrastive
   (SimCLR-style) baseline on sample-efficiency of the downstream probe,
   holding encoder capacity fixed? **Blocked** on the probe-task redesign
   (see open leads) — issue #104 left ~0.001 of headroom, so this arc
   cannot measure anything until that is fixed. That block is the main
   reason Arc 2 above goes first.
4. **Arc 4 — Stochastic futures (MoP-JEPA-style).** On a toy environment
   with genuine multimodality (e.g. a ball reaching a fork with two
   valid continuations), does a mixture-of-predictors resolve the
   blurry/averaged single-predictor failure MoP-JEPA reports?
5. **Arc 5 — Toy world-model planning.** Can predictor-rollout energy
   minimization do short-horizon planning (e.g. "reach target region")
   in the toy environment, echoing V-JEPA 2's zero-shot robot planning
   at toy scale?

## Where to find things

This file stays a short, stable router (per `CONVENTIONS.md`'s "Project
experiment logs" entry) — it does not grow a new section per issue.

- **`experiments/<thread>/NNN-slug.md`** (or a top-level
  `experiments/NNN-slug.md` for a standalone question) — the full
  write-up for each completed experiment. `experiments/TEMPLATE.md` is
  the skeleton for the next one.
- **`results.csv`** — every experiment's numeric results in tidy long
  format: one row per `(issue, variant, seed, metric)` datapoint.
- **`LITERATURE.md`** — every paper this project has cited or tested,
  with a verdict. Check here before proposing a paper as a "new" lead.

## Experiment index

Verdict key: ✅ helped · ⚠️ partial/modest · ❌ no effect · 🔻 actively worse.

**Slice 1 — baseline**

| issue | question | verdict | where |
|---|---|---|---|
| #69 | Does the full EMA-target JEPA avoid collapse (vs. a no-EMA ablation) and out-probe a random-init baseline? | ⚠️ partial — collapse-avoidance confirmed via `effective_rank` (not `embedding_std`), probe-R² hypothesis not confirmed. **Probe numbers superseded by #104**; the collapse half is unaffected | `experiments/001-baseline-collapse-avoidance.md` |

**Methodology corrections**

| issue | question | verdict | where |
|---|---|---|---|
| #104 | Were this project's probe-R² numbers measuring the encoder, or the solver? | 🔻 the solver — `lstsq`'s float32 rank cut discarded most of the signal, non-reproducibly. Corrected: all variants tie at 0.9763–0.9773 because **the probe is saturated (untrained encoder already at x/y R²=0.9998) and ~97.8% position by variance — total headroom ~0.001** | `experiments/003-probe-solver-correctness.md` |

**Slice 2 — collapse boundary**

| issue | question | verdict | where |
|---|---|---|---|
| #97 | Where is the EMA-momentum collapse boundary, and how does it move with training length? | ✅ answered — no boundary at the low end (momentum 0.0–0.999 is a flat healthy plateau, 3.15–4.03); collapse only at 0.9999, *below* random init. **Stop-gradient, not EMA smoothing, is the mechanism.** | `experiments/002-momentum-collapse-boundary.md` |

**Slice 3 — masking ratio & predictor depth**

| issue | question | verdict | where |
|---|---|---|---|
| #107 | Do masking ratio or predictor depth move the collapse boundary? | ❌ neither — across depth {1,2,4} × masked fraction {0.26,0.49,0.70} × 3 seeds, between-configuration variation is *smaller* than seed noise (**F = 0.674**). Masking ratio affects training stability only (heavy: loss still rising in 9/9 runs). **Step count outweighed both axes.** | `experiments/004-masking-ratio-predictor-depth.md` |

## Open leads

- ~~Slice 2 should vary `ema_momentum` and `steps` jointly — issue #69
  found the collapse-avoidance gap needs ~3000 steps to appear at
  momentum 0.996.~~ Done in issue #97: swept momentum
  `(0.0, 0.9, 0.99, 0.996, 0.999, 0.9999)` × 3 seeds × checkpoints at
  300/1000/3000/6000. Momentum turned out to be near-irrelevant across
  four orders of magnitude — momentum **0.0** (stop-gradient, zero
  smoothing) is the *best* cell, while `use_ema=False` (no stop-gradient)
  is the worst. See `002-momentum-collapse-boundary.md`.
- **Momentum schedules are now a different question than they looked.**
  BYOL-style ramps were the obvious follow-up, but since the plateau is
  broad and flat and only the extreme high end collapses, a schedule
  ramping *toward* 0.9999 would ramp toward the one configuration that
  fails. The open question is whether any schedule beats a flat 0.0.
- The interval between momentum 0.999 and 0.9999 is uncharted — the grid
  jumps an order of magnitude in target staleness across the only
  interval where behaviour changes, so whether the transition is sharp or
  gradual is unknown.
- ~~`linear_probe_r2` is **nondeterministic** on this data.~~ Fixed in
  issue #104. The nondeterminism was the mild symptom: `lstsq`'s default
  driver was rank-truncating a float32 ill-conditioned system and
  returning fits 15–40× worse than attainable on their own objective. The
  probe now runs in float64 with an explicitly regularized (ridge) fit.
  See `003-probe-solver-correctness.md`.
- **The probe task itself is now the blocker, not the probe code.**
  Pooled probe R² is ~97.8% position by target variance, and an
  *untrained* encoder already scores x/y R² = 0.9998, so the whole
  measurable range for a training effect is ~0.001. Velocity is
  structurally unrecoverable (single-frame samples; R² < 0 for every
  variant) and carries only 2.2% of the variance. **Arc 3 (latent vs.
  pixel) is specified around downstream-probe sample-efficiency and
  cannot produce a meaningful result until this is fixed** — needs
  multi-frame inputs,
  per-dimension or standardized-target scoring, and a task a random
  projection doesn't already solve.
- Probe R² is **not a collapse detector** here: `no_ema` sits at
  effective_rank 1.25–1.46 (dimensionally collapsed) and still probes at
  0.977, identical to the healthy model. `effective_rank` should stay
  Arc 1's primary metric.
- ~~Issue #69's probe-R² negative result may be an artifact of
  `PatchEncoder`'s shallow architecture — worth re-testing against a
  deeper encoder.~~ Much weaker motivation after #104: the shallow
  encoder isn't what limits the score, the ceiling is. Change the task
  before the architecture.
- **Training length is the untested axis that actually mattered** (issue
  #107). Every cell of Slice 3's 3000-step grid overlaps the *untrained*
  random-init band; Slice 2's 6000-step plateau clears it. Step count
  moved `effective_rank` further than momentum, masking ratio, or
  predictor depth — and it has never been studied as the primary
  variable, only as a checkpoint ladder underneath a momentum sweep. It
  is also cheap. This is the strongest remaining Arc-1-adjacent lead.
- **`effective_rank` needs more than 3 seeds per cell.** Slice 3's
  within-cell seed spread reached 1.333, comparable to the entire range
  across all nine configurations, which is why that grid returned F <
  1. Any future sweep on this metric should budget more seeds per cell
  or it will be unable to answer its own question regardless of what it
  varies. **Quantified while scoping Arc 2** (no new compute, from
  `results.csv`): treating seed as a *blocking* factor rather than
  leaving it in the residual removes 26% of the within-cell sum of
  squares and lifts F from 0.674 to 0.804 — **still below 1**, so the
  null survives the better analysis. The number worth carrying forward is
  the detection floor it implies: residual sd 0.388 at n = 3 gives a
  minimum detectable difference of **≈0.39** between marginal means,
  against observed spreads of 0.117 (depth) and 0.267 (masking). Slice 3
  could not have resolved its own largest observed effect. Reaching
  MDD ≤ 0.20 needs ~12 seeds per cell. Seed should be a *pre-registered*
  block in any future analysis here, not one chosen after seeing F.
- **Regression thresholds here need margin for torch-version drift.**
  Slice 3 found the recorded 2026-08-03 `effective_rank` values no
  longer reproduce exactly under `torch 2.13.0+cu126` (up to +0.06 on
  seed 0); the *unmodified* code reproduces the new numbers, so it is
  the environment, not the code. Seed spread (~0.75) is an order of
  magnitude larger, so no conclusion was affected — but a tight
  threshold would break spuriously.
- **Training length is proposed as Arc 2** now that Arc 1 has landed its
  finding (**stop-gradient is the mechanism; the other two levers are
  inert**) — design at
  `docs/superpowers/specs/2026-08-20-jepa-arc2-scale-and-duration-design.md`,
  awaiting approval. It is scoped around the *finding* (duration
  dominated) rather than the null, and its load-bearing slice is the one
  that separates "the levers are inert" from "3000 steps is too few for
  them to show" — by checkpointing one set of long runs so the same
  compute yields both a powered replication of Slice 3's 3000-step null
  and the same grid at convergence. Arcs 3-5 remain the standing backlog;
  Arc 3 additionally needs the probe-task fix above before it can measure
  anything, which is why it is no longer first.
- A repo-wide spec/plan conventions overhaul is in flight (separate
  effort) that may formalize where Arc/Slice work like this is tracked
  going forward (a `docs/design/` Design→Arc→Slice spec tree already
  exists, used so far only by `projects/em-piml`'s modernization work —
  see that tree's `README.md`). This project intentionally scaffolded
  under today's plain-Issue conventions rather than pre-adopting that
  tree; revisit whether Arc 1 (or a later Arc) should migrate into it
  once that overhaul's direction is settled.

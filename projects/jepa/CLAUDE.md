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

Only Arc 1 is broken down into implementable work so far; Arcs 2-4 are
backlog, to be detailed once Arc 1 produces a mechanism finding worth
building on.

1. **Arc 1 — Collapse-avoidance mechanism study** (in progress). What
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
   - **Slice 2 (not yet filed)** — ablation sweep over EMA momentum /
     masking ratio / predictor depth, once Slice 1's infrastructure
     exists, to actually answer Arc 1's research question.
2. **Arc 2 — Latent vs. pixel-space prediction.** Does latent-space
   prediction beat a pixel-space autoencoder baseline and a contrastive
   (SimCLR-style) baseline on sample-efficiency of the downstream probe,
   holding encoder capacity fixed?
3. **Arc 3 — Stochastic futures (MoP-JEPA-style).** On a toy environment
   with genuine multimodality (e.g. a ball reaching a fork with two
   valid continuations), does a mixture-of-predictors resolve the
   blurry/averaged single-predictor failure MoP-JEPA reports?
4. **Arc 4 — Toy world-model planning.** Can predictor-rollout energy
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
| #69 | Does the full EMA-target JEPA avoid collapse (vs. a no-EMA ablation) and out-probe a random-init baseline? | ⚠️ partial — collapse-avoidance confirmed via `effective_rank` (not `embedding_std`), probe-R² hypothesis not confirmed | `experiments/001-baseline-collapse-avoidance.md` |

## Open leads

- Slice 2 (EMA momentum / masking ratio / predictor depth sweep) should
  vary `ema_momentum` and `steps` jointly — issue #69 found the
  collapse-avoidance gap needs ~3000 steps to appear at momentum 0.996,
  suggesting a momentum schedule tuned to training length (as BYOL uses)
  could reveal it far more cheaply.
- Issue #69's probe-R² negative result (full model doesn't reliably
  out-probe a random-init encoder) may be an artifact of `PatchEncoder`'s
  shallow architecture rather than a real JEPA-training limitation — worth
  re-testing against a deeper encoder before concluding training doesn't
  help position-decodability here.
- Arcs 2-4 above are the standing backlog once Arc 1 lands a finding.
- A repo-wide spec/plan conventions overhaul is in flight (separate
  effort) that may formalize where Arc/Slice work like this is tracked
  going forward (a `docs/design/` Design→Arc→Slice spec tree already
  exists, used so far only by `projects/em-piml`'s modernization work —
  see that tree's `README.md`). This project intentionally scaffolded
  under today's plain-Issue conventions rather than pre-adopting that
  tree; revisit whether Arc 1 (or a later Arc) should migrate into it
  once that overhaul's direction is settled.

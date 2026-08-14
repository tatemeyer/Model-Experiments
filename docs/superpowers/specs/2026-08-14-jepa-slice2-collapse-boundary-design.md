# jepa Arc 1 Slice 2: the EMA-momentum collapse boundary

**Date:** 2026-08-14
**Status:** approved
**Project:** `projects/jepa`
**Arc:** 1 — collapse-avoidance mechanism study
**Follows:** Slice 1 Task E (issue #69, `experiments/001-baseline-collapse-avoidance.md`)

## Why this exists

Slice 1 established that EMA prevents representation collapse at exactly one
operating point: `ema_momentum = 0.996`, 3000 steps. `effective_rank` separated
the full model (2.35–2.79) from the no-EMA ablation (1.25–1.46) with no seed
overlap.

That is a single point on a two-dimensional surface. Slice 2 maps the surface:
**where is the collapse boundary on the momentum axis, and how does it move
with training length?**

The two axes are not independent, which is the reason this slice exists as one
study rather than two. Task E measured the full model's `effective_rank`
climbing 1.62 → 1.97 → 2.78 → 3.61 at 300/1000/3000/6000 steps while the
no-EMA ablation *declined* 1.61 → 1.47 → 1.39 → 1.34. **At 300 steps the two
are indistinguishable.** A momentum sweep at a single fixed step count would
therefore measure an artifact of that choice rather than a property of
momentum. Task E's own lead #1 says exactly this.

### Scope: deliberately narrower than the roadmap

`projects/jepa/CLAUDE.md`'s roadmap describes Slice 2 as a sweep over "EMA
momentum / masking ratio / predictor depth". This spec covers **momentum ×
steps only**. Masking ratio and predictor depth move to a future Slice 3.

Rationale: momentum-versus-training-length is a mechanism question that Task E
produced direct evidence for; masking ratio and predictor depth are
architecture/task knobs. Bundling them would mean a three-axis grid whose
cheapest cells all sit in the ≤300-step regime Task E already showed reveals
nothing. Splitting keeps each slice one PR-sized question. `CLAUDE.md`'s
roadmap is updated to reflect the split.

## What gets measured

**Momentum grid:** `{0.0, 0.9, 0.99, 0.996, 0.999, 0.9999}`.

Chosen to span both degenerate ends, not just the plausible middle:

- **0.0** — the target is the online encoder copied every step: stop-gradient,
  but zero smoothing. Distinct from `use_ema=False`, where the target is a
  separate network trained by gradient descent.
- **0.996** — Task E's operating point, included so the two slices are
  directly comparable.
- **0.9999** — the target barely moves from random initialization.

**Steps as a nested axis, not an independent one.** Each configuration is
trained **once** to 6000 steps, with metrics recorded at checkpoints
`{300, 1000, 3000, 6000}`. This yields the identical momentum × steps surface
at roughly a quarter the compute of retraining per cell, and matches Task E's
step points exactly.

**Seeds:** `{0, 1, 2}` — matching Task E, so numbers are directly comparable.

**Controls, evaluated at the same checkpoints:**
- the `use_ema=False` ablation (3 seeds) — the known-collapsing reference;
- an untrained random-init encoder — the known-untrained reference.

**Metrics:**
- **`effective_rank` on per-patch embeddings — primary.** The only metric Task
  E showed separates variants with no seed overlap.
- `embedding_std` — recorded for continuity. Task E showed it does *not*
  reliably discriminate, so it informs but never decides.
- **No linear-probe R².** Task E showed random-init sometimes scores highest of
  all variants, so including it would invite a wrong conclusion later, not
  merely add noise. Task E's lead #2 (re-testing the probe against a deeper
  encoder) is out of scope here.

## The confound this design must handle

High momentum barely trains. Task E measured **random-init** `effective_rank`
at 2.44–2.93 — *higher* than the trained full model's 2.35–2.79.

So a healthy-looking rank at `momentum = 0.9999` is ambiguous: it could mean
"learned a good representation" or "never moved from initialization," and
`effective_rank` alone cannot distinguish them. Any analysis that reads high
rank as success would draw exactly the wrong conclusion at the high-momentum
end.

**Mitigation:** every cell additionally records **final training loss and the
loss-curve slope** over the last 10% of steps. A cell counts as genuine
collapse-avoidance only if it shows a non-degenerate `effective_rank` **and**
evidence that the loss actually moved. The write-up must report both, and the
random-init control's rank must appear in the results table as the reference
line for "healthy-looking but untrained."

## What gets built

### 1. A checkpoint-evaluation hook in `train_jepa`

`train_jepa()` gains an optional callback fired at specified step counts,
receiving the current step and the live encoder. Everything else about the
signature and behaviour is unchanged.

**Determinism claim to verify, not assume.** `train_jepa` seeds globally once
before model construction, then uses explicit `batch_rng` (a `torch.Generator`)
and `mask_rng` (a `numpy` generator) inside its loop — never torch's global
RNG. A mid-training evaluation that touches the global RNG should therefore not
perturb training. This is a reading of the code, and the implementation plan's
**first task** is a test proving it: a checkpointed run and an unhooked run at
the same seed must produce bit-identical loss curves. If that fails, the sweep
must retrain per step count instead, at 4× the cost.

### 2. `src/jepa/momentum_steps_sweep.py`

An offline exploratory script, following `helmholtz_capacity_sweep.py` and
`point_draw_sweep.py`: **not** part of the default test suite. Sets
`torch.set_num_threads(1)` — measured faster here, not merely conventional
(23.9s vs 39.1s for 300 steps on 12 cores).

### 3. A fast regression test

`tests/test_momentum_collapse_boundary.py`, locking in whatever boundary the
sweep finds, at a reduced step budget. Follows Task E's precedent of a single
combined slow test with an explicit named threshold constant.

### 4. Records

`experiments/002-momentum-collapse-boundary.md` per `experiments/TEMPLATE.md`,
`results.csv` rows (one per momentum/seed/checkpoint/metric datapoint),
`CLAUDE.md` experiment-index row, and the roadmap updated for the Slice 2/3
split.

## Cost

Measured single-threaded: 3000 steps = 264.0s, i.e. **~0.088 s/step**, linear.
(The 300-step measurements — 23.9s and 26.2s across two runs — imply a lower
per-step figure because they amortize fixed startup over fewer steps; the
3000-step number is the one to plan against.) A 6000-step run is therefore
**~8.8 minutes**.

- 6 momenta × 3 seeds = 18 runs
- no-EMA control × 3 seeds = 3 runs
- random-init control = no training required

**21 runs ≈ 3.1 hours**, offline and in the background. Precedent for a sweep
of this scale: em-piml's PirateNets paper-scale config at 767–796 s/seed.

This budget is stated up front deliberately. If it must shrink later, the
reduction gets recorded explicitly in the write-up (issue #46's precedent),
never applied silently.

## Success criteria

1. Determinism verified: a checkpointed run is bit-identical to an unhooked
   run at the same seed.
2. The full sweep runs and `results.csv` carries every
   momentum/seed/checkpoint/metric datapoint.
3. The write-up states where the collapse boundary lies on the momentum axis,
   how it moves with training length, and **explicitly addresses the
   high-momentum confound** using the loss-curve evidence.
4. A regression test locks the finding and passes.
5. `uv run pytest -q` and `uvx ruff check .` clean; CI green.

## Out of scope

- Masking ratio and predictor depth (future Slice 3).
- Linear-probe R² and Task E's lead #2 (deeper encoder).
- Momentum *schedules* (BYOL-style ramps). Task E's lead #1 raises them, but
  this slice measures the fixed-momentum surface first — a schedule is only
  interpretable once the static boundary is known.
- Multi-frame/temporal data. Task E confirmed velocity is structurally
  unrecoverable from these single-frame samples.

## Process note

This spec lives in `docs/superpowers/specs/`, not the
`docs/design/specs/` Design→Arc→Slice tree, because that tree's own `README.md`
scopes it as a bounded exception for the em-piml modernization Design.
`projects/jepa` runs on the plain Intent-Issue → PR loop, so implementation
still needs an Intent issue filed against this spec before work starts.

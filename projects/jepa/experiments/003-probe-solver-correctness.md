# Were this project's probe-R² numbers measuring the encoder, or the solver? (issue #104)

`CLAUDE.md`'s open leads carried a note that `linear_probe_r2` was
*nondeterministic* — `torch.linalg.lstsq` returning 0.9761533737 three times and
0.9763703942 once on bit-identical inputs — flagged as something to address
before trusting future probe work. Investigating it found the nondeterminism was
the mild symptom of a much larger defect: the probe was not solving its own
problem, and had been reporting fabricated numbers for two experiments.

No new paper is cited by this work, so `../LITERATURE.md` is unchanged.

## The defect

`_fit_linear_probe` called `torch.linalg.lstsq(design, targets)` on the flattened
probe design matrix: 4000×2049, float32, condition number ~1.9e8. `lstsq`'s
default CPU driver is `gelsy` (complete orthogonal factorization), which applies
an rcond-based rank cutoff. At float32 precision on a matrix this ill-conditioned,
that cutoff discarded most of the genuine signal.

The giveaway is the *training* residual — the quantity least-squares exists to
minimize. Against `gelsd` (SVD-based) on the identical float32 system, over four
random-init encoders:

| seed | `gelsy` residual (what the probe used) | `gelsd` residual | ratio | `gelsy` R² | `gelsd` R² |
|---|---|---|---|---|---|
| 0 | 1.50e5 | 9.81e3 | 15.3× | 0.6589 | 0.9764 |
| 1 | 3.21e5 | 9.83e3 | 32.7× | 0.3075 | 0.9762 |
| 2 | 3.96e5 | 9.87e3 | 40.1× | 0.1741 | 0.9772 |
| 3 | 3.93e5 | 9.84e3 | 39.9× | 0.1554 | 0.9761 |

A 15–40× worse fit on the objective being minimized is not "a different valid
minimum" — it is a failure to solve the system. And it was unstable: four
identical calls to the old implementation on one system returned **0.363583,
0.443417, 0.612194, 0.368235**.

**The cause is precision, not the algorithm.** Cast the same system to float64
and the two drivers agree to seven figures — 4.937237e3 both — so `gelsy`'s rank
heuristic only misfires when float32 rounding makes a genuinely full-rank matrix
look deficient. Note also that float32 `gelsd`, the "good" driver, still lands at
9.81e3 against that 4.94e3 optimum: precision alone costs a factor of two even
when the rank cut behaves.

## Implementation

`src/jepa/eval.py`'s probe was rewritten (`_design_matrix`, `_ridge_fit`,
`select_probe_alpha`, `linear_probe_r2`):

- **float64 for the whole fit**, cast at the design-matrix boundary. Beyond
  fixing the rank cut, this is what makes the metric reproducible: the float32
  path drifts ~4e-7 across BLAS thread counts (reduction order) — the same order
  as the 0.9761/0.9763 discrepancy that started this — while float64 holds to
  ~1e-12 across 1/2/4/8 threads.
- **Ridge instead of bare OLS**, solving `(XᵀX + αI)w = Xᵀy` with the intercept
  column unpenalized. This removes the dependence on any driver's rank heuristic
  rather than trading one heuristic for another, and is the standard SSL
  linear-probe protocol. `α` is chosen per call by held-out R² on a validation
  slice cut from the **train** split — the test split is never touched by model
  selection.

Ridge is not merely defensive. At `n_train=200` it scores 0.9576 where bare OLS
(`gelsd`, float64) manages 0.6498 — with 2049 features the unregularized fit
overfits badly, which is the *other* reason the original numbers swung around.

## Result: the corrected numbers overturn the evidence for issue #69's probe finding, but not the finding

Re-running issue #69's comparison (3000 steps, seeds 0/1/2) under the corrected
probe:

| variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| full (EMA) | 0.9768 | 0.9766 | 0.9773 |
| no_ema (ablation) | 0.9770 | 0.9766 | 0.9773 |
| random_init | 0.9767 | 0.9763 | 0.9770 |

Every one of the nine cells falls in **0.9763–0.9773** — a total spread of 0.001.
Issue #69 reported these same nine cells as scattered across 0.10–0.98.

So the original conclusion ("the full model does not reliably out-probe a
random-init encoder") survives, but nothing that was said *about* it does. It was
recorded as a noisy, seed-dependent non-ordering; it is actually an exact tie.

## Why the tie: the metric is saturated, and it is a position-only metric

Breaking the pooled R² out per target dimension (seed 0):

| variant | x | y | vx | vy |
|---|---|---|---|---|
| random_init | 0.9998 | 0.9998 | −0.058 | −0.026 |
| full (EMA) | 1.0000 | 1.0000 | −0.055 | −0.031 |
| *share of target variance* | *48.1%* | *49.7%* | *1.1%* | *1.2%* |

Two things fall out, and together they invalidate the metric rather than the
model:

1. **An untrained encoder already recovers position at R² = 0.9998.** Issue
   #69's Johnson–Lindenstrauss/intensity-weighted-centroid argument — that a
   random projection of a single-blob canvas already preserves position
   linearly — was correct all along. It just happened to be arguing from
   corrupted numbers. Training moves x/y from 0.9998 to 1.0000.
2. **Velocity is unrecoverable and nearly weightless.** Both velocity dimensions
   score *below zero* (worse than predicting the mean) for every variant —
   expected, since these are single-frame (`n_frames=1`) samples with no motion
   cue — and they carry only 2.2% of the target variance between them, because
   velocity's raw units are so much smaller than position's on a 32-pixel canvas.

Pooled R² is therefore ~97.8% position by variance weight, and position is
already at ceiling before training starts. **The maximum attainable pooled score
is ~0.978 and an untrained encoder scores 0.9767, so the entire measurable
headroom for any training effect is ~0.001.** This metric cannot express a JEPA
training benefit no matter what the encoder does.

That is the real finding here, and it is a stronger negative result than issue
#69's: the problem was never that JEPA training failed to help
position-decodability. It is that this probe, on this task, has no room to show
whether it does.

`tests/test_eval.py` locks in three properties (all `@pytest.mark.slow`, built on
a real `PatchEncoder` — the defect is invisible on well-conditioned synthetic
data): reproducibility across repeated calls, stability across BLAS thread counts
(< 1e-9), and that the solve path reaches the least-squares optimum (within 1.05×
of an SVD reference at negligible penalty). The end-to-end assertion `R² > 0.9`
fails against the old implementation, which returns 0.36–0.61 on that system
(0.363583/0.443417/0.612194/0.368235 across four identical calls).
`tests/test_baseline_collapse_avoidance.py`'s `PROBE_R2_FLOOR` moves 0.05 → 0.95,
which would have caught this defect on its own.

**Unaffected:** every `effective_rank` and `embedding_std` number in issues #69
and #97, and both collapse findings built on them — those go through
`torch.linalg.svdvals`, never the least-squares path. Issue #97 deliberately did
not use `probe_r2` at all, so Slice 2's conclusion (stop-gradient, not EMA
smoothing, prevents collapse) needs no revision. `results.csv`'s nine bad rows
are retained under the metric name `probe_r2_superseded_104` rather than deleted.

**Leads for whoever picks this up next:**

1. **Arc 2 cannot run as currently specified.** Its question — does latent-space
   prediction beat pixel-autoencoder and contrastive baselines on
   *sample-efficiency of the downstream probe* — is measured by exactly the
   metric shown here to have ~0.001 of headroom. Fixing the probe task is a
   prerequisite, not a detail: multi-frame inputs (so velocity becomes
   recoverable and stops being dead weight), per-dimension or standardized-target
   R² (so position cannot drown the other factors by raw variance), and a task
   whose factors a random projection does *not* already solve.
2. **`no_ema` probes at 0.977 while sitting at effective_rank 1.25–1.46.** A
   dimensionally-collapsed encoder scores identically to a healthy one, so probe
   R² is not a collapse detector on this task — it is blind to precisely the
   failure mode Arc 1 exists to study. This retroactively justifies Slice 2's
   choice to drop `probe_r2`, and argues `effective_rank` should stay the
   primary metric for the rest of Arc 1.
3. The standing lead to re-test probe R² against a **deeper encoder** is now
   much weaker motivation than it looked: the shallow encoder is not what is
   limiting the score, the ceiling is. Change the task before changing the
   architecture.

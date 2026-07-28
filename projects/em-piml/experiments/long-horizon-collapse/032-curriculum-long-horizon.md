# Does a short-to-long training curriculum fix the long-horizon causality collapse? (issue #32)

Three fixes have now been tried against the long-horizon collapse
(issue #23) and its sibling two-mode spectral-bias gap
(`../two-mode-spectral-bias/`): Fourier feature bandwidth, causal
loss-reweighting (Wang, Sankaran, Perdikaris), and PINNsFormer
pseudo-sequence tokenization (`030-pseudo-sequence-long-horizon.md`).
All three changed representation or loss weighting; none fixed the
collapse, and issue #30 diagnosed why — the failure is a degenerate,
trivially-low-residual solution (near-constant output) that gradient
descent finds easily, and none of those three fixes change what makes
that solution locally attractive. Krishnapriyan et al. ("Characterizing
possible failure modes in physics-informed neural networks," NeurIPS
2021, arXiv:2109.01050) propose curriculum-based training — start on a
restricted/simplified version of the problem, then progressively extend
it — as a fix for exactly this kind of PINN failure mode. This issue
tests the direct analogue: train on `t in [0, PERIOD]` first, then
widen the domain in stages, instead of exposing the full 5-period domain
from the first training step the way
`train_cavity_long_horizon`/`train_cavity_causal_long_horizon` both do.

`train_cavity_curriculum_long_horizon` (`src/em_piml/train.py`) reuses
`train_cavity_long_horizon`'s exact architecture, total step budget
(4000), point counts, and `lr` — training is split into `n_stages=5`
equal-length stages, each calling the existing `_train_pinn_adam`
(unchanged) with `t_max` widened one increment of `PERIOD` per stage
(stage `k` trains on `t in [0, k/5 * 5*PERIOD]`, i.e. stages see 1, 2,
3, 4, then 5 periods). The same model instance carries across stages
(continuing from the previous stage's weights, not reinitializing) —
only the optimizer is recreated fresh each stage, since Adam's
per-parameter moment estimates were tuned against a different (narrower)
point distribution in the prior stage.

**Erratum found while writing this issue's test, affecting issue #23's
`023-long-horizon-causal.md` and issue #30's
`030-pseudo-sequence-long-horizon.md` too — read before trusting any
long-horizon number in this project.** This project's `PERIOD` is `2`,
not `2*pi` (`OMEGA = pi`, `PERIOD = 2*pi/OMEGA = 2` — see
`src/em_piml/physics.py`). The first version of this issue's own
regression test copied a `T_MAX = 5.0 * (2 * math.pi) ≈ 31.4` constant
from `test_causal_long_horizon.py` (issue #23) — assuming `PERIOD =
2*pi`, which is wrong — so it evaluated relative L2 error over **~15.7
periods, not the intended 5**. This surfaced as a real puzzle: this
project's own standing rule is to verify determinism before trusting any
number (see `../../CLAUDE.md`), and the curriculum-training numbers
*were* bit-identical across repeated runs and across
`torch.set_num_threads(1/2/4/12)` — genuinely deterministic — yet the
pytest-run number (0.9569) didn't match the scratch-script number
(0.9141) for the identical `seed=0`. The difference traced to the two
scripts computing `t_max` differently, not to any nondeterminism.
Checking `test_causal_long_horizon.py` (issue #23, already merged) and
the first version of `test_pseudo_sequence_long_horizon.py` (issue #30,
already merged) found the same buggy constant in both. **Impact:** issue
#23's documented numbers were computed over the wrong domain throughout
(now corrected in `023-long-horizon-causal.md` — the qualitative
finding, causal weighting doesn't help, is unchanged, but the absolute
values were off). Issue #30's own numbers were *not* affected (that
issue's analysis script used `t_max = 5.0 * PERIOD` directly, correctly)
— only its shipped test file's `T_MAX` constant needed fixing. All three
test files now compute `T_MAX = 5.0 * PERIOD` via import rather than a
hardcoded approximation of `PERIOD`. The table below uses corrected
numbers throughout.

**Result: a real but modest improvement — not a fix.**

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| uniform (issue #23, corrected) | 0.9225, 0.9255, 0.9229, 0.9249 |
| causal, epsilon=1.0 (issue #23, corrected) | 0.9251, 0.9230, 0.9236, 0.9249 |
| pseudo-sequence (issue #30) | 0.9973, 1.0844, 0.9792, 1.1015 |
| curriculum (this issue) | 0.9141, 0.9120, 0.8985, 0.9111 |

Curriculum training lands at 0.8985-0.9141 — consistently below every
prior approach (all four seeds beat the best seed of uniform/causal,
the closest competitors), the first fix in this project's long-horizon
thread to move the number at all. The margin is real but modest (about
0.01-0.03 absolute against uniform/causal, notably smaller than it
first appeared before the erratum above was found and fixed — the
buggy larger evaluation domain had been inflating the apparent gap). It
remains nowhere close to the single-period baseline's 0.026-0.046 — an
improvement in degree, not a qualitative fix.

**A pointwise check (seed 0) shows why: the same collapse mechanism as
issue #23/#30, just delayed and less severe, not a different or
qualitatively resolved failure.** Evaluating at `x=0.5` across
increasing `t`:

| t | true field | predicted (curriculum) | predicted (issue #23 uniform, for comparison) |
|---|---|---|---|
| 0 | 1.0000 | 0.9835 | 0.975 (2-period run, comparable) |
| 0.25*PERIOD | -0.0000 | 0.0373 | - |
| 0.5*PERIOD | -1.0000 | -0.5331 | -0.21 (at t=1.4, comparable point) |
| 1.0*PERIOD | 1.0000 | 0.1766 | - |
| 1.5*PERIOD | -1.0000 | 0.0120 | - |
| 2.0*PERIOD | 1.0000 | -0.0318 | ~0.05-0.09 plateau from t~2.8 on |
| 3.0*PERIOD | 1.0000 | -0.0209 | - |
| 4.0*PERIOD | 1.0000 | -0.0081 | - |
| 4.9*PERIOD | 0.8090 | -0.0022 | - |

The curriculum-trained model tracks the true oscillation's amplitude
much better through the first period (e.g. `-0.53` vs. true `-1.0` at
`t=0.5*PERIOD`, vs. issue #23/#30's models which were already down to
roughly a fifth of the true amplitude by the comparable point) — the
curriculum's first stage clearly does teach the model to represent one
full period of oscillation well, more faithfully than any prior
approach managed. But starting around `t=1.5*PERIOD` the same collapse
sets in: predictions decay toward a near-zero plateau (`0.01` to
`-0.03`) and stay there through the rest of the domain, while the true
field keeps cycling through its full `[-1, 1]` range. The curriculum
delays the collapse and improves accuracy over the domain it does
delay it through, but each new stage's freshly-widened domain still
gives the optimizer the same "collapse to near-constant, trivially
satisfy the residual" escape hatch issue #23 identified — the curriculum
doesn't remove that escape hatch, it just gives the model more practice
at the easy part of the domain before it becomes available.

`tests/test_curriculum_long_horizon.py` locks in both halves of this
finding as regression checks: relative L2 error `> 0.5` (still a clear
failure overall, no accuracy bar to clear on this target) and `< 0.92`
(below uniform/causal's 0.9225-0.9255, to catch a regression of the
improvement itself) — both observed comfortably at seed 0 (0.9141).

**Leads for whoever picks this up next:**
1. `n_stages=5` (one stage per period) and equal per-stage step
   budgets were both arbitrary choices, untuned — more/finer-grained
   stages (e.g. widening by half a period at a time), or weighting
   later stages with more steps since they cover more not-yet-collapsed
   domain, are both untried and could plausibly extend how far the
   improvement reaches before the same collapse takes over.
2. The escape hatch itself (a near-constant/near-linear function
   trivially satisfies the wave-equation residual) still isn't directly
   addressed by any of the four fixes tried so far in this project
   (Fourier bands, causal weighting, pseudo-sequence tokenization,
   curriculum). An explicit penalty against near-constant output (e.g.
   a minimum-variance or minimum-curvature term added to the loss,
   distinct from the existing residual/BC/IC terms) would attack the
   root cause directly rather than changing representation, loss
   weighting, or training schedule around it — the most literature-
   direct remaining lead flagged by issue #30 and reaffirmed here.
   (Queued as issue #35, anti-trivial-solution regularizer.)
3. Optimizer state was deliberately reset at each stage boundary
   (Adam's moments from a narrower domain don't obviously transfer) —
   untested whether *persisting* optimizer state across stages instead
   changes the outcome either direction.
4. Network capacity remains untested for this specific failure mode
   across all four issues in this thread — still an open, cheap-to-try
   variable given the infrastructure already exists.
5. This project's existing "verify determinism before trusting a
   number" convention (see `../../CLAUDE.md`) caught that this issue's
   result was reproducible, but didn't catch that it was reproducibly
   *wrong* — a hardcoded approximation of `PERIOD` copied between test
   files, rather than importing the real constant, drifted from
   `physics.py`'s actual value undetected across two prior merged PRs.
   Worth generalizing the lesson: derived constants (like an eval-domain
   size) should be computed from their source of truth (`from
   em_piml.physics import PERIOD`), never hand-copied as a literal
   approximation, even when the literal "looks obviously right" (`2*pi`
   reads as a very natural guess for a period).

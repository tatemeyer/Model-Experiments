# Does pseudo-sequence tokenization fix the long-horizon causality collapse? (issue #30)

`../020-pseudo-sequence-tokenization.md` (issue #20) tested PINNsFormer-style
pseudo-sequence tokenization (`PseudoSequenceCavityPINN`) only against the
original single-mode baseline — a target a plain MLP already solves well
(0.026-0.046) — and found it performed markedly worse (0.958-1.383). That
result never distinguished "tokenization doesn't help" from "there was
nothing here for it to help with." `023-long-horizon-causal.md` (issue #23)
has since characterized a genuine failure mode (the plain baseline collapses
to ~0.96 relative L2 over a 5-period horizon, and causal loss-reweighting
doesn't fix it). PINNsFormer's architecture — expanding each `(x, t)` into a
short sequence of nearby *future* timesteps, mixed via self-attention — is a
mechanism aimed squarely at temporal propagation, unlike Fourier features
(frequency content, `../two-mode-spectral-bias/`) or causal loss-reweighting
(loss scheduling, issue #23). This issue tests it against that same
long-horizon target.

`train_pseudo_sequence_cavity_long_horizon` (`src/em_piml/train.py`)
reuses `train_pseudo_sequence_cavity`'s exact shipped config from issue
#20 (`d_model=16, heads=2, ff_dim=32, num_layers=1, k=3, dt=1e-3`,
`steps=600`, `n_collocation=30/n_boundary=16/n_initial=16`) — the only
variable is `t_max = horizon_periods * PERIOD` (default `5.0`, matching
issue #23's `horizon_periods=5`), threaded through
`_train_pseudo_sequence_pinn_adam` to `_sample_points` the same way
issue #23 threaded it through `_train_pinn_adam`.

**Result: no better — if anything, slightly worse. Same failure
mechanism as the plain baseline, not a different one.**

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| plain baseline, uniform (issue #23, corrected — see `032-curriculum-long-horizon.md` erratum) | 0.9225, 0.9255, 0.9229, 0.9249 |
| plain baseline, causal, epsilon=1.0 (issue #23, corrected) | 0.9251, 0.9230, 0.9236, 0.9249 |
| pseudo-sequence (this issue) | 0.9973, 1.0844, 0.9792, 1.1015 |

(Note: this issue's own pseudo-sequence numbers were always computed at
the correct `t_max = 5.0 * PERIOD` — only `tests/test_causal_long_horizon.py`
and, initially, `tests/test_pseudo_sequence_long_horizon.py` had a buggy
`T_MAX` constant, since fixed; see `032-curriculum-long-horizon.md`'s
erratum for the full story.)

Pseudo-sequence tokenization lands at 0.9792-1.1015 across seeds
0/1/2/7 — statistically no improvement over the plain baseline's
0.9225-0.9255, and two of four seeds (1.0844, 1.1015) are actually
*worse* than total failure (relative L2 > 1 means the prediction is
farther from the true field than a zero-output model would be).

**A pointwise check (seed 0) confirms this is the same collapse
mechanism issue #23 diagnosed, not a different pathology:** evaluating
the trained model at `x=0.5` across increasing `t`:

| t | true field | predicted |
|---|---|---|
| 0 | 1.0000 | 0.9930 |
| 0.25*PERIOD | -0.0000 | 0.5713 |
| 0.5*PERIOD | -1.0000 | 0.2319 |
| 1.0*PERIOD | 1.0000 | 0.0588 |
| 1.5*PERIOD | -1.0000 | 0.0001 |
| 2.0*PERIOD | 1.0000 | 0.0043 |
| 3.0*PERIOD | 1.0000 | -0.0173 |
| 4.0*PERIOD | 1.0000 | -0.0027 |
| 4.9*PERIOD | 0.8090 | -0.0388 |

The model tracks the true field closely right at `t=0` (0.993 vs.
1.0), then collapses to a near-zero plateau within about one period —
by `t=PERIOD` onward, predictions stay within roughly ±0.06 of zero
while the true field keeps cycling through its full `[-1, 1]` range.
This is the identical "near-constant output trivially satisfies the
wave equation" collapse issue #23 found and instrumented for the plain
MLP, not a new failure mode specific to this architecture. Unsurprising
in hindsight: the degenerate low-residual collapse is a property of the
wave equation itself (near-constant/near-linear functions have
near-zero second derivatives in both `x` and `t`, satisfying the PDE
residual almost for free) — nothing about self-attention over a short
pseudo-sequence of nearby timesteps changes what makes that collapsed
solution locally attractive to gradient descent.

`tests/test_pseudo_sequence_long_horizon.py` locks in the finding as a
regression check (not an accuracy bar — there's no bar to clear on this
target yet): seed 0, default config, asserts relative L2 error `> 0.5`
— comfortable margin below the observed 0.979-1.102 range.

**This is now the third failure mode in this project (after
`../two-mode-spectral-bias/025-num-bands-sweep.md` and issue #23's
causal-reweighting result) where a literature-motivated
architecture/technique change didn't transfer to this project's
specific failure mechanism** — and the second time specifically for the
long-horizon collapse. The common thread across all three: this
project's failures are various flavors of "the optimizer finds a
cheap, locally-satisfying degenerate solution," and none of the fixes
tried so far (more embedding frequencies, causal loss scheduling,
sequence-based temporal mixing) change what makes that degenerate
solution attractive — they change representation or loss weighting,
not the fact that a trivial low-residual solution exists and is easy to
find.

**Leads for whoever picks this up next:**
1. All three literature fixes tried against this project's two hard
   failure modes (the `num_bands` sweep, causal reweighting, this
   issue's pseudo-sequence tokenization) targeted representation or
   loss-weighting. None directly attacks the root cause identified
   here and in issue #23: a degenerate low-residual solution exists and
   is reachable from random init. A technique that explicitly
   penalizes trivial/near-constant solutions (e.g. a curvature or
   variance floor on the prediction, or an explicit short-to-long
   curriculum) is the more literature-direct next step than another
   representation change. (Curriculum tried next: see
   `032-curriculum-long-horizon.md`.)
2. Network capacity was held fixed at issue #20's shipped
   `PseudoSequenceCavityPINN` size throughout — untested whether a
   larger model changes this outcome, though issue #23's equivalent
   lead (same question for the plain MLP) is equally untested and
   there's no specific reason to expect capacity to be the missing
   piece for a collapse that isn't about representational capacity.
3. `k` (pseudo-sequence length) and `dt` (timestep spacing) were left
   at issue #20's shipped values (`k=3, dt=1e-3`) — both are tiny
   relative to this target's 5-period (~31.4 time-unit) horizon, so the
   sequence only ever spans a `2e-3` time-unit window regardless of
   where in the domain a point falls. A much larger `dt` (spanning a
   meaningful fraction of one period) is untried and is arguably a
   fairer test of whether temporal-sequence mixing helps propagate
   information forward — this issue deliberately kept the shipped
   config fixed per its own controlled-comparison constraint, but this
   is the most literature-plausible reason a larger `dt` might matter.

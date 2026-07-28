from __future__ import annotations

import pytest
from em_piml.physics import PERIOD
from em_piml.train import evaluate_relative_l2_error, train_cavity_curriculum_long_horizon

# Issue #32: three prior fixes for the long-horizon collapse (issue #23) all changed
# representation or loss weighting and none helped -- Fourier bandwidth (issue #25), causal
# loss-reweighting (issue #23), pseudo-sequence tokenization (issue #30). Per Krishnapriyan et al.
# ("Characterizing possible failure modes in physics-informed neural networks," NeurIPS 2021,
# arXiv:2109.01050), this issue instead trains on a restricted time domain first and progressively
# extends it in stages (n_stages=5, same total step budget as train_cavity_long_horizon).
#
# Result: a real but modest improvement, not a fix. Relative L2 error 0.8985-0.9141 across seeds
# 0/1/2/7 -- consistently, though only modestly, better than issue #23's uniform (0.9225-0.9255)
# and causal (0.9230-0.9251) and issue #30's pseudo-sequence (0.9792-1.1015), but nowhere close to
# the single-period baseline's 0.026-0.046. (Numbers here use the correct T_MAX = 5.0 * PERIOD --
# an earlier draft of this test copied a buggy T_MAX = 5.0 * (2 * math.pi) from
# test_causal_long_horizon.py, which evaluated ~15.7 periods instead of 5; see CLAUDE.md's issue
# #32 erratum. That bug inflated the apparent improvement margin -- the real margin is real but
# smaller than first measured.) A pointwise check (seed 0) showed the same degenerate
# near-zero-plateau collapse mechanism as issue #23/#30, just delayed and less severe: the model
# tracks the true field through most of the first period (e.g. t=0.5*PERIOD: predicted -0.53 vs
# true -1.0, far better than issue #23/#30's ~0.2-0.23 at the same point) before decaying to a
# near-zero plateau by around t=1.5-2*PERIOD onward. See CLAUDE.md for the full writeup.
FAILURE_LOWER_BOUND = 0.5
IMPROVEMENT_UPPER_BOUND = 0.92  # below uniform/causal's 0.9225-0.9255 -- locks in the improvement
T_MAX = 5.0 * PERIOD  # 5 periods of this project's fundamental mode


@pytest.mark.slow
def test_curriculum_improves_on_prior_approaches_but_does_not_fix_collapse():
    """Curriculum training should land measurably below issue #23/#30's ~0.92-1.10 range, but
    still fail badly relative to the single-period baseline -- a partial improvement, not a fix."""
    model = train_cavity_curriculum_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected curriculum training to still fail on the long-horizon target overall, got "
        f"relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in CLAUDE.md "
        f"issue #32 needs revisiting"
    )
    assert relative_l2 < IMPROVEMENT_UPPER_BOUND, (
        f"expected curriculum training to measurably improve on issue #23/#30's ~0.92-1.10 range, "
        f"got relative L2 error: {relative_l2:.4f} -- if this regressed, the finding in CLAUDE.md "
        f"issue #32 needs revisiting"
    )

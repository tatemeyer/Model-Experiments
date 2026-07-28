from __future__ import annotations

import pytest
from em_piml.physics import PERIOD
from em_piml.train import evaluate_relative_l2_error, train_cavity_ntk_reweighted_long_horizon

# Issue #34: this repo's own CLAUDE.md (issues #20/#25/#30) repeatedly flagged NTK-based adaptive
# loss-term reweighting (Wang, Teng, Perdikaris, "Understanding and Mitigating Gradient Flow
# Pathologies in Physics-Informed Neural Networks," SIAM J. Sci. Comput. 2021, arXiv:2001.04536)
# as the most-promising unexplored lever against the long-horizon collapse (issue #23). This test
# implements it and finds it doesn't just fail to help -- it makes things measurably worse.
#
# Result: relative L2 error 0.9999-1.1336 across seeds 0/1/2/7, worse than uniform (0.9225-0.9255,
# issue #23) and causal reweighting (0.9230-0.9251, issue #23) and no better than pseudo-sequence
# tokenization (0.9792-1.1015, issue #30). Diagnosed mechanistically (not just a bare null result):
# inspecting the adaptive weights during training shows the PDE-residual loss's gradient norm stays
# tiny (~1e-3 to 5e-3) throughout, while the BC/IC terms' gradient norms stay large (~0.5-2.5) --
# the *opposite* of the regime this reweighting scheme assumes (where the PDE-residual term
# typically dominates and needs down-weighting relative to BC/IC). Since the scheme's weight for
# term i is g_pde / g_i, this crushes the IC weight toward ~1e-3-1e-2 (from a uniform baseline of
# 1.0) instead of upweighting a neglected term -- removing exactly the constraint (matching the
# true initial condition) that was pulling the model away from the trivial near-constant collapse
# in the first place. See CLAUDE.md for the full writeup and weight trajectory.
FAILURE_LOWER_BOUND = 0.5
T_MAX = 5.0 * PERIOD  # 5 periods of this project's fundamental mode


@pytest.mark.slow
def test_ntk_reweighting_does_not_fix_long_horizon_collapse():
    """NTK-based adaptive loss reweighting should still fail badly on the long-horizon target --
    no better than (in fact slightly worse than) uniform weighting, not a fix."""
    model = train_cavity_ntk_reweighted_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected NTK-reweighted training to still fail badly on the long-horizon target, got "
        f"relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in CLAUDE.md "
        f"issue #34 needs revisiting"
    )

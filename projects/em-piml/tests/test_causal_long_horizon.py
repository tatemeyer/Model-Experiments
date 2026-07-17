from __future__ import annotations

import math

import pytest
from em_piml.train import (
    evaluate_relative_l2_error,
    train_cavity_causal_long_horizon,
    train_cavity_long_horizon,
)

# Issue #23: does training over a longer time horizon (5 periods instead of 1) break the plain
# baseline, and does causal loss-reweighting (Wang, Sankaran, Perdikaris, arXiv:2203.07404) fix
# it? See CLAUDE.md for the full writeup, mechanism diagnosis, and multi-seed/multi-epsilon sweep.
# Both bounds document a failure signature, not an accuracy target -- there's no bar to clear on
# this target yet. Observed: uniform 0.9592-0.9633, causal (epsilon 0.1-500) 0.9571-0.9679 across
# seeds 0/1/2/7 -- both far above this bound with wide margin, and statistically indistinguishable
# from each other.
FAILURE_LOWER_BOUND = 0.5
T_MAX = 5.0 * (2 * math.pi)  # 5 periods of this project's fundamental mode (OMEGA = pi, C = L = 1)


@pytest.mark.slow
def test_long_horizon_degrades_baseline_accuracy():
    """Uniform weighting over a 5-period horizon should still fail badly (vs. 0.026-0.046 on the
    single-period baseline) -- the model collapses to a near-constant plateau instead of
    continuing the true oscillation past the first period or two."""
    model = train_cavity_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected the long-horizon baseline to still fail badly, got relative L2 error: "
        f"{relative_l2:.4f} -- if this now passes, the finding in CLAUDE.md issue #23 needs "
        f"revisiting"
    )


@pytest.mark.slow
def test_causal_weighting_does_not_recover_accuracy():
    """Causal loss-reweighting, applied exactly as the paper specifies (exponential down-weight
    of later time-chunk PDE residual by cumulative earlier-chunk residual), should still fail --
    see CLAUDE.md for why the mechanism doesn't transfer to this project's specific failure mode
    (a trivially-low-residual collapse, not an unconverged-residual lag)."""
    model = train_cavity_causal_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected causal weighting to still fail to recover accuracy, got relative L2 error: "
        f"{relative_l2:.4f} -- if this now passes, the finding in CLAUDE.md issue #23 needs "
        f"revisiting"
    )

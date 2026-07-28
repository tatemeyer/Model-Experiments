from __future__ import annotations

import math

import pytest
from em_piml.train import evaluate_relative_l2_error, train_pseudo_sequence_cavity_long_horizon

# Issue #30: issue #20 found pseudo-sequence tokenization (Zhao et al., "PINNsFormer", ICLR 2024,
# arXiv:2307.11833) performs markedly worse than the baseline, but only against the too-easy
# single-mode target a plain MLP already solves well -- leaving open whether it might fare
# differently against a target the baseline actually fails on. Issue #23 characterized exactly
# such a target (a 5-period horizon collapses the plain baseline, and causal loss-reweighting
# doesn't fix it). This issue tests pseudo-sequence tokenization against that same target.
#
# Result: no better -- if anything, slightly worse. Relative L2 error 0.9792-1.1015 across seeds
# 0/1/2/7, vs. issue #23's uniform-weighting baseline (0.9592-0.9633) and causal-reweighted
# (0.9571-0.9679) on the identical target. A pointwise check (seed 0) confirmed the same failure
# mechanism issue #23 diagnosed for the plain MLP: the model tracks the true field near t=0
# (predicted 0.993 vs true 1.0 at t=0) then collapses to a near-zero plateau within about one
# period (predicted 0.0001-0.06 from t=PERIOD onward, vs. true values cycling through the full
# [-1, 1] range) -- the same "near-constant output trivially satisfies the wave equation" collapse,
# not a different pathology. See CLAUDE.md for the full writeup.
FAILURE_LOWER_BOUND = 0.5
T_MAX = 5.0 * (2 * math.pi)  # 5 periods of this project's fundamental mode (OMEGA = pi, C = L = 1)


@pytest.mark.slow
def test_pseudo_sequence_does_not_fix_long_horizon_collapse():
    """Pseudo-sequence tokenization should still fail badly on the long-horizon target -- same
    order of magnitude as the plain baseline and causal reweighting (issue #23), not a fix."""
    model = train_pseudo_sequence_cavity_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected pseudo-sequence tokenization to still fail badly on the long-horizon target, "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #30 needs revisiting"
    )

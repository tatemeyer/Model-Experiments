from __future__ import annotations

import pytest
from em_piml.physics import PERIOD
from em_piml.train import evaluate_relative_l2_error, train_pseudo_sequence_cavity_long_horizon

# Issue #30: issue #20 found pseudo-sequence tokenization (Zhao et al., "PINNsFormer", ICLR 2024,
# arXiv:2307.11833) performs markedly worse than the baseline, but only against the too-easy
# single-mode target a plain MLP already solves well -- leaving open whether it might fare
# differently against a target the baseline actually fails on. Issue #23 characterized exactly
# such a target (a 5-period horizon collapses the plain baseline, and causal loss-reweighting
# doesn't fix it). This issue tests pseudo-sequence tokenization against that same target.
#
# Result: no better -- if anything, slightly worse. Relative L2 error 0.9792-1.1015 across seeds
# 0/1/2/7, vs. issue #23's uniform-weighting baseline (0.9225-0.9255, corrected -- see issue #32's
# erratum in CLAUDE.md) and causal-reweighted (0.9230-0.9251, corrected) on the identical target.
# A pointwise check (seed 0) confirmed the same failure mechanism issue #23 diagnosed for the
# plain MLP: the model tracks the true field near t=0 (predicted 0.993 vs true 1.0 at t=0) then
# collapses to a near-zero plateau within about one period (predicted 0.0001-0.06 from t=PERIOD
# onward, vs. true values cycling through the full [-1, 1] range) -- the same "near-constant
# output trivially satisfies the wave equation" collapse, not a different pathology. See CLAUDE.md
# for the full writeup. Note: this T_MAX was originally miscomputed as 5.0 * (2 * math.pi) -- this
# project's PERIOD is 2, not 2*pi (see em_piml.physics), so that evaluated ~15.7 periods instead
# of 5. The numbers in this file's docstring/CLAUDE.md were already computed with the correct
# T_MAX = 5.0 * PERIOD (the analysis script used the right formula directly); only this test's own
# constant needed fixing, found and fixed as part of issue #32.
FAILURE_LOWER_BOUND = 0.5
T_MAX = 5.0 * PERIOD  # 5 periods of this project's fundamental mode


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

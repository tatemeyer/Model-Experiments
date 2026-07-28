from __future__ import annotations

import pytest
from em_piml.physics import PERIOD
from em_piml.train import evaluate_relative_l2_error, train_cavity_antitrivial_long_horizon

# Issue #35: every prior fix in the long-horizon-collapse thread (causal reweighting #23,
# pseudo-sequence tokenization #30, curriculum #32, NTK reweighting #34) changed representation,
# loss scheduling, or training schedule -- none of them made the degenerate near-constant trivial
# solution itself less attractive to gradient descent. Leiteritz & Pflueger ("How to Avoid Trivial
# Solutions in Physics-Informed Neural Networks," arXiv:2112.05620) propose exactly that: an
# additional loss term penalizing the max squared gradient of the PDE residual across collocation
# points (their eq. 8/13) -- a smoothness/anti-spike penalty on the residual field itself, meant to
# obstruct the trivial solution from being reachable in the first place, not just reweight terms
# around it.
#
# Result: does not fix the collapse -- lands slightly *worse* than uniform weighting. Relative L2
# error 0.9680-0.9716 across seeds 0/1/2/7 (tightly clustered), vs. uniform's 0.9225-0.9255 (issue
# #23) and causal's 0.9230-0.9251 (issue #23) -- a real, if small, regression, though nowhere near
# as bad as pseudo-sequence tokenization (0.9792-1.1015, issue #30) or NTK reweighting
# (0.9999-1.1336, issue #34). Diagnosed mechanistically (see CLAUDE.md / the experiment writeup):
# instrumenting the penalty term's magnitude and its spatial distribution across the domain found
# no localized spike for it to suppress -- this project's collapse is a smooth, domain-wide settling
# into near-constant output, not the abrupt truth-to-trivial switch the paper's mechanism targets --
# so the (nonzero-cost) penalty mostly just competes with the IC term for fitting capacity instead
# of preventing anything.
# See CLAUDE.md / experiments/long-horizon-collapse/035-antitrivial-regularizer.md for the full
# writeup, pointwise diagnosis, and result table.
FAILURE_LOWER_BOUND = 0.5
REGRESSION_UPPER_BOUND = 1.05  # below NTK-reweighted's 0.9999-1.1336 -- not as bad as that fix
T_MAX = 5.0 * PERIOD  # 5 periods of this project's fundamental mode


@pytest.mark.slow
def test_antitrivial_regularizer_does_not_fix_long_horizon_collapse():
    """The anti-trivial-solution regularizer should still fail on the long-horizon target -- a
    small regression relative to uniform weighting, not a fix, and not as severe as NTK
    reweighting's actively-worse result."""
    model = train_cavity_antitrivial_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected the anti-trivial-solution regularizer to still show the long-horizon collapse "
        f"signature on this target, got relative L2 error: {relative_l2:.4f} -- if this now "
        f"passes, the finding in CLAUDE.md issue #35 needs revisiting"
    )
    assert relative_l2 < REGRESSION_UPPER_BOUND, (
        f"expected the regression to stay milder than NTK reweighting's ~1.00-1.13, got relative "
        f"L2 error: {relative_l2:.4f} -- if this now exceeds that range, the finding in CLAUDE.md "
        f"issue #35 needs revisiting"
    )

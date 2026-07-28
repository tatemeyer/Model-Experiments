from __future__ import annotations

import pytest
from em_piml.physics import PERIOD
from em_piml.train import evaluate_relative_l2_error, train_cavity_r3_long_horizon

# Issue #37: every plain-MLP fix in the long-horizon-collapse thread so far (causal reweighting
# #23, pseudo-sequence tokenization #30, curriculum #32, NTK reweighting #34, anti-trivial
# regularizer #35) changed representation, loss weighting, or training schedule -- none touched
# the collocation-*sampling* strategy itself. This tests R3 (Retain-Resample-Release) sampling
# (Daw, Bu, Wang, Perdikaris, Karpatne, "Mitigating Propagation Failures in Physics-Informed
# Neural Networks using Retain-Resample-Release (R3) Sampling," ICML 2023, arXiv:2207.02338):
# retain collocation points whose |residual| exceeds the pool's own mean, release the rest,
# resample fresh uniform points to refill -- so points near a persistently high-residual region
# accumulate across steps instead of being redrawn away every step.
#
# Result: does not fix the collapse -- a small, consistent regression, similar in shape to issue
# #35's anti-trivial regularizer but milder. Relative L2 error 0.9303-0.9343 across seeds 0/1/2/7,
# vs. uniform's 0.9225-0.9255 (issue #23) and causal's 0.9230-0.9251 (issue #23) -- every R3 seed
# is worse than every uniform/causal seed, though far milder than NTK reweighting's 0.9999-1.1336
# (issue #34) or the anti-trivial regularizer's 0.9680-0.9716 (issue #35). Diagnosed
# mechanistically: R3 correctly retains points from the genuinely highest-residual region (the
# first ~40% of the domain, where the model is still doing real work tracking the true
# oscillation), but that's not where the collapse lives -- the collapsed region (the back ~60% of
# the domain) has near-zero residual by construction (a near-constant function trivially satisfies
# the wave equation), so R3's residual-based retain criterion never once flags it across the
# entire training run, and concentrating collocation budget on the already-best-fit early region
# implicitly reduces how often the collapsed region gets sampled at all relative to uniform's flat
# rate. See experiments/long-horizon-collapse/037-r3-long-horizon.md for the full retained-point
# distribution and per-chunk residual tables, and the pointwise check confirming it's the same
# collapse mechanism, not a different one.
FAILURE_LOWER_BOUND = 0.5
T_MAX = 5.0 * PERIOD  # 5 periods of this project's fundamental mode


@pytest.mark.slow
def test_r3_sampling_does_not_fix_long_horizon_collapse():
    """R3 (Retain-Resample-Release) adaptive collocation sampling should still fail on the
    long-horizon target -- a small regression relative to uniform weighting, not a fix, and
    milder than NTK reweighting's or the anti-trivial regularizer's regressions."""
    model = train_cavity_r3_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected R3 sampling to still show the long-horizon collapse signature on this target, "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #37 needs revisiting"
    )

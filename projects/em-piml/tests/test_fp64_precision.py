from __future__ import annotations

import pytest
import torch
from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs_fp64

# Issue #38: "FP64 is All You Need" (arXiv:2505.10949) argues L-BFGS's internal convergence test
# fires prematurely under FP32, causing PINN "failure modes" that look like genuine local optima
# but are actually precision artifacts -- switching to torch.float64 is claimed to resolve them
# with no other change. This tests that claim directly against this project's own num_bands=4
# L-BFGS plateau from issues #6/#8 (32-hidden, n_collocation=200 -- the actual pre-density-fix,
# pre-capacity-fix setup that produced the ~0.79-0.88 FP32 plateau, not issue #8's already-fixed
# 2000-point default).
#
# Result: FP64 does NOT fix the plateau. Observed relative L2 error across seeds 0/1/2/7:
# 0.888-0.922 -- statistically indistinguishable from (if anything marginally worse than) the
# documented FP32 numbers (0.822, 0.851 for seeds 0/1). See
# experiments/num-bands-gap/038-fp64-precision.md for the full comparison, including the
# already-good 64-hidden shipped config (also unaffected by precision: FP64 0.028 vs FP32 0.027
# at seed 0) and the runtime tradeoff that forced this test to use the (cheaper, ~400s/seed)
# original 200-point config rather than the shipped 2000-point one for a *reduced-budget* CI
# check -- reducing outer_steps/max_iter was tried and found to invalidate the result entirely
# (collapses to total failure regardless of precision), so this test uses the SAME
# outer_steps=50/max_iter=50 budget as the documented FP32 numbers; there is no cheaper faithful
# version of this check.
FAILURE_LOWER_BOUND = 0.5  # matches this project's usual "still fails badly" bound


@pytest.mark.slow
def test_fp64_does_not_fix_original_num_bands_4_plateau():
    """FP32's ~0.79-0.88 plateau (issues #6/#8) persists under FP64 -- not a precision artifact."""
    model = train_fourier_cavity_lbfgs_fp64(
        seed=0, num_bands=4, hidden=32, n_collocation=200, n_boundary=64, n_initial=64
    )
    relative_l2 = evaluate_relative_l2_error(model, dtype=torch.float64)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected FP64 to still fail badly on the original num_bands=4/32-hidden/200-point "
        f"config, got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #38 needs revisiting"
    )


def test_fp64_dtype_plumbing_is_consistent():
    """Fast (not slow-marked) sanity check that dtype=torch.float64 threads correctly through
    model construction, point sampling, and evaluation without erroring or silently staying
    float32 -- distinct from the accuracy claim above, which needs the expensive full budget to
    mean anything (see this file's module docstring)."""
    model = train_fourier_cavity_lbfgs_fp64(
        seed=0,
        num_bands=4,
        hidden=8,
        outer_steps=2,
        max_iter=5,
        n_collocation=20,
        n_boundary=8,
        n_initial=8,
    )
    for param in model.parameters():
        assert param.dtype == torch.float64
    assert model.embedding.frequencies.dtype == torch.float64

    relative_l2 = evaluate_relative_l2_error(model, dtype=torch.float64)
    assert relative_l2 == relative_l2  # not NaN
    assert relative_l2 >= 0.0

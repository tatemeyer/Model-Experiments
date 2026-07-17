from __future__ import annotations

import pytest
from em_piml.physics import analytical_field_two_mode
from em_piml.train import (
    evaluate_relative_l2_error,
    train_cavity_two_mode,
    train_fourier_cavity_two_mode,
)

# Issue #22: does a two-mode target (n=1 + n=8, see analytical_field_two_mode) break the plain
# baseline via spectral bias, and does the existing Fourier embedding (num_bands=2 default) fix
# it? Across seeds 0/1/2/7:
#   plain (CavityPINN):           0.7699-0.7947 -- vs. 0.026-0.046 on the single-mode baseline,
#                                  confirming a real failure induced by the added high mode.
#   fourier (num_bands=2):        0.6995-0.7063 -- only a modest improvement, not a fix.
# A pointwise check (not asserted here, see CLAUDE.md) confirms *why*: both models' predictions
# track the smooth n=1 envelope almost exactly and miss the n=8 ripple entirely -- num_bands=2's
# basis frequencies {pi, 2pi} don't include 8*pi, so it has no representational capacity for the
# n=8 mode regardless of training. These bounds document that failure signature, not an accuracy
# target: > 0.5 leaves ~2.6-3.4x margin above zero while comfortably below the observed ~0.70-0.79
# range, so the tests would catch either a surprising full recovery or a bug that makes it worse.
FAILURE_LOWER_BOUND = 0.5


@pytest.mark.slow
def test_plain_baseline_fails_on_two_mode_target():
    """Spectral bias: the plain MLP should still fail on the n=1+n=8 superposition."""
    model = train_cavity_two_mode(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected the plain baseline to still fail on the two-mode target (spectral bias), "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #22 needs revisiting"
    )


@pytest.mark.slow
def test_fourier_num_bands_2_does_not_fully_fix_two_mode_target():
    """num_bands=2 lacks the 8*pi basis frequency needed to represent the n=8 mode."""
    model = train_fourier_cavity_two_mode(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected num_bands=2 to still fall well short of fixing the two-mode target, "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #22 needs revisiting"
    )

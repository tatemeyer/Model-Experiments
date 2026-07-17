from __future__ import annotations

import pytest
from em_piml.physics import analytical_field_two_mode
from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs_two_mode

# Issue #25: does raising num_bands (specifically to 4, whose basis {pi,2pi,4pi,8pi} includes the
# 8*pi the n=8 mode needs) close the two-mode spectral-bias gap issue #22 found? See CLAUDE.md for
# the full multi-seed, multi-optimizer sweep (src/em_piml/num_bands_sweep.py). Both bounds below
# use L-BFGS specifically: its between-seed spread was tighter than SOAP's at every num_bands
# tested (stdev <= 0.008 vs. SOAP's 0.069-292 at num_bands=6/8), making it the more reliable
# single-seed regression signal.
STILL_FAILS_BOUND = 0.5  # same bound issue #22 used; num_bands=4 lbfgs observed 0.6986-0.7049
COLLAPSE_BOUND = 0.9  # num_bands=6 lbfgs observed 1.0205-1.0336 -- confirms collapse, not a fix


@pytest.mark.slow
def test_num_bands_4_still_falls_short_on_two_mode_target():
    """num_bands=4 adds the needed 8*pi basis frequency but doesn't close the gap."""
    model = train_fourier_cavity_lbfgs_two_mode(seed=0, num_bands=4)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > STILL_FAILS_BOUND, (
        f"expected num_bands=4 (lbfgs) to still fall well short of fixing the two-mode target, "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #25 needs revisiting"
    )


@pytest.mark.slow
def test_num_bands_6_destabilizes_rather_than_helps():
    """num_bands=6 collapses training rather than improving on num_bands=4's partial result."""
    model = train_fourier_cavity_lbfgs_two_mode(seed=0, num_bands=6)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > COLLAPSE_BOUND, (
        f"expected num_bands=6 (lbfgs) to collapse rather than improve on the two-mode target, "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #25 needs revisiting"
    )

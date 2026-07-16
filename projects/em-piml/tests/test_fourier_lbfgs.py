from __future__ import annotations

import pytest
from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs

# Standard 0.1 bar (see test_baseline_cavity.py / test_fourier_embedding.py), finally reached here.
# issue #6: 32-hidden L-BFGS plateaued ~0.8 relative L2 at 200 points. issue #8: densifying the
# collocation set (2000/400/400) got that down to 0.065-0.104 -- close but not reliably under 0.1
# (seed 1 landed at 0.104). issue #10: widening the MLP body to 64-hidden (up from 32, sized for
# the higher-dimensional num_bands=4 Fourier input) closed the rest of the gap -- observed
# 0.018-0.041 across seeds 0/1/2/7, ~2.4x margin below this bound. See projects/em-piml/CLAUDE.md.
RELATIVE_L2_TOLERANCE = 0.1


@pytest.mark.slow
def test_wider_body_closes_num_bands_4_lbfgs_gap_to_standard_bar():
    """32-hidden plateaued at 0.065-0.104 (issue #8); 64-hidden reaches 0.018-0.041."""
    model = train_fourier_cavity_lbfgs(seed=0, num_bands=4)
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < RELATIVE_L2_TOLERANCE, f"relative L2 error too high: {relative_l2:.4f}"

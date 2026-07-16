from __future__ import annotations

import pytest
from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_soap

# Issue #11: does SOAP (Vyas et al., "Improving and Stabilizing Shampoo using Adam") close the
# num_bands=4 gap left by L-BFGS (issues #6/#8: 0.065-0.104, see test_fourier_lbfgs.py)? Same
# architecture/point-set (2000/400/400) as the L-BFGS default, optimizer swapped in for L-BFGS.
# Observed relative L2 error across seeds 0/1/2/7: 0.0232-0.0357 — fully closes the gap and lands
# in the same range as the raw-coordinate Adam baseline (0.026-0.046), not just under the 0.1 bar.
# 0.08 chosen with ~2.2-3.4x margin above the observed worst case (0.0357), same margin style as
# test_baseline_cavity.py's 0.1 over its 0.026-0.046 range. See projects/em-piml/CLAUDE.md.
SOAP_BOUND = 0.08


@pytest.mark.slow
def test_soap_closes_num_bands_4_gap():
    """L-BFGS plateaus at 0.065-0.104 at num_bands=4. SOAP reaches 0.0232-0.0357."""
    model = train_fourier_cavity_soap(seed=0, num_bands=4)
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < SOAP_BOUND, (
        f"expected SOAP to close the L-BFGS plateau gap at num_bands=4, "
        f"got relative L2 error: {relative_l2:.4f}"
    )

from __future__ import annotations

import pytest
from em_piml.train import evaluate_relative_l2_error, train_cavity_baseline

RELATIVE_L2_TOLERANCE = 0.1  # baseline PINN, not tuned for high precision — see project CLAUDE.md


@pytest.mark.slow
def test_baseline_pinn_matches_analytical_solution():
    model = train_cavity_baseline(seed=0)
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < RELATIVE_L2_TOLERANCE, f"relative L2 error too high: {relative_l2:.4f}"

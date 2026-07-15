from __future__ import annotations

import torch
from em_piml.physics import PERIOD, L, analytical_field
from em_piml.train import train_cavity_baseline

RELATIVE_L2_TOLERANCE = 0.1  # baseline PINN, not tuned for high precision — see project CLAUDE.md


def test_baseline_pinn_matches_analytical_solution():
    model = train_cavity_baseline(seed=0)

    torch.manual_seed(123)  # different seed than training — genuinely held-out points
    x = torch.rand(500, 1) * L
    t = torch.rand(500, 1) * PERIOD

    with torch.no_grad():
        predicted = model(x, t)
        true = analytical_field(x, t)

    relative_l2 = (torch.linalg.norm(predicted - true) / torch.linalg.norm(true)).item()
    assert relative_l2 < RELATIVE_L2_TOLERANCE, f"relative L2 error too high: {relative_l2:.4f}"

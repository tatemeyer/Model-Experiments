from __future__ import annotations

import math

import pytest
import torch
from em_piml.helmholtz import L, analytical_mode, eigenvalue, helmholtz_residual
from em_piml.train import evaluate_relative_l2_error_helmholtz, train_helmholtz_mode


class _AnalyticalMode(torch.nn.Module):
    """Wraps the closed-form eigenfunction as a Module so helmholtz_residual can be applied to it
    -- the residual of the exact solution must be zero by construction."""

    def __init__(self, mode_order: int):
        super().__init__()
        self.mode_order = mode_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return analytical_mode(x, self.mode_order)


def test_analytical_mode_satisfies_helmholtz_equation_and_bcs():
    """Locks in the closed-form reference: E_n(x) = sin(k_n x) has zero Helmholtz residual and
    vanishes at both PEC walls. This is the exactness guarantee every relative-L2 number in
    experiments/043-helmholtz-eigenvalue-capacity.md is measured against."""
    mode_order = 3
    x = torch.linspace(0.05, L - 0.05, 64).reshape(-1, 1)
    residual = helmholtz_residual(_AnalyticalMode(mode_order), x, mode_order)
    assert residual.abs().max().item() == pytest.approx(0.0, abs=1e-3)

    walls = torch.tensor([[0.0], [L]])
    assert analytical_mode(walls, mode_order).abs().max().item() == pytest.approx(0.0, abs=1e-5)

    assert eigenvalue(mode_order) == pytest.approx(mode_order * math.pi / L)


def test_training_is_deterministic_for_a_fixed_seed():
    """This project's standing rule (issues #19 and #32): same seed in, bit-identical result out.
    Cheap enough to run in the default suite at 50 steps -- the point is the seeding contract,
    not accuracy."""
    first = train_helmholtz_mode(mode_order=4, hidden=32, seed=0, steps=50)
    second = train_helmholtz_mode(mode_order=4, hidden=32, seed=0, steps=50)
    assert evaluate_relative_l2_error_helmholtz(
        first, mode_order=4
    ) == evaluate_relative_l2_error_helmholtz(second, mode_order=4)

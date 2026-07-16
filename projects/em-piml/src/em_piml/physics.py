from __future__ import annotations

import math

import torch

L = 1.0
C = 1.0
N_MODE = 1
AMPLITUDE = 1.0
OMEGA = N_MODE * math.pi * C / L
PERIOD = 2 * math.pi / OMEGA

# Second mode for the two-mode superposition target (issue #22) — a much higher spatial mode
# added on top of the fundamental, equal amplitude. PERIOD/OMEGA above stay derived from N_MODE
# (the fundamental) only, so the training/eval domain (one period of the slowest mode) already
# spans exactly N_MODE_2 full oscillations of this faster mode with no domain-size change needed.
N_MODE_2 = 8
AMPLITUDE_1 = 0.5
AMPLITUDE_2 = 0.5
OMEGA_2 = N_MODE_2 * math.pi * C / L


def analytical_field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Closed-form E_z(x, t) for the fundamental mode of a 1D PEC cavity of length L."""
    return AMPLITUDE * torch.sin(N_MODE * math.pi * x / L) * torch.cos(OMEGA * t)


def analytical_field_two_mode(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Closed-form E_z(x, t): fundamental mode (n=1) plus a higher spatial mode (n=8), equal
    amplitude. Exact solution of the same PDE/BCs as analytical_field — the wave equation is
    linear and both terms individually satisfy E(0,t)=E(L,t)=0 and dE/dt(x,0)=0, so their sum
    does too. See projects/em-piml/CLAUDE.md issue #22: this exists to test whether a plain
    coordinate MLP shows the classic spectral-bias failure (fits the low mode, misses the high
    one) that a single-mode target can't expose.
    """
    return (
        AMPLITUDE_1 * torch.sin(N_MODE * math.pi * x / L) * torch.cos(OMEGA * t)
        + AMPLITUDE_2 * torch.sin(N_MODE_2 * math.pi * x / L) * torch.cos(OMEGA_2 * t)
    )


def pde_residual(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """d^2E/dt^2 - c^2 * d^2E/dx^2 at (x, t), via autograd — zero where the wave equation holds."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    e = model(x, t)

    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_tt = torch.autograd.grad(e_t, t, grad_outputs=torch.ones_like(e_t), create_graph=True)[0]

    return e_tt - (C**2) * e_xx

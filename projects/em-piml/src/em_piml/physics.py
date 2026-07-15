from __future__ import annotations

import math

import torch

L = 1.0
C = 1.0
N_MODE = 1
AMPLITUDE = 1.0
OMEGA = N_MODE * math.pi * C / L
PERIOD = 2 * math.pi / OMEGA


def analytical_field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Closed-form E_z(x, t) for the fundamental mode of a 1D PEC cavity of length L."""
    return AMPLITUDE * torch.sin(N_MODE * math.pi * x / L) * torch.cos(OMEGA * t)


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

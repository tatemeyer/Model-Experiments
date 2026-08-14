from __future__ import annotations

import math

import torch

# 1D Helmholtz eigenvalue waveguide-mode problem (issue #43). Independent of em_piml.physics's L
# even though both happen to be 1.0 -- this is a genuinely different problem (no time dimension,
# no causality), not a variant of the cavity IVP, so it gets its own module rather than adding a
# time-independent branch to physics.py.
L = 1.0


def eigenvalue(mode_order: int) -> float:
    """k_n = n*pi/L: the Helmholtz eigenvalue for mode order n of a 1D PEC-bounded waveguide slice
    satisfying E''(x) + k_n^2 * E(x) = 0, E(0) = E(L) = 0. Given (not solved for) here -- this
    tests representational/optimization capacity for a known mode, not eigenvalue discovery."""
    return mode_order * math.pi / L


def anchor_x(mode_order: int) -> float:
    """x-location of mode n's first peak (sin(k_n * x) = 1). E=0 trivially satisfies both the PDE
    residual and the Dirichlet BCs for *any* amplitude/eigenvalue, so without an amplitude-pinning
    constraint a PINN could collapse to the trivial null solution -- the same "free" low-residual
    escape hatch documented in projects/em-piml/CLAUDE.md's long-horizon-collapse thread. This
    plays the role the initial condition plays in the time-domain cavity problem (physics.py)."""
    return L / (2 * mode_order)


def analytical_mode(x: torch.Tensor, mode_order: int) -> torch.Tensor:
    """Closed-form eigenfunction E_n(x) = sin(n*pi*x/L) for mode order n, amplitude 1."""
    return torch.sin(eigenvalue(mode_order) * x)


def helmholtz_residual(model: torch.nn.Module, x: torch.Tensor, mode_order: int) -> torch.Tensor:
    """d^2E/dx^2 + k_n^2 * E at x, via autograd -- zero where the Helmholtz eigenvalue equation
    holds for mode order n's (fixed, known) eigenvalue k_n."""
    x = x.clone().requires_grad_(True)
    e = model(x)
    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    k = eigenvalue(mode_order)
    return e_xx + (k**2) * e

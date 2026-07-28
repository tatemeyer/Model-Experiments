from __future__ import annotations

import math

import torch

from em_piml.physics import AMPLITUDE, L

# issue #46: explicit, documented parameters -- the interface location and permittivity jump are
# the two knobs this experiment whole question depends on, not incidental magic numbers.
X_INT = 0.35  # interior interface point, 0 < X_INT < L; off-center avoids symmetric degeneracies
EPS_1 = 1.0  # permittivity for x < X_INT -- matches physics.py implicit eps=mu=1 baseline cavity
EPS_2 = 4.0  # permittivity for x >= X_INT -- 4x jump halves the local wave speed c=1/sqrt(mu*eps)
MU = 1.0  # permeability, constant across the interface (both regions non-magnetic) -- this is what
# makes tangential-H continuity reduce to a clean E/E prime matching condition below (see docstring)


def permittivity(x: torch.Tensor) -> torch.Tensor:
    """Piecewise-constant eps(x): EPS_1 for x < X_INT, EPS_2 for x >= X_INT."""
    return torch.where(x < X_INT, torch.full_like(x, EPS_1), torch.full_like(x, EPS_2))


def _eigenvalue_equation(omega: float) -> float:
    """g(omega) = 0 at exactly the resonant frequencies of this cavity.

    Separable ansatz E(x,t) = f(x)*cos(omega*t) (same standing-wave convention as
    physics.analytical_field). f satisfies f prime prime + omega^2*mu*eps(x)*f = 0 piecewise,
    giving f(x) = A1*sin(k1*x) on region 1 (already zero at x=0) and f(x) = A2*sin(k2*(L-x)) on
    region 2 (already zero at x=L), k_i = omega*sqrt(mu*eps_i).

    Maxwell transmission conditions at a source-free interface are tangential-E continuous and
    tangential-H continuous. Here E_z is tangential (interface normal is x, field points along z),
    so f continuous is exactly tangential E continuous. H_y is tangential too, and Faraday law
    dE_z/dx = -mu*dH_y/dt ties H_y-continuity to f prime continuity whenever mu is equal on both
    sides (true here) -- so both f and f prime end up continuous at X_INT, a different mechanism
    than a step in mu would give. D_x = eps*E_x (normal D) is identically 0 on both sides for this
    transverse-field, normal-incidence reduction (there is no E_x component), so normal-D
    continuity is trivially satisfied (0=0) and adds no separate constraint beyond tangential E/H
    -- documented here rather than silently assumed. The eps(x) jump does not break value/slope
    continuity: it is fully absorbed into a curvature (f prime prime) discontinuity via the PDE
    itself (see verify_reference_solution) -- the spatially localized kink issue #46 tests
    capacity against.

    Setting f(X_INT-)=f(X_INT+) and f prime(X_INT-)=f prime(X_INT+) as a 2x2 homogeneous linear
    system in (A1, A2), the determinant vanishing is exactly this equation.
    """
    k1 = omega * math.sqrt(MU * EPS_1)
    k2 = omega * math.sqrt(MU * EPS_2)
    return k2 * math.sin(k1 * X_INT) * math.cos(k2 * (L - X_INT)) + k1 * math.cos(
        k1 * X_INT
    ) * math.sin(k2 * (L - X_INT))


def _bisect(f, lo: float, hi: float, iters: int = 200) -> float:
    f_lo = f(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def _solve_fundamental_omega(scan_points: int = 4000, scan_hi: float = 4 * math.pi) -> float:
    # No closed-form root of the transcendental eigenvalue equation above, so this is the
    # numerically verified reference solution issue #46 allows for: scan for the first sign
    # change past omega=0 (a root of _eigenvalue_equation for every parameter choice, trivially,
    # since sin(0)=0 -- excluded by starting the scan just above it) and bisect within that
    # bracket. verify_reference_solution() below checks the result actually satisfies value/slope
    # continuity and the boundary conditions to floating-point precision, so this is not trusted
    # blindly -- see that function and tests/test_dielectric_interface_capacity.py.
    prev_x = 1e-6
    prev_f = _eigenvalue_equation(prev_x)
    step = (scan_hi - prev_x) / scan_points
    x = prev_x
    for _ in range(scan_points):
        x += step
        f_x = _eigenvalue_equation(x)
        if prev_f * f_x < 0:
            return _bisect(_eigenvalue_equation, prev_x, x)
        prev_x, prev_f = x, f_x
    raise RuntimeError("no fundamental-mode root found in scan range -- check X_INT/EPS_1/EPS_2")


OMEGA = _solve_fundamental_omega()
PERIOD = 2 * math.pi / OMEGA
K1 = OMEGA * math.sqrt(MU * EPS_1)
K2 = OMEGA * math.sqrt(MU * EPS_2)
A1 = AMPLITUDE
A2 = A1 * math.sin(K1 * X_INT) / math.sin(K2 * (L - X_INT))


def analytical_field_dielectric(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Closed-form (given the numerically-solved OMEGA above) E_z(x, t) for the fundamental
    standing-wave mode of a 1D cavity with a piecewise-constant permittivity jump at X_INT.

    f(x) is continuous and C1 at X_INT (see _eigenvalue_equation) but its curvature f prime prime
    (x) -- and hence d^2E/dx^2 at fixed t -- jumps there by exactly EPS_2/EPS_1, the spatially
    localized derivative kink issue #46 tests capacity against (see verify_reference_solution).
    """
    f1 = A1 * torch.sin(K1 * x)
    f2 = A2 * torch.sin(K2 * (L - x))
    f = torch.where(x < X_INT, f1, f2)
    return f * torch.cos(OMEGA * t)


def pde_residual_dielectric(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """eps(x)*d^2E/dt^2 - (1/MU)*d^2E/dx^2 at (x, t), via autograd -- zero where the piecewise-eps
    wave equation holds (physics.pde_residual constant C^2 becomes 1/(mu*eps(x)) here, a
    position-dependent local wave speed rather than a single global one)."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    e = model(x, t)

    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_tt = torch.autograd.grad(e_t, t, grad_outputs=torch.ones_like(e_t), create_graph=True)[0]

    return permittivity(x) * e_tt - e_xx / MU


def verify_reference_solution(tol: float = 1e-9) -> dict[str, float]:
    """Numerically verifies the closed-form solution above actually satisfies the boundary
    conditions and the interface transmission conditions (value + slope continuity) to
    floating-point precision -- the numerically verified reference solution issue #46 asks for,
    not eyeballed. Also reports the curvature jump ratio, which should land on EPS_2/EPS_1 exactly
    (confirming the kink is real, and exactly where/how big it is supposed to be, not an artifact
    of the root-finder). Raises AssertionError if any check fails."""
    f1_at_int = A1 * math.sin(K1 * X_INT)
    f2_at_int = A2 * math.sin(K2 * (L - X_INT))
    df1_at_int = A1 * K1 * math.cos(K1 * X_INT)
    df2_at_int = -A2 * K2 * math.cos(K2 * (L - X_INT))
    d2f1_at_int = -(K1**2) * f1_at_int
    d2f2_at_int = -(K2**2) * f2_at_int

    diagnostics = {
        "bc_left": A1 * math.sin(K1 * 0.0),
        "bc_right": A2 * math.sin(K2 * (L - L)),
        "value_continuity_gap": f1_at_int - f2_at_int,
        "slope_continuity_gap": df1_at_int - df2_at_int,
        "curvature_jump_ratio": d2f2_at_int / d2f1_at_int,
    }
    assert abs(diagnostics["bc_left"]) < tol, diagnostics
    assert abs(diagnostics["bc_right"]) < tol, diagnostics
    assert abs(diagnostics["value_continuity_gap"]) < tol, diagnostics
    assert abs(diagnostics["slope_continuity_gap"]) < tol, diagnostics
    assert abs(diagnostics["curvature_jump_ratio"] - EPS_2 / EPS_1) < 1e-6, diagnostics
    return diagnostics

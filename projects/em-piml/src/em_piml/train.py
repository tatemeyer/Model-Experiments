from __future__ import annotations

import torch
from pytorch_optimizer import SOAP

from em_piml.model import CavityPINN, FourierCavityPINN
from em_piml.physics import PERIOD, L, analytical_field, pde_residual


def _pinn_loss(
    model: torch.nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
) -> torch.Tensor:
    loss_pde = (pde_residual(model, x_c, t_c) ** 2).mean()
    loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()
    loss_ic = ((model(x_i, t_i) - analytical_field(x_i, t_i)) ** 2).mean()

    t_i_grad = t_i.clone().requires_grad_(True)
    e_i = model(x_i, t_i_grad)
    e_i_dot = torch.autograd.grad(
        e_i, t_i_grad, grad_outputs=torch.ones_like(e_i), create_graph=True
    )[0]
    loss_ic_dot = (e_i_dot**2).mean()  # dE/dt(x, 0) = 0 for this standing-wave mode

    return loss_pde + loss_bc + loss_ic + loss_ic_dot


def _sample_points(
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, ...]:
    # generator=None draws from the global RNG (whatever torch.manual_seed set up before this
    # call) - the original behavior, kept as the default so existing callers/tests are
    # bit-for-bit unaffected. Passing an explicit generator (see train_fourier_cavity_lbfgs's
    # points_seed) decouples "which points get drawn" from the model-init seed.
    x_c = torch.rand(n_collocation, 1, generator=generator) * L
    t_c = torch.rand(n_collocation, 1, generator=generator) * PERIOD
    t_b = torch.rand(n_boundary, 1, generator=generator) * PERIOD
    x_b0 = torch.zeros(n_boundary, 1)
    x_bl = torch.full((n_boundary, 1), L)
    x_i = torch.rand(n_initial, 1, generator=generator) * L
    t_i = torch.zeros(n_initial, 1)
    return x_c, t_c, x_b0, x_bl, t_b, x_i, t_i


def _train_pinn_adam(
    model: torch.nn.Module,
    steps: int,
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    lr: float,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        points = _sample_points(n_collocation, n_boundary, n_initial)
        loss = _pinn_loss(model, *points)
        loss.backward()
        optimizer.step()

    return model


def _train_pinn_lbfgs(
    model: torch.nn.Module,
    outer_steps: int,
    max_iter: int,
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    points_generator: torch.Generator | None = None,
) -> torch.nn.Module:
    # L-BFGS assumes a fixed (deterministic) objective across its internal line-search
    # evaluations, so — unlike Adam — collocation points are sampled once, not per step.
    points = _sample_points(n_collocation, n_boundary, n_initial, generator=points_generator)
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=max_iter, history_size=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _pinn_loss(model, *points)
        loss.backward()
        return loss

    for _ in range(outer_steps):
        optimizer.step(closure)

    return model


def _train_pinn_soap(
    model: torch.nn.Module,
    steps: int,
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    lr: float,
) -> torch.nn.Module:
    # Same fixed (not resampled) point set as _train_pinn_lbfgs, so swapping the optimizer is
    # the only variable in the L-BFGS vs. SOAP comparison. SOAP doesn't need this determinism
    # (no line search), but keeping it matches the controlled-comparison premise.
    #
    # SOAP recomputes a Shampoo-style eigenbasis preconditioner every precondition_frequency
    # steps via extra linear algebra (eigh) on top of the Adam-like update — on this sandbox,
    # that made it ~200x more sensitive to CPU oversubscription (OpenMP thread contention with
    # unrelated concurrent processes) than Adam/L-BFGS at the same matrix sizes. Pinning to a
    # single thread avoids that thread-scheduling overhead; restored afterward so it doesn't
    # leak into other tests/training runs sharing this process.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        points = _sample_points(n_collocation, n_boundary, n_initial)
        optimizer = SOAP(model.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = _pinn_loss(model, *points)
            loss.backward()
            optimizer.step()
    finally:
        torch.set_num_threads(prior_threads)

    return model


def train_cavity_baseline(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
) -> CavityPINN:
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    return _train_pinn_adam(model, steps, n_collocation, n_boundary, n_initial, lr)


def train_fourier_cavity_baseline(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    num_bands: int = 2,
) -> FourierCavityPINN:
    torch.manual_seed(seed)
    model = FourierCavityPINN(hidden=32, num_layers=3, num_bands=num_bands)
    return _train_pinn_adam(model, steps, n_collocation, n_boundary, n_initial, lr)


def train_fourier_cavity_lbfgs(
    seed: int = 0,
    num_bands: int = 4,
    outer_steps: int = 50,
    max_iter: int = 50,
    n_collocation: int = 2000,
    n_boundary: int = 400,
    n_initial: int = 400,
    points_seed: int | None = None,
) -> FourierCavityPINN:
    # points_seed=None (default): points are drawn from the same global RNG stream as model
    # init, exactly as before (issue #8's behavior, unaffected). Passing an explicit
    # points_seed draws the collocation/boundary/initial set from an independent generator, so
    # `seed` (model init) and `points_seed` (which points get drawn) can be varied separately -
    # see issue #12, which needed this to tell "point count" and "which points" apart.
    torch.manual_seed(seed)
    # hidden=64 (up from 32, used everywhere else including num_bands=2) -- issue #10 found the
    # higher-dimensional Fourier-embedded input at num_bands=4 was capacity-bottlenecked at 32-wide;
    # see project CLAUDE.md. num_bands=2 is untouched (train_fourier_cavity_baseline still uses 32).
    model = FourierCavityPINN(hidden=64, num_layers=3, num_bands=num_bands)
    points_generator = None
    if points_seed is not None:
        points_generator = torch.Generator().manual_seed(points_seed)
    return _train_pinn_lbfgs(
        model,
        outer_steps,
        max_iter,
        n_collocation,
        n_boundary,
        n_initial,
        points_generator=points_generator,
    )


def train_fourier_cavity_soap(
    seed: int = 0,
    num_bands: int = 4,
    steps: int = 2000,
    n_collocation: int = 2000,
    n_boundary: int = 400,
    n_initial: int = 400,
    lr: float = 3e-3,
) -> FourierCavityPINN:
    torch.manual_seed(seed)
    model = FourierCavityPINN(hidden=32, num_layers=3, num_bands=num_bands)
    return _train_pinn_soap(model, steps, n_collocation, n_boundary, n_initial, lr)


def evaluate_relative_l2_error(model: torch.nn.Module, seed: int = 123) -> float:
    torch.manual_seed(seed)
    x = torch.rand(500, 1) * L
    t = torch.rand(500, 1) * PERIOD
    with torch.no_grad():
        predicted = model(x, t)
        true = analytical_field(x, t)
    return (torch.linalg.norm(predicted - true) / torch.linalg.norm(true)).item()


def main() -> None:
    baseline = train_cavity_baseline()
    fourier = train_fourier_cavity_baseline()
    fourier_lbfgs = train_fourier_cavity_lbfgs()
    fourier_soap = train_fourier_cavity_soap()
    print(f"baseline           relative L2 error: {evaluate_relative_l2_error(baseline):.4f}")
    print(f"fourier (adam)     relative L2 error: {evaluate_relative_l2_error(fourier):.4f}")
    print(f"fourier (lbfgs)    relative L2 error: {evaluate_relative_l2_error(fourier_lbfgs):.4f}")
    print(f"fourier (soap)     relative L2 error: {evaluate_relative_l2_error(fourier_soap):.4f}")


if __name__ == "__main__":
    main()

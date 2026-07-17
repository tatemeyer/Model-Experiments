from __future__ import annotations

from collections.abc import Callable

import torch
from pytorch_optimizer import SOAP

from em_piml.model import CavityPINN, FourierCavityPINN, PseudoSequenceCavityPINN, _pseudo_sequence
from em_piml.physics import PERIOD, C, L, analytical_field, analytical_field_two_mode, pde_residual


def _pinn_loss(
    model: torch.nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.Tensor:
    # field_fn defaults to the single-mode analytical_field (existing behavior, unaffected);
    # issue #22's two-mode training functions pass analytical_field_two_mode instead — dE/dt(x,0)
    # = 0 still holds for that target too (each mode's cos(.) has zero time-derivative at t=0, so
    # the sum's does too), so loss_ic_dot below needs no field_fn-specific change.
    loss_pde = (pde_residual(model, x_c, t_c) ** 2).mean()
    loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()
    loss_ic = ((model(x_i, t_i) - field_fn(x_i, t_i)) ** 2).mean()

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
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        points = _sample_points(n_collocation, n_boundary, n_initial)
        loss = _pinn_loss(model, *points, field_fn=field_fn)
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
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.nn.Module:
    # L-BFGS assumes a fixed (deterministic) objective across its internal line-search
    # evaluations, so — unlike Adam — collocation points are sampled once, not per step.
    points = _sample_points(n_collocation, n_boundary, n_initial, generator=points_generator)
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=max_iter, history_size=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _pinn_loss(model, *points, field_fn=field_fn)
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
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
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
            loss = _pinn_loss(model, *points, field_fn=field_fn)
            loss.backward()
            optimizer.step()
    finally:
        torch.set_num_threads(prior_threads)

    return model


def _sequence_derivative(u: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
    """Per-position Jacobian diagonal: d(u[:, j, :])/d(wrt[:, j, :]) for each sequence position j
    independently. Necessary because PseudoSequenceCavityPINN's self-attention mixes information
    across sequence positions, so a single torch.autograd.grad(u, wrt, grad_outputs=ones) call
    would sum cross-position contributions instead of isolating each position's own derivative —
    not what PINNsFormer eq. 5 (arXiv:2307.11833) asks for. One backward pass per sequence
    position; k is small (3 by default here) so this stays bounded, though in practice this
    project's implementation ended up far more expensive than the paper's own reported ~3-4x
    overhead — see projects/em-piml/CLAUDE.md issue #20 for why.
    """
    k = u.shape[1]
    columns = []
    for j in range(k):
        (grad_j,) = torch.autograd.grad(
            u[:, j, :],
            wrt,
            grad_outputs=torch.ones_like(u[:, j, :]),
            create_graph=True,
            retain_graph=True,
        )
        columns.append(grad_j[:, j : j + 1, :])
    return torch.cat(columns, dim=1)


def _pde_residual_sequence(
    model: PseudoSequenceCavityPINN, x_seq: torch.Tensor, t_seq: torch.Tensor
) -> torch.Tensor:
    x_seq = x_seq.clone().requires_grad_(True)
    t_seq = t_seq.clone().requires_grad_(True)
    e = model.forward_sequence(x_seq, t_seq)
    e_x = _sequence_derivative(e, x_seq)
    e_xx = _sequence_derivative(e_x, x_seq)
    e_t = _sequence_derivative(e, t_seq)
    e_tt = _sequence_derivative(e_t, t_seq)
    return e_tt - (C**2) * e_xx


def _pseudo_sequence_pinn_loss(
    model: PseudoSequenceCavityPINN,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
) -> torch.Tensor:
    x_c_seq, t_c_seq = _pseudo_sequence(x_c, t_c, model.k, model.dt)
    loss_pde = (_pde_residual_sequence(model, x_c_seq, t_c_seq) ** 2).mean()

    x_b0_seq, t_b_seq = _pseudo_sequence(x_b0, t_b, model.k, model.dt)
    x_bl_seq, _ = _pseudo_sequence(x_bl, t_b, model.k, model.dt)
    loss_bc = (model.forward_sequence(x_b0_seq, t_b_seq) ** 2).mean() + (
        model.forward_sequence(x_bl_seq, t_b_seq) ** 2
    ).mean()

    x_i_seq, t_i_seq = _pseudo_sequence(x_i, t_i, model.k, model.dt)
    e_i_seq = model.forward_sequence(x_i_seq, t_i_seq)
    # Only the first sequence position corresponds to the true initial condition t=0; later
    # positions have t=j*dt > 0 and aren't part of the IC constraint (PINNsFormer section 3.4).
    loss_ic = ((e_i_seq[:, 0, :] - analytical_field(x_i, t_i)) ** 2).mean()

    t_i_seq_grad = t_i_seq.clone().requires_grad_(True)
    e_i_for_grad = model.forward_sequence(x_i_seq, t_i_seq_grad)
    e_i_dot = _sequence_derivative(e_i_for_grad, t_i_seq_grad)[:, 0, :]
    loss_ic_dot = (e_i_dot**2).mean()

    return loss_pde + loss_bc + loss_ic + loss_ic_dot


def _train_pseudo_sequence_pinn_adam(
    model: PseudoSequenceCavityPINN,
    steps: int,
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    lr: float,
) -> PseudoSequenceCavityPINN:
    # Single-threaded for the same reason as _train_pinn_soap: this workload is dominated by many
    # small ops (_sequence_derivative does k backward passes per derivative, and the residual loss
    # needs 4 such calls for 2nd-order x/t derivatives), and default intra-op threading overhead
    # dominates at this scale — pinning to 1 thread measured ~3-4x faster in this project's
    # sandbox. Restored afterward so it doesn't leak into other training calls.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            points = _sample_points(n_collocation, n_boundary, n_initial)
            loss = _pseudo_sequence_pinn_loss(model, *points)
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


def train_fourier_cavity_lbfgs_two_mode(
    seed: int = 0,
    num_bands: int = 4,
    outer_steps: int = 50,
    max_iter: int = 50,
    n_collocation: int = 2000,
    n_boundary: int = 400,
    n_initial: int = 400,
    points_seed: int | None = None,
) -> FourierCavityPINN:
    # issue #25: same shipped recipe as train_fourier_cavity_lbfgs (issue #10's capacity fix,
    # issue #8's density fix) -- only field_fn differs, same pattern as train_*_two_mode above.
    # Adam destabilizes at num_bands>=2 on this target the same way issue #4 found on the
    # single-mode baseline (see CLAUDE.md), so this and the SOAP variant below are what let
    # num_bands be tested at all past 2 without confounding "can't train" with "can't represent."
    torch.manual_seed(seed)
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
        field_fn=analytical_field_two_mode,
    )


def train_fourier_cavity_soap_two_mode(
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
    return _train_pinn_soap(
        model, steps, n_collocation, n_boundary, n_initial, lr, field_fn=analytical_field_two_mode
    )


def train_pseudo_sequence_cavity(
    steps: int = 600,
    seed: int = 0,
    n_collocation: int = 30,
    n_boundary: int = 16,
    n_initial: int = 16,
    lr: float = 3e-3,
    k: int = 3,
    dt: float = 1e-3,
) -> PseudoSequenceCavityPINN:
    torch.manual_seed(seed)
    model = PseudoSequenceCavityPINN(k=k, dt=dt)
    return _train_pseudo_sequence_pinn_adam(model, steps, n_collocation, n_boundary, n_initial, lr)


def train_cavity_two_mode(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
) -> CavityPINN:
    # Same shipped defaults as train_cavity_baseline (issue #22's constraint: only the target
    # field's mode content changes, everything else held fixed) -- only field_fn differs.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    return _train_pinn_adam(
        model, steps, n_collocation, n_boundary, n_initial, lr, field_fn=analytical_field_two_mode
    )


def train_fourier_cavity_two_mode(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    num_bands: int = 2,
) -> FourierCavityPINN:
    # Same shipped defaults as train_fourier_cavity_baseline, including num_bands=2 -- see
    # projects/em-piml/CLAUDE.md issue #22 for why that default's frequency coverage is itself
    # part of what's being tested here, not something to silently retune.
    torch.manual_seed(seed)
    model = FourierCavityPINN(hidden=32, num_layers=3, num_bands=num_bands)
    return _train_pinn_adam(
        model, steps, n_collocation, n_boundary, n_initial, lr, field_fn=analytical_field_two_mode
    )


def evaluate_relative_l2_error(
    model: torch.nn.Module,
    seed: int = 123,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> float:
    torch.manual_seed(seed)
    x = torch.rand(500, 1) * L
    t = torch.rand(500, 1) * PERIOD
    with torch.no_grad():
        predicted = model(x, t)
        true = field_fn(x, t)
    return (torch.linalg.norm(predicted - true) / torch.linalg.norm(true)).item()


def main() -> None:
    baseline = train_cavity_baseline()
    fourier = train_fourier_cavity_baseline()
    fourier_lbfgs = train_fourier_cavity_lbfgs()
    fourier_soap = train_fourier_cavity_soap()
    pseudo_sequence = train_pseudo_sequence_cavity()
    print(f"baseline           relative L2 error: {evaluate_relative_l2_error(baseline):.4f}")
    print(f"fourier (adam)     relative L2 error: {evaluate_relative_l2_error(fourier):.4f}")
    print(f"fourier (lbfgs)    relative L2 error: {evaluate_relative_l2_error(fourier_lbfgs):.4f}")
    print(f"fourier (soap)     relative L2 error: {evaluate_relative_l2_error(fourier_soap):.4f}")
    ps_err = evaluate_relative_l2_error(pseudo_sequence)
    print(f"pseudo-sequence    relative L2 error: {ps_err:.4f}")

    two_mode = train_cavity_two_mode()
    two_mode_fourier = train_fourier_cavity_two_mode()
    two_mode_err = evaluate_relative_l2_error(two_mode, field_fn=analytical_field_two_mode)
    two_mode_fourier_err = evaluate_relative_l2_error(
        two_mode_fourier, field_fn=analytical_field_two_mode
    )
    print(f"two-mode (plain)   relative L2 error: {two_mode_err:.4f}")
    print(f"two-mode (fourier) relative L2 error: {two_mode_fourier_err:.4f}")


if __name__ == "__main__":
    main()

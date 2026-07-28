from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from pytorch_optimizer import SOAP

from em_piml.dielectric import PERIOD as DIELECTRIC_PERIOD
from em_piml.dielectric import analytical_field_dielectric, pde_residual_dielectric
from em_piml.model import CavityPINN, FourierCavityPINN, PseudoSequenceCavityPINN, _pseudo_sequence
from em_piml.physics import PERIOD, C, L, analytical_field, analytical_field_two_mode, pde_residual


def _pinn_loss_components(
    model: torch.nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # field_fn defaults to the single-mode analytical_field (existing behavior, unaffected);
    # issue #22's two-mode training functions pass analytical_field_two_mode instead — dE/dt(x,0)
    # = 0 still holds for that target too (each mode's cos(.) has zero time-derivative at t=0, so
    # the sum's does too), so loss_ic_dot below needs no field_fn-specific change.
    # Split out from _pinn_loss (issue #34) so the NTK-reweighting training loop can compute each
    # term's own gradient norm separately -- every other caller still goes through _pinn_loss,
    # unaffected.
    loss_pde = (pde_residual(model, x_c, t_c) ** 2).mean()
    loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()
    loss_ic = ((model(x_i, t_i) - field_fn(x_i, t_i)) ** 2).mean()

    t_i_grad = t_i.clone().requires_grad_(True)
    e_i = model(x_i, t_i_grad)
    e_i_dot = torch.autograd.grad(
        e_i, t_i_grad, grad_outputs=torch.ones_like(e_i), create_graph=True
    )[0]
    loss_ic_dot = (e_i_dot**2).mean()  # dE/dt(x, 0) = 0 for this standing-wave mode

    return loss_pde, loss_bc, loss_ic, loss_ic_dot


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
    loss_pde, loss_bc, loss_ic, loss_ic_dot = _pinn_loss_components(
        model, x_c, t_c, x_b0, x_bl, t_b, x_i, t_i, field_fn=field_fn
    )
    return loss_pde + loss_bc + loss_ic + loss_ic_dot


def _sample_points(
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    generator: torch.Generator | None = None,
    t_max: float = PERIOD,
) -> tuple[torch.Tensor, ...]:
    # generator=None draws from the global RNG (whatever torch.manual_seed set up before this
    # call) - the original behavior, kept as the default so existing callers/tests are
    # bit-for-bit unaffected. Passing an explicit generator (see train_fourier_cavity_lbfgs's
    # points_seed) decouples "which points get drawn" from the model-init seed. t_max=PERIOD
    # (default) preserves the original single-period domain; issue #23's long-horizon functions
    # pass a multiple of PERIOD instead.
    x_c = torch.rand(n_collocation, 1, generator=generator) * L
    t_c = torch.rand(n_collocation, 1, generator=generator) * t_max
    t_b = torch.rand(n_boundary, 1, generator=generator) * t_max
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
    t_max: float = PERIOD,
    history: list[float] | None = None,
) -> torch.nn.Module:
    # history=None (default) preserves the original behavior exactly, zero-cost -- same opt-in
    # idiom as _sample_points' generator=None. Pass a list to record loss.item() every step (e.g.
    # for mx_viz.training.plot_loss_curve); every existing call site is bit-for-bit unaffected.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        points = _sample_points(n_collocation, n_boundary, n_initial, t_max=t_max)
        loss = _pinn_loss(model, *points, field_fn=field_fn)
        loss.backward()
        optimizer.step()
        if history is not None:
            history.append(loss.item())

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
    t_max: float = PERIOD,
) -> PseudoSequenceCavityPINN:
    # Single-threaded for the same reason as _train_pinn_soap: this workload is dominated by many
    # small ops (_sequence_derivative does k backward passes per derivative, and the residual loss
    # needs 4 such calls for 2nd-order x/t derivatives), and default intra-op threading overhead
    # dominates at this scale — pinning to 1 thread measured ~3-4x faster in this project's
    # sandbox. Restored afterward so it doesn't leak into other training calls.
    # t_max=PERIOD (default) preserves train_pseudo_sequence_cavity's original single-period
    # domain; issue #30's long-horizon variant passes a multiple of PERIOD instead, matching
    # issue #23's t_max plumbing through _sample_points/_train_pinn_adam.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            points = _sample_points(n_collocation, n_boundary, n_initial, t_max=t_max)
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


def train_pseudo_sequence_cavity_long_horizon(
    steps: int = 600,
    seed: int = 0,
    n_collocation: int = 30,
    n_boundary: int = 16,
    n_initial: int = 16,
    lr: float = 3e-3,
    k: int = 3,
    dt: float = 1e-3,
    horizon_periods: float = 5.0,
) -> PseudoSequenceCavityPINN:
    # issue #30: same shipped config as train_pseudo_sequence_cavity (issue #20) -- only the
    # training/eval time domain changes (t_max = horizon_periods * PERIOD instead of one PERIOD),
    # mirroring train_cavity_long_horizon's (issue #23) t_max-only variable, so this is a direct
    # comparison against that issue's plain-baseline (~0.96) and causal-reweighted (~0.96) numbers
    # on the exact same long-horizon target.
    torch.manual_seed(seed)
    model = PseudoSequenceCavityPINN(k=k, dt=dt)
    return _train_pseudo_sequence_pinn_adam(
        model, steps, n_collocation, n_boundary, n_initial, lr, t_max=horizon_periods * PERIOD
    )


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


def train_cavity_long_horizon(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    horizon_periods: float = 5.0,
    history: list[float] | None = None,
) -> CavityPINN:
    # issue #23: same shipped defaults as train_cavity_baseline -- only the training/eval time
    # domain changes (t_max = horizon_periods * PERIOD instead of one PERIOD), uniform loss
    # weighting throughout (the causal variant below is the only variable-controlled comparison).
    # history is optional loss-curve recording (see _train_pinn_adam), for mx_viz.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    return _train_pinn_adam(
        model,
        steps,
        n_collocation,
        n_boundary,
        n_initial,
        lr,
        t_max=horizon_periods * PERIOD,
        history=history,
    )


def _sample_points_causal(
    n_per_chunk: int,
    n_chunks: int,
    t_max: float,
    n_boundary: int,
    n_initial: int,
) -> tuple[torch.Tensor, ...]:
    # Collocation points are stratified into n_chunks equal-width, temporally ordered bins (not
    # drawn uniformly at random over the whole domain like _sample_points) so that every training
    # step has a guaranteed, non-empty sample from every time chunk -- required to compute a
    # per-chunk residual loss each step (see _causal_pinn_loss). Points are laid out chunk-major
    # (all of chunk 0, then all of chunk 1, ...) so a single pde_residual call's output can be
    # reshaped into (n_chunks, n_per_chunk) without a second autograd pass per chunk.
    chunk_width = t_max / n_chunks
    chunk_offsets = torch.arange(n_chunks).repeat_interleave(n_per_chunk).unsqueeze(1).float()
    n_total = n_chunks * n_per_chunk
    x_c = torch.rand(n_total, 1) * L
    t_c = chunk_offsets * chunk_width + torch.rand(n_total, 1) * chunk_width
    t_b = torch.rand(n_boundary, 1) * t_max
    x_b0 = torch.zeros(n_boundary, 1)
    x_bl = torch.full((n_boundary, 1), L)
    x_i = torch.rand(n_initial, 1) * L
    t_i = torch.zeros(n_initial, 1)
    return x_c, t_c, x_b0, x_bl, t_b, x_i, t_i


def _causal_pinn_loss(
    model: torch.nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    n_chunks: int,
    epsilon: float,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.Tensor:
    # Causal weighting (Wang, Sankaran, Perdikaris, arXiv:2203.07404): the PDE residual loss for
    # time chunk i is down-weighted by exp(-epsilon * sum of *earlier* chunks' residual losses),
    # so a chunk's contribution to the gradient stays small until every chunk before it is
    # already well satisfied -- unlike uniform weighting, which lets the optimizer "solve" late
    # timesteps before early ones converge. Only the PDE residual term is causally weighted, per
    # the paper's own formulation (section 3.2, eq. 11-13) -- BC/IC terms stay uniformly weighted,
    # same as the uniform baseline, since causality is specifically a claim about how residual
    # satisfaction should propagate forward in time, not about the initial/boundary constraints
    # themselves. The cumulative-loss term inside exp() is detached: it's a per-chunk weight, not
    # something to backprop through (matches the paper's stop-gradient treatment of the weights).
    residual = pde_residual(model, x_c, t_c)
    chunk_losses = (residual**2).view(n_chunks, -1).mean(dim=1)
    with torch.no_grad():
        cumulative_prior = torch.cumsum(chunk_losses, dim=0) - chunk_losses
        weights = torch.exp(-epsilon * cumulative_prior)
    loss_pde = (weights * chunk_losses).sum() / weights.sum()

    loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()
    loss_ic = ((model(x_i, t_i) - field_fn(x_i, t_i)) ** 2).mean()

    t_i_grad = t_i.clone().requires_grad_(True)
    e_i = model(x_i, t_i_grad)
    e_i_dot = torch.autograd.grad(
        e_i, t_i_grad, grad_outputs=torch.ones_like(e_i), create_graph=True
    )[0]
    loss_ic_dot = (e_i_dot**2).mean()

    return loss_pde + loss_bc + loss_ic + loss_ic_dot


def _train_pinn_adam_causal(
    model: torch.nn.Module,
    steps: int,
    n_per_chunk: int,
    n_chunks: int,
    t_max: float,
    n_boundary: int,
    n_initial: int,
    lr: float,
    epsilon: float,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        points = _sample_points_causal(n_per_chunk, n_chunks, t_max, n_boundary, n_initial)
        loss = _causal_pinn_loss(
            model, *points, n_chunks=n_chunks, epsilon=epsilon, field_fn=field_fn
        )
        loss.backward()
        optimizer.step()

    return model


def train_cavity_causal_long_horizon(
    steps: int = 4000,
    seed: int = 0,
    n_per_chunk: int = 20,
    n_chunks: int = 10,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    horizon_periods: float = 5.0,
    epsilon: float = 1.0,
) -> CavityPINN:
    # n_per_chunk * n_chunks = 200 == train_cavity_long_horizon's n_collocation default, so the
    # two functions compare the same total collocation-point budget over the same extended
    # domain -- causal chunking/weighting is the only variable, per issue #23's constraint.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    t_max = horizon_periods * PERIOD
    return _train_pinn_adam_causal(
        model, steps, n_per_chunk, n_chunks, t_max, n_boundary, n_initial, lr, epsilon
    )


def train_cavity_curriculum_long_horizon(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    horizon_periods: float = 5.0,
    n_stages: int = 5,
) -> CavityPINN:
    # issue #32: same total step budget, point counts, lr, and architecture as
    # train_cavity_long_horizon/train_cavity_causal_long_horizon (issue #23) -- the only variable
    # is *when* the model sees which part of the time domain. Per Krishnapriyan et al.
    # ("Characterizing possible failure modes in physics-informed neural networks," NeurIPS 2021,
    # arXiv:2109.01050), training is split into n_stages stages of equal step count, each stage
    # widening t_max by one more increment of PERIOD (stage k trains on t in [0, k/n_stages *
    # horizon_periods * PERIOD]) -- the model must fit the shorter horizon before ever seeing the
    # next chunk of the domain, rather than being exposed to the full 5-period domain from step 1.
    # The same model/optimizer state carries across stages (a fresh _train_pinn_adam call per
    # stage, continuing from the previous stage's weights) -- a fresh Adam optimizer each stage,
    # not persisted across the t_max change, since Adam's per-parameter moment estimates were
    # tuned against a different (narrower) point distribution in the prior stage.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    steps_per_stage = steps // n_stages
    for stage in range(1, n_stages + 1):
        stage_t_max = (stage / n_stages) * horizon_periods * PERIOD
        model = _train_pinn_adam(
            model, steps_per_stage, n_collocation, n_boundary, n_initial, lr, t_max=stage_t_max
        )
    return model


def _grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    # L2 norm of d(loss)/d(params), flattened across all parameter tensors. allow_unused=True
    # because loss_ic_dot's graph only touches the layers t actually flows through in this simple
    # MLP -- in practice every loss term here touches every parameter (the whole network sits
    # between input and output), but this is defensive rather than assumed.
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = torch.zeros(())
    for g in grads:
        if g is not None:
            total = total + g.pow(2).sum()
    return total.sqrt()


def _train_pinn_adam_ntk_reweighted(
    model: torch.nn.Module,
    steps: int,
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    lr: float,
    t_max: float,
    update_every: int = 10,
    alpha: float = 0.9,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.nn.Module:
    # NTK-based adaptive loss-term reweighting (Wang, Teng, Perdikaris, "Understanding and
    # Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks," SIAM J. Sci.
    # Comput. 2021, arXiv:2001.04536, Algorithm 1 -- the paper's practical "learning rate
    # annealing" scheme, motivated by their NTK eigenvalue analysis). Unlike issue #23's causal
    # reweighting (which only reweights the PDE-residual term across time chunks) or issue #32's
    # curriculum (which changes the training schedule, not the loss balance), this rebalances the
    # *global* weight of each non-PDE loss term against the PDE-residual term's own gradient
    # magnitude: every update_every steps, each term i's weight is nudged toward
    # hat_lambda_i = max(|grad(loss_pde)|) / mean(|grad(loss_i)|) (approximated here via the L2
    # norm rather than a per-parameter max/mean, since this project's network is small enough that
    # this is a reasonable proxy and avoids per-parameter bookkeeping the paper's own public
    # implementations don't require either), smoothed via an exponential moving average
    # (weight_i <- (1 - alpha) * weight_i + alpha * hat_lambda_i, alpha=0.9 per the paper) so a
    # single noisy step doesn't cause the weights to swing wildly. loss_pde's own weight stays
    # fixed at 1.0 (the reference term every other term is balanced against, per the paper).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    params = list(model.parameters())
    weight_bc, weight_ic, weight_ic_dot = 1.0, 1.0, 1.0
    eps = 1e-8

    for step in range(steps):
        optimizer.zero_grad()
        points = _sample_points(n_collocation, n_boundary, n_initial, t_max=t_max)
        loss_pde, loss_bc, loss_ic, loss_ic_dot = _pinn_loss_components(
            model, *points, field_fn=field_fn
        )

        if step % update_every == 0:
            g_pde = _grad_norm(loss_pde, params)
            g_bc = _grad_norm(loss_bc, params)
            g_ic = _grad_norm(loss_ic, params)
            g_ic_dot = _grad_norm(loss_ic_dot, params)
            hat_bc = (g_pde / (g_bc + eps)).item()
            hat_ic = (g_pde / (g_ic + eps)).item()
            hat_ic_dot = (g_pde / (g_ic_dot + eps)).item()
            weight_bc = (1 - alpha) * weight_bc + alpha * hat_bc
            weight_ic = (1 - alpha) * weight_ic + alpha * hat_ic
            weight_ic_dot = (1 - alpha) * weight_ic_dot + alpha * hat_ic_dot

        loss = loss_pde + weight_bc * loss_bc + weight_ic * loss_ic + weight_ic_dot * loss_ic_dot
        loss.backward()
        optimizer.step()

    return model


def train_cavity_ntk_reweighted_long_horizon(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    horizon_periods: float = 5.0,
    update_every: int = 10,
    alpha: float = 0.9,
) -> CavityPINN:
    # issue #34: same shipped architecture/step budget/point counts/lr as
    # train_cavity_long_horizon (issue #23) -- the only variable is per-loss-term weighting via
    # NTK-based adaptive reweighting (see _train_pinn_adam_ntk_reweighted) instead of issue #23's
    # causal per-time-chunk reweighting or uniform weighting.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    t_max = horizon_periods * PERIOD
    return _train_pinn_adam_ntk_reweighted(
        model, steps, n_collocation, n_boundary, n_initial, lr, t_max, update_every, alpha
    )


def _pde_residual_and_input_grad_sq(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Leiteritz & Pflueger, "How to Avoid Trivial Solutions in Physics-Informed Neural Networks,"
    # arXiv:2112.05620, eq. 8/13: they observed the trivial (near-constant/zero) solution is
    # reachable because the raw PDE residual can jump from "tracking the true solution" to
    # "satisfied by the trivial solution" between two collocation points without penalty, and that
    # jump shows up as a spike in the residual's own gradient. Their fix adds max_i (d(residual)/dt
    # at collocation point i)^2 to the loss -- a smoothness/anti-spike penalty on the residual field
    # itself, distinct from every fix already tried in this project's long-horizon thread (none of
    # which touch the residual's own derivative). Their benchmark (1D harmonic oscillator) has one
    # input dimension (t) so eq. 13's bracketed term is literally d(residual)/dt; this project's
    # wave equation has two (x, t), so this generalizes it to the squared L2 norm of the residual's
    # gradient over both inputs, ||(dr/dx, dr/dt)||^2 -- same "residual should not spike" intent,
    # extended to a 2D domain rather than arbitrarily picking one axis.
    #
    # Duplicates pde_residual's derivative chain (rather than calling it directly) because the
    # further autograd.grad call below needs a handle to the exact x/t leaf tensors the residual
    # was computed from -- pde_residual clones its inputs internally and doesn't expose that clone,
    # so reusing it here would silently differentiate w.r.t. a disconnected leaf instead of x, t.
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    e = model(x, t)

    e_x = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x, x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    e_t = torch.autograd.grad(e, t, grad_outputs=torch.ones_like(e), create_graph=True)[0]
    e_tt = torch.autograd.grad(e_t, t, grad_outputs=torch.ones_like(e_t), create_graph=True)[0]
    residual = e_tt - (C**2) * e_xx

    r_x, r_t = torch.autograd.grad(
        residual, [x, t], grad_outputs=torch.ones_like(residual), create_graph=True
    )
    grad_sq = r_x**2 + r_t**2
    return residual, grad_sq


def _antitrivial_pinn_loss(
    model: torch.nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
    lambda_grad: float = 1.0,
) -> torch.Tensor:
    # Same four terms as _pinn_loss (PDE residual, BC, IC, dE/dt(x,0)=0), plus one new term: the
    # anti-trivial-solution penalty (see _pde_residual_and_input_grad_sq), weighted by lambda_grad
    # (default 1.0, matching the paper's own eq. 13, which adds the term unscaled, no coefficient).
    residual, grad_sq = _pde_residual_and_input_grad_sq(model, x_c, t_c)
    loss_pde = (residual**2).mean()
    loss_antitrivial = grad_sq.max()  # eq. 8's max_i, not mean -- penalizes the worst spike only

    loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()
    loss_ic = ((model(x_i, t_i) - field_fn(x_i, t_i)) ** 2).mean()

    t_i_grad = t_i.clone().requires_grad_(True)
    e_i = model(x_i, t_i_grad)
    e_i_dot = torch.autograd.grad(
        e_i, t_i_grad, grad_outputs=torch.ones_like(e_i), create_graph=True
    )[0]
    loss_ic_dot = (e_i_dot**2).mean()

    return loss_pde + loss_bc + loss_ic + loss_ic_dot + lambda_grad * loss_antitrivial


def _train_pinn_adam_antitrivial(
    model: torch.nn.Module,
    steps: int,
    n_collocation: int,
    n_boundary: int,
    n_initial: int,
    lr: float,
    t_max: float,
    lambda_grad: float = 1.0,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
) -> torch.nn.Module:
    # Single-threaded for the same reason as _train_pinn_soap/_train_pseudo_sequence_pinn_adam:
    # the anti-trivial penalty needs a third-order derivative (one more autograd.grad call beyond
    # pde_residual's own second-order one), and this project's sandbox has repeatedly shown that
    # torch's default intra-op threading compounds badly under the concurrent-session CPU
    # oversubscription documented in CLAUDE.md (issue #10) -- pinning to 1 thread avoided the
    # thread-storm for SOAP/pseudo-sequence and is applied here pre-emptively for the same reason.
    # Restored afterward so it doesn't leak into other training calls sharing this process.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            points = _sample_points(n_collocation, n_boundary, n_initial, t_max=t_max)
            loss = _antitrivial_pinn_loss(
                model, *points, field_fn=field_fn, lambda_grad=lambda_grad
            )
            loss.backward()
            optimizer.step()
    finally:
        torch.set_num_threads(prior_threads)

    return model


def train_cavity_antitrivial_long_horizon(
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
    horizon_periods: float = 5.0,
    lambda_grad: float = 1.0,
) -> CavityPINN:
    # issue #35: same shipped architecture/step budget/point counts/lr as
    # train_cavity_long_horizon (issue #23) -- the only variable is the added anti-trivial-solution
    # regularization term (_antitrivial_pinn_loss), per Leiteritz & Pflueger (arXiv:2112.05620),
    # instead of uniform/causal/curriculum/NTK-reweighted loss handling.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=32, num_layers=3)
    t_max = horizon_periods * PERIOD
    return _train_pinn_adam_antitrivial(
        model, steps, n_collocation, n_boundary, n_initial, lr, t_max, lambda_grad
    )


def _dielectric_pinn_loss(
    model: torch.nn.Module,
    x_c: torch.Tensor,
    t_c: torch.Tensor,
    x_b0: torch.Tensor,
    x_bl: torch.Tensor,
    t_b: torch.Tensor,
    x_i: torch.Tensor,
    t_i: torch.Tensor,
) -> torch.Tensor:
    # issue #46: same four-term loss shape as _pinn_loss, but the PDE residual uses the
    # piecewise-eps(x) wave equation (pde_residual_dielectric) and IC/IC-dot are supervised
    # against the kinked reference solution (analytical_field_dielectric) instead of the
    # homogeneous-cavity one -- everything else (BC terms, loss structure) is unchanged.
    loss_pde = (pde_residual_dielectric(model, x_c, t_c) ** 2).mean()
    loss_bc = (model(x_b0, t_b) ** 2).mean() + (model(x_bl, t_b) ** 2).mean()
    loss_ic = ((model(x_i, t_i) - analytical_field_dielectric(x_i, t_i)) ** 2).mean()

    t_i_grad = t_i.clone().requires_grad_(True)
    e_i = model(x_i, t_i_grad)
    e_i_dot = torch.autograd.grad(
        e_i, t_i_grad, grad_outputs=torch.ones_like(e_i), create_graph=True
    )[0]
    loss_ic_dot = (e_i_dot**2).mean()  # dE/dt(x,0)=0 holds here too: cos(OMEGA*t) at t=0

    return loss_pde + loss_bc + loss_ic + loss_ic_dot


def _train_dielectric_pinn_adam(
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
        points = _sample_points(n_collocation, n_boundary, n_initial, t_max=DIELECTRIC_PERIOD)
        loss = _dielectric_pinn_loss(model, *points)
        loss.backward()
        optimizer.step()
    return model


def train_dielectric_cavity(
    hidden: int = 64,
    num_layers: int = 3,
    steps: int = 4000,
    seed: int = 0,
    n_collocation: int = 200,
    n_boundary: int = 64,
    n_initial: int = 64,
    lr: float = 3e-3,
) -> CavityPINN:
    # issue #46: same CavityPINN architecture family/training recipe as train_cavity_baseline
    # (plain coordinate MLP, Adam, same step/point-count/lr defaults) -- hidden is the swept
    # capacity variable (num_layers held fixed at 3, the baseline depth), everything else held at
    # the project's standard single-mode recipe so this is a controlled comparison against #25's
    # capacity finding, not a differently-tuned training setup entangled with the new physics.
    torch.manual_seed(seed)
    model = CavityPINN(hidden=hidden, num_layers=num_layers)
    return _train_dielectric_pinn_adam(model, steps, n_collocation, n_boundary, n_initial, lr)


def evaluate_relative_l2_error(
    model: torch.nn.Module,
    seed: int = 123,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
    t_max: float = PERIOD,
) -> float:
    torch.manual_seed(seed)
    x = torch.rand(500, 1) * L
    t = torch.rand(500, 1) * t_max
    with torch.no_grad():
        predicted = model(x, t)
        true = field_fn(x, t)
    return (torch.linalg.norm(predicted - true) / torch.linalg.norm(true)).item()


def evaluate_field_grid(
    model: torch.nn.Module,
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
    t_max: float = PERIOD,
    x_range: tuple[float, float] = (0.0, L),
    n_x: int = 100,
    n_t: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Evaluates a trained model + its ground-truth field over a regular (x, t) grid and returns
    # plain numpy arrays (x_grid, t_grid, predicted, true) -- mx_viz.fields.plot_field_heatmap
    # takes numpy arrays rather than model objects to stay framework-agnostic (see tools/viz's
    # CLAUDE.md-equivalent rationale); this is the torch-specific "evaluate over a grid" half that
    # belongs here, not in the plotting library.
    xs = torch.linspace(x_range[0], x_range[1], n_x)
    ts = torch.linspace(0.0, t_max, n_t)
    grid_x, grid_t = torch.meshgrid(xs, ts, indexing="xy")
    x_flat = grid_x.reshape(-1, 1)
    t_flat = grid_t.reshape(-1, 1)
    with torch.no_grad():
        predicted = model(x_flat, t_flat).reshape(grid_x.shape)
        true = field_fn(x_flat, t_flat).reshape(grid_x.shape)
    return grid_x.numpy(), grid_t.numpy(), predicted.numpy(), true.numpy()


def evaluate_field_slice(
    model: torch.nn.Module,
    t_values: Sequence[float],
    field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
    x: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    # Evaluates predicted vs. true field across t_values at a fixed x -- the numeric half of the
    # pointwise checks this project has hand-tabulated as CLAUDE.md markdown tables since issue
    # #23; mx_viz.fields.plot_field_slice turns the same arrays into a line plot.
    t = torch.tensor(t_values, dtype=torch.float32).reshape(-1, 1)
    x_t = torch.full_like(t, x)
    with torch.no_grad():
        predicted = model(x_t, t).reshape(-1)
        true = field_fn(x_t, t).reshape(-1)
    return predicted.numpy(), true.numpy()


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

    long_horizon = train_cavity_long_horizon()
    long_horizon_causal = train_cavity_causal_long_horizon()
    long_horizon_pseudo_sequence = train_pseudo_sequence_cavity_long_horizon()
    long_horizon_curriculum = train_cavity_curriculum_long_horizon()
    long_horizon_ntk = train_cavity_ntk_reweighted_long_horizon()
    long_horizon_antitrivial = train_cavity_antitrivial_long_horizon()
    t_max = 5.0 * PERIOD
    long_horizon_err = evaluate_relative_l2_error(long_horizon, t_max=t_max)
    long_horizon_causal_err = evaluate_relative_l2_error(long_horizon_causal, t_max=t_max)
    long_horizon_pseudo_sequence_err = evaluate_relative_l2_error(
        long_horizon_pseudo_sequence, t_max=t_max
    )
    long_horizon_curriculum_err = evaluate_relative_l2_error(long_horizon_curriculum, t_max=t_max)
    long_horizon_ntk_err = evaluate_relative_l2_error(long_horizon_ntk, t_max=t_max)
    long_horizon_antitrivial_err = evaluate_relative_l2_error(long_horizon_antitrivial, t_max=t_max)
    print(f"long-horizon (uniform)         relative L2 error: {long_horizon_err:.4f}")
    print(f"long-horizon (causal)          relative L2 error: {long_horizon_causal_err:.4f}")
    print(
        f"long-horizon (pseudo-sequence) relative L2 error: "
        f"{long_horizon_pseudo_sequence_err:.4f}"
    )
    print(f"long-horizon (curriculum)      relative L2 error: {long_horizon_curriculum_err:.4f}")
    print(f"long-horizon (NTK-reweighted)  relative L2 error: {long_horizon_ntk_err:.4f}")
    print(f"long-horizon (anti-trivial)    relative L2 error: {long_horizon_antitrivial_err:.4f}")


if __name__ == "__main__":
    main()

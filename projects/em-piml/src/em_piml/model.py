from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from em_piml.embeddings import FourierFeatureEmbedding
from em_piml.physics import PERIOD, C, L, analytical_field


def _mlp(dims: list[int]) -> nn.Sequential:
    modules: list[nn.Module] = []
    for i in range(len(dims) - 1):
        modules.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class CavityPINN(nn.Module):
    """Coordinate-input MLP mapping (x, t) -> predicted E_z. Deliberately plain — the baseline."""

    def __init__(self, hidden: int = 64, num_layers: int = 4):
        super().__init__()
        self.net = _mlp([2] + [hidden] * num_layers + [1])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class FourierCavityPINN(nn.Module):
    """Same MLP body shape as CavityPINN; (x, t) pass through a Fourier feature embedding first."""

    def __init__(self, hidden: int = 32, num_layers: int = 3, num_bands: int = 4):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(num_bands=num_bands)
        in_dim = 2 * self.embedding.out_dim_per_scalar
        self.net = _mlp([in_dim] + [hidden] * num_layers + [1])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x / L, t / PERIOD)
        return self.net(embedded)


class RWFLinear(nn.Module):
    """Random Weight Factorization (Wang, Wang, Seidman, Perdikaris, "Random Weight Factorization
    Improves the Training of Continuous Neural Representations," arXiv:2210.01274) -- a drop-in
    replacement for nn.Linear that reparameterizes the weight matrix W (out_features, in_features)
    as W = diag(s) @ V: a per-output-neuron scalar factor s and a matrix V of the same shape as W,
    both trained directly instead of W itself. This changes the effective per-row gradient scaling
    during training (the paper's claimed mechanism for improving loss-landscape conditioning /
    mitigating spectral bias) -- a reparameterization of the weights themselves, independent of and
    complementary to any input embedding (issue #39, vs. every prior two-mode-spectral-bias fix,
    which changed the input embedding instead).

    Initialization (paper section 3.1): draw W, bias from nn.Linear's own default init, then
    factorize row-wise: s_j = exp(mu + sigma * z_j), z_j ~ N(0, 1), V = W / s (broadcast per row).
    This makes the initial effective weight diag(s) @ V equal to W exactly -- only the training
    dynamics differ from a standard nn.Linear, not the initial forward pass. mu=0.5/sigma=0.1
    (default) is the paper's own Navier-Stokes setting (Table 4, Appendix D) -- the nearest PDE
    task to this project's wave equation among the paper's own ablations (it has no
    Helmholtz/wave-equation benchmark of its own); the general-purpose default the paper states
    elsewhere is mu=1.0/sigma=0.1. Untuned for this project's specific problem either way -- see
    the issue #39 experiment write-up.
    """

    def __init__(self, in_features: int, out_features: int, mu: float = 0.5, sigma: float = 0.1):
        super().__init__()
        base = nn.Linear(in_features, out_features)
        s_init = torch.exp(mu + sigma * torch.randn(out_features, 1))
        self.s = nn.Parameter(s_init)
        self.v = nn.Parameter(base.weight.data / s_init)
        self.bias = nn.Parameter(base.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.s * self.v
        return nn.functional.linear(x, weight, self.bias)


def _rwf_mlp(dims: list[int]) -> nn.Sequential:
    modules: list[nn.Module] = []
    for i in range(len(dims) - 1):
        modules.append(RWFLinear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class RWFCavityPINN(nn.Module):
    """CavityPINN with Random Weight Factorization (see RWFLinear) applied to every linear layer of
    the MLP body -- no input embedding, so this isolates RWF's effect from Fourier features
    entirely (issue #39's "RWF alone" variant). Same MLP body shape as CavityPINN otherwise."""

    def __init__(self, hidden: int = 32, num_layers: int = 3):
        super().__init__()
        self.net = _rwf_mlp([2] + [hidden] * num_layers + [1])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


class RWFFourierCavityPINN(nn.Module):
    """FourierCavityPINN with Random Weight Factorization (see RWFLinear) applied to every linear
    layer of the MLP body -- tests RWF combined with the existing Fourier embedding (issue #39),
    since the two mechanisms (input embedding vs. weight reparameterization) are independent per
    Wang et al. Same embedding + MLP body shape as FourierCavityPINN otherwise."""

    def __init__(self, hidden: int = 32, num_layers: int = 3, num_bands: int = 4):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(num_bands=num_bands)
        in_dim = 2 * self.embedding.out_dim_per_scalar
        self.net = _rwf_mlp([in_dim] + [hidden] * num_layers + [1])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x / L, t / PERIOD)
        return self.net(embedded)


class Wavelet(nn.Module):
    """omega1*sin(x) + omega2*cos(x), learnable omega1/omega2 (Zhao et al., "PINNsFormer", ICLR
    2024, arXiv:2307.11833) — anticipates a Fourier decomposition of the target signal. The
    paper's own ablation found this necessary for the pseudo-sequence architecture below to
    converge at all: ReLU/Sigmoid fail outright, plain Sin is inconsistent across PDEs, Wavelet is
    the only activation that reliably works. Deliberately used without LayerNorm — the same
    ablation found LayerNorm didn't help and sometimes destabilized training (NaN) when paired
    with Wavelet.
    """

    def __init__(self):
        super().__init__()
        self.omega1 = nn.Parameter(torch.tensor(1.0))
        self.omega2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.omega1 * torch.sin(x) + self.omega2 * torch.cos(x)


def _pseudo_sequence(
    x: torch.Tensor, t: torch.Tensor, k: int, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand pointwise (x, t), each (N, 1), into a pseudo-sequence of k nearby timesteps:
    [x,t], [x,t+dt], ..., [x,t+(k-1)dt] -> (N, k, 1) each (PINNsFormer's Pseudo Sequence
    Generator, arXiv:2307.11833 section 3.1).
    """
    offsets = (torch.arange(k, dtype=x.dtype, device=x.device) * dt).view(1, k, 1)
    x_seq = x.unsqueeze(1).expand(-1, k, -1)
    t_seq = t.unsqueeze(1) + offsets
    return x_seq, t_seq


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), Wavelet(), nn.Linear(ff_dim, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        return x + self.ff(x)


class _DecoderLayer(nn.Module):
    """No self-attention, per PINNsFormer's design — the decoder reuses the encoder's own
    embeddings as its query and only cross-attends to the encoder's (temporally-mixed) memory."""

    def __init__(self, d_model: int, heads: int, ff_dim: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), Wavelet(), nn.Linear(ff_dim, d_model))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + attn_out
        return tgt + self.ff(tgt)


class PseudoSequenceCavityPINN(nn.Module):
    """PINNsFormer-style (Zhao et al., ICLR 2024, arXiv:2307.11833): expands pointwise (x, t) into
    a pseudo-sequence of k nearby timesteps, mixes them via a small encoder-decoder Transformer
    with Wavelet activation, then reads off the first sequence position as the pointwise field
    prediction (see `forward`). `forward_sequence` exposes the full (N, k, 1) output for the
    sequential loss in train.py, which needs it — the encoder's self-attention entangles sequence
    positions, so that loss can't just call torch.autograd.grad(u, t_seq, ...) naively (see the
    comment on `_sequence_derivative` in train.py).
    """

    def __init__(
        self,
        d_model: int = 16,
        heads: int = 2,
        ff_dim: int = 32,
        num_layers: int = 1,
        k: int = 3,
        dt: float = 1e-3,
    ):
        super().__init__()
        self.k = k
        self.dt = dt
        self.mixer = nn.Linear(2, d_model)
        self.encoder = nn.ModuleList(
            [_EncoderLayer(d_model, heads, ff_dim) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [_DecoderLayer(d_model, heads, ff_dim) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, 1)

    def forward_sequence(self, x_seq: torch.Tensor, t_seq: torch.Tensor) -> torch.Tensor:
        """x_seq, t_seq: (N, k, 1), already expanded (see _pseudo_sequence). Returns (N, k, 1)."""
        emb = self.mixer(torch.cat([x_seq, t_seq], dim=-1))
        memory = emb
        for layer in self.encoder:
            memory = layer(memory)
        out = emb
        for layer in self.decoder:
            out = layer(out, memory)
        return self.out(out)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Pointwise API matching CavityPINN/FourierCavityPINN: (N, 1), (N, 1) -> (N, 1)."""
        x_seq, t_seq = _pseudo_sequence(x, t, self.k, self.dt)
        return self.forward_sequence(x_seq, t_seq)[:, 0, :]


def _sine_basis(x: torch.Tensor, num_modes: int) -> torch.Tensor:
    """Dirichlet sine basis b_k(x) = sin(k*pi*x/L), k=1..num_modes, satisfying b_k(0)=b_k(L)=0
    by construction -- NeuSA's spectral basis for this project's PEC cavity (Bizzi et al.,
    "Neuro-Spectral Architectures for Causal Physics-Informed Networks," NeurIPS 2025,
    arXiv:2509.04966). x: (N, 1) -> (N, num_modes).
    """
    k = torch.arange(1, num_modes + 1, dtype=x.dtype, device=x.device)
    return torch.sin(x * k * math.pi / L)


def _project_onto_sine_basis(
    values_fn: Callable[[torch.Tensor], torch.Tensor], num_modes: int, n_quad: int = 2000
) -> torch.Tensor:
    """Numerically project a function of x over [0, L] onto the num_modes-term sine basis via
    trapezoidal quadrature: a_k = (2/L) * integral_0^L f(x) sin(k*pi*x/L) dx (exact sine-series
    orthogonality on [0, L]). Returns (num_modes,). Used only at construction time to set NeuSA's
    initial spectral coefficients from the true initial condition -- not recomputed during
    training/inference.
    """
    x_grid = torch.linspace(0.0, L, n_quad).unsqueeze(1)
    basis = _sine_basis(x_grid, num_modes)
    f_vals = values_fn(x_grid)
    integrand = f_vals * basis
    return torch.trapezoid(integrand, x_grid.squeeze(1), dim=0) * (2.0 / L)


def _rk4_integrate(
    model: NeuSACavityPINN, t_max: float, n_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-step RK4 integration of model.vector_field from model.state0 at t=0 to t=t_max over
    n_steps equal steps -- hand-rolled, no new dependency (torchdiffeq/torchdyn not needed for a
    small fixed-dimensional linear-ish ODE; see the experiment write-up for why this was
    preferred). Returns (grid (n_steps+1,), states (n_steps+1, 2*num_modes)), fully differentiable
    w.r.t. model.correction's parameters (standard backprop-through-the-solver, per the paper's
    own training setup).
    """
    h = t_max / n_steps
    state = model.state0
    states = [state]
    for _ in range(n_steps):
        k1 = model.vector_field(state)
        k2 = model.vector_field(state + 0.5 * h * k1)
        k3 = model.vector_field(state + 0.5 * h * k2)
        k4 = model.vector_field(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(state)
    # device=state.device (device-abstraction Arc, Slice 2, issue #59): must match `states` for
    # _interpolate_trajectory's torch.searchsorted call below, which requires both operands on
    # the same device -- extends this file's existing device=x.device idiom (_sine_basis/
    # _pseudo_sequence).
    grid = torch.linspace(0.0, t_max, n_steps + 1, device=state.device)
    return grid, torch.stack(states)


def _interpolate_trajectory(
    grid: torch.Tensor, states: torch.Tensor, t_query: torch.Tensor
) -> torch.Tensor:
    """Linear interpolation of a (n+1, dim) trajectory at arbitrary query times t_query (N,1),
    clamped to [grid[0], grid[-1]]. Needed because evaluate_relative_l2_error (and NeuSA's own
    forward, for API compatibility with CavityPINN et al.) query the field at random t, not at the
    RK4 integration grid itself.
    """
    t_flat = t_query.squeeze(-1).clamp(grid[0], grid[-1])
    idx = torch.searchsorted(grid.detach(), t_flat.detach(), right=True).clamp(1, grid.shape[0] - 1)
    t0, t1 = grid[idx - 1], grid[idx]
    weight = ((t_flat - t0) / (t1 - t0)).unsqueeze(-1)
    return states[idx - 1] + weight * (states[idx] - states[idx - 1])


class NeuSACavityPINN(nn.Module):
    """Neuro-Spectral Architecture (Bizzi et al., NeurIPS 2025, arXiv:2509.04966) for this
    project's 1D PEC cavity -- issue #36's test of a genuinely different architecture (not another
    loss/schedule modification layered on the coordinate-input MLP) against the long-horizon
    collapse (issue #23). See _rk4_integrate/_sine_basis docstrings and the experiment write-up
    (experiments/long-horizon-collapse/036-neusa-long-horizon.md) for the full derivation.
    """

    def __init__(
        self,
        num_modes: int = 8,
        hidden: int = 16,
        eps: float = 0.1,
        steps_per_unit_time: float = 20.0,
        field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = analytical_field,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.eps = eps
        self.steps_per_unit_time = steps_per_unit_time
        k = torch.arange(1, num_modes + 1, dtype=torch.float32)
        self.register_buffer("omega_sq", (k * math.pi * C / L) ** 2)
        self.correction = _mlp([2 * num_modes, hidden, hidden, num_modes])

        a0 = _project_onto_sine_basis(lambda x: field_fn(x, torch.zeros_like(x)), num_modes)

        def _field_dt0(x: torch.Tensor) -> torch.Tensor:
            t0 = torch.zeros_like(x, requires_grad=True)
            e = field_fn(x, t0)
            (e_t,) = torch.autograd.grad(e, t0, grad_outputs=torch.ones_like(e))
            return e_t.detach()

        w0 = _project_onto_sine_basis(_field_dt0, num_modes)
        self.register_buffer("state0", torch.cat([a0, w0]))

    def vector_field(self, state: torch.Tensor) -> torch.Tensor:
        a, w = state[: self.num_modes], state[self.num_modes :]
        da_dt = w
        dw_dt = -self.omega_sq * a + self.eps * self.correction(state)
        return torch.cat([da_dt, dw_dt])

    def n_steps_for(self, t_max: float) -> int:
        return max(round(t_max * self.steps_per_unit_time), 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_max = max(float(t.detach().max()), 1e-6)
        grid, states = _rk4_integrate(self, t_max, self.n_steps_for(t_max))
        coeffs = _interpolate_trajectory(grid, states, t)
        a = coeffs[:, : self.num_modes]
        basis = _sine_basis(x, self.num_modes)
        return (a * basis).sum(dim=-1, keepdim=True)

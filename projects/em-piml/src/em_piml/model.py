from __future__ import annotations

import torch
from torch import nn

from em_piml.embeddings import FourierFeatureEmbedding
from em_piml.physics import PERIOD, L


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

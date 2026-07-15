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

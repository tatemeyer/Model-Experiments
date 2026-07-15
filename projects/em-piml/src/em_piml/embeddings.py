from __future__ import annotations

import torch
from torch import nn


class FourierFeatureEmbedding(nn.Module):
    """NeRF/Tancik-style positional encoding for a scalar coordinate normalized to [0, 1]:
    [u, sin(2^0 pi u), cos(2^0 pi u), ..., sin(2^(k-1) pi u), cos(2^(k-1) pi u)].
    Applied independently to x and t, then concatenated.
    """

    def __init__(self, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_bands) * torch.pi)

    @property
    def out_dim_per_scalar(self) -> int:
        return 1 + 2 * self.num_bands

    def _embed_scalar(self, u: torch.Tensor) -> torch.Tensor:
        angles = u * self.frequencies
        return torch.cat([u, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._embed_scalar(x_norm), self._embed_scalar(t_norm)], dim=-1)

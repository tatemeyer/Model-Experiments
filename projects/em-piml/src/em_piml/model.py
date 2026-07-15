from __future__ import annotations

import torch
from torch import nn


class CavityPINN(nn.Module):
    """Coordinate-input MLP mapping (x, t) -> predicted E_z. Deliberately plain — the baseline."""

    def __init__(self, hidden: int = 64, num_layers: int = 4):
        super().__init__()
        dims = [2] + [hidden] * num_layers + [1]
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))

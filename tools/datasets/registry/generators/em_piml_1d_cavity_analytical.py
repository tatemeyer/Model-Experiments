#!/usr/bin/env python3
"""Analytical E(x,t) grid for the 1D PEC cavity mode — ground truth for em-piml's baseline PINN.

Reuses em_piml.physics.analytical_field (installed via the shared uv workspace venv) rather
than duplicating the closed-form solution here.
"""

from pathlib import Path

import numpy as np
import torch
from em_piml.physics import AMPLITUDE, N_MODE, OMEGA, PERIOD, C, L, analytical_field

N_PERIODS = 2

dest = Path(".data/em-piml-1d-cavity-analytical")
dest.mkdir(parents=True, exist_ok=True)

x = np.linspace(0.0, L, 200)
t = np.linspace(0.0, N_PERIODS * PERIOD, 200)
grid_x, grid_t = np.meshgrid(x, t, indexing="ij")
field = analytical_field(torch.from_numpy(grid_x), torch.from_numpy(grid_t)).numpy()

np.savez(
    dest / "cavity_mode.npz",
    x=x,
    t=t,
    grid_x=grid_x,
    grid_t=grid_t,
    field=field,
    L=L,
    c=C,
    n=N_MODE,
    amplitude=AMPLITUDE,
    omega=OMEGA,
)

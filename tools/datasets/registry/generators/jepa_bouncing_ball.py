#!/usr/bin/env python3
"""Seeded bouncing-ball video dataset -- ground truth for projects/jepa's toy JEPA environment.

Reuses jepa.bouncing_ball (installed via the shared uv workspace venv) rather than duplicating
the closed-form physics here, mirroring em_piml_1d_cavity_analytical.py's pattern.
"""

from pathlib import Path

import numpy as np
from jepa.bouncing_ball import generate_dataset

dest = Path(".data/jepa-bouncing-ball")
dest.mkdir(parents=True, exist_ok=True)

dataset = generate_dataset()

np.savez(
    dest / "bouncing_ball.npz",
    frames=dataset["frames"],
    positions=dataset["positions"],
    velocities=dataset["velocities"],
    initial_states=dataset["initial_states"],
    canvas_size=dataset["canvas_size"],
    radius=dataset["radius"],
    speed=dataset["speed"],
    master_seed=dataset["master_seed"],
)

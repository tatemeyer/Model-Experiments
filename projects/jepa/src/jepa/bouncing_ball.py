"""Closed-form bouncing-ball physics: a point mass moving at constant speed inside a square
canvas, reflecting elastically off all four walls, no gravity/friction. Ground truth for
`projects/jepa`'s toy JEPA environment -- see CLAUDE.md's "Toy environment" section. Position and
velocity have an exact closed form (a folded/triangle-wave function of time), the same
exact-solution standard `projects/em-piml` holds its analytical field to -- no numerical
integration needed to know where the ball is at any time t.
"""

from __future__ import annotations

import numpy as np

CANVAS_SIZE = 32
RADIUS = 3.0
SPEED = 1.6  # pixels per frame
N_FRAMES = 30
N_SEQUENCES = 200
RENDER_SIGMA = RADIUS / 1.5

# Ball center ranges over [RADIUS, CANVAS_SIZE - RADIUS] in each axis; shifting by RADIUS gives
# the [0, BOUND] local coordinate _fold operates on.
BOUND = CANVAS_SIZE - 2 * RADIUS


def _fold(u: np.ndarray, bound: float) -> np.ndarray:
    """Triangle-wave fold of u into [0, bound]: the closed-form position of a 1D point bouncing
    elastically between 0 and bound, starting at u(0) with no reflection yet applied. mod-based,
    not iterative -- valid for any t, however many reflections have occurred by then."""
    period = 2 * bound
    m = np.mod(u, period)
    return np.where(m <= bound, m, period - m)


def _fold_sign(u: np.ndarray, bound: float) -> np.ndarray:
    """d(_fold(u, bound))/du -- +1 on the rising leg, -1 on the reflected (falling) leg. The
    instantaneous velocity along this axis is the original constant velocity times this sign."""
    period = 2 * bound
    m = np.mod(u, period)
    return np.where(m <= bound, 1.0, -1.0)


def sample_initial_state(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Uniform-random ball center within bounds, uniform-random heading at fixed SPEED."""
    x0 = rng.uniform(RADIUS, CANVAS_SIZE - RADIUS)
    y0 = rng.uniform(RADIUS, CANVAS_SIZE - RADIUS)
    theta = rng.uniform(0.0, 2 * np.pi)
    vx0 = SPEED * np.cos(theta)
    vy0 = SPEED * np.sin(theta)
    return float(x0), float(y0), float(vx0), float(vy0)


def position_at(
    x0: float, y0: float, vx0: float, vy0: float, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ball center (x(t), y(t)) for scalar or array t, in canvas pixel coordinates."""
    x = RADIUS + _fold(np.asarray(t) * vx0 + (x0 - RADIUS), BOUND)
    y = RADIUS + _fold(np.asarray(t) * vy0 + (y0 - RADIUS), BOUND)
    return x, y


def velocity_at(
    x0: float, y0: float, vx0: float, vy0: float, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form instantaneous velocity (vx(t), vy(t)) -- vx0/vy0 scaled by the fold's sign."""
    t = np.asarray(t)
    vx = vx0 * _fold_sign(t * vx0 + (x0 - RADIUS), BOUND)
    vy = vy0 * _fold_sign(t * vy0 + (y0 - RADIUS), BOUND)
    return vx, vy


def render_frame(
    x: float, y: float, canvas_size: int = CANVAS_SIZE, sigma: float = RENDER_SIGMA
) -> np.ndarray:
    """Grayscale (canvas_size, canvas_size) uint8 frame: a soft Gaussian blob at (x, y). Purely a
    deterministic function of (x, y) and the fixed grid -- rendering itself draws no randomness,
    only the initial state (sample_initial_state) does."""
    grid = np.arange(canvas_size, dtype=np.float64) + 0.5  # pixel centers
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    dist_sq = (gx - x) ** 2 + (gy - y) ** 2
    intensity = np.exp(-dist_sq / (2 * sigma**2))
    return np.clip(intensity * 255.0, 0, 255).astype(np.uint8)


def generate_sequence(
    seed: np.random.SeedSequence | int, n_frames: int = N_FRAMES
) -> dict[str, np.ndarray]:
    """One deterministic (frames, positions, velocities) sequence from a single seed."""
    rng = np.random.default_rng(seed)
    x0, y0, vx0, vy0 = sample_initial_state(rng)
    t = np.arange(n_frames, dtype=np.float64)
    x, y = position_at(x0, y0, vx0, vy0, t)
    vx, vy = velocity_at(x0, y0, vx0, vy0, t)
    frames = np.stack([render_frame(x[i], y[i]) for i in range(n_frames)])
    return {
        "frames": frames,
        "positions": np.stack([x, y], axis=-1),
        "velocities": np.stack([vx, vy], axis=-1),
        "initial_state": np.array([x0, y0, vx0, vy0]),
    }


def generate_dataset(
    n_sequences: int = N_SEQUENCES, n_frames: int = N_FRAMES, master_seed: int = 0
) -> dict[str, np.ndarray]:
    """Full dataset: n_sequences independent sequences, each n_frames long. Per-sequence seeds are
    spawned from one SeedSequence (not master_seed + i) so substreams are statistically
    independent, not just differently offset -- standard numpy.random practice for reproducible
    parallel/batched sampling."""
    seeds = np.random.SeedSequence(master_seed).spawn(n_sequences)
    sequences = [generate_sequence(seed, n_frames=n_frames) for seed in seeds]
    return {
        "frames": np.stack([s["frames"] for s in sequences]),
        "positions": np.stack([s["positions"] for s in sequences]),
        "velocities": np.stack([s["velocities"] for s in sequences]),
        "initial_states": np.stack([s["initial_state"] for s in sequences]),
        "canvas_size": CANVAS_SIZE,
        "radius": RADIUS,
        "speed": SPEED,
        "master_seed": master_seed,
    }

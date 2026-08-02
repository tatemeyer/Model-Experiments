from __future__ import annotations

import numpy as np
from jepa.bouncing_ball import (
    BOUND,
    CANVAS_SIZE,
    RADIUS,
    generate_dataset,
    generate_sequence,
    position_at,
    velocity_at,
)


def _bruteforce_position(
    x0: float, y0: float, vx0: float, vy0: float, t_target: float, dt: float = 1e-4
) -> tuple[float, float]:
    """Independent ground truth: small-step Euler integration with explicit elastic-collision
    handling at the walls, as opposed to position_at's closed-form modulo/fold formula. Used only
    to verify the closed form is physically correct, not as the dataset's own generation path."""
    x, y, vx, vy = x0, y0, vx0, vy0
    steps = int(round(t_target / dt))
    lo, hi = RADIUS, RADIUS + BOUND
    for _ in range(steps):
        x, y = x + vx * dt, y + vy * dt
        if x < lo:
            x, vx = 2 * lo - x, -vx
        elif x > hi:
            x, vx = 2 * hi - x, -vx
        if y < lo:
            y, vy = 2 * lo - y, -vy
        elif y > hi:
            y, vy = 2 * hi - y, -vy
    return x, y


def test_position_matches_independent_bruteforce_simulation():
    # A heading and sample times chosen to cross several wall reflections within the checked
    # window, so the fold formula's multi-reflection behavior is actually exercised, not just its
    # unreflected first leg.
    x0, y0, vx0, vy0 = 5.0, 5.0, 1.3, -0.9
    for t_target in [1.0, 5.0, 12.5, 20.0]:
        x_closed, y_closed = position_at(x0, y0, vx0, vy0, np.array(t_target))
        x_bf, y_bf = _bruteforce_position(x0, y0, vx0, vy0, t_target)
        assert abs(float(x_closed) - x_bf) < 1e-2
        assert abs(float(y_closed) - y_bf) < 1e-2


def test_velocity_matches_finite_difference_of_position():
    x0, y0, vx0, vy0 = 5.0, 5.0, 1.3, -0.9
    for t_target in [1.0, 5.0, 12.5, 20.0]:
        eps = 1e-5
        x_plus, y_plus = position_at(x0, y0, vx0, vy0, np.array(t_target + eps))
        x_minus, y_minus = position_at(x0, y0, vx0, vy0, np.array(t_target - eps))
        vx_fd = (float(x_plus) - float(x_minus)) / (2 * eps)
        vy_fd = (float(y_plus) - float(y_minus)) / (2 * eps)
        vx, vy = velocity_at(x0, y0, vx0, vy0, np.array(t_target))
        assert abs(float(vx) - vx_fd) < 1e-3
        assert abs(float(vy) - vy_fd) < 1e-3


def test_position_stays_within_canvas_bounds_over_many_reflections():
    x0, y0, vx0, vy0 = 5.0, 5.0, 1.3, -0.9
    t = np.arange(0, 200, dtype=np.float64)
    x, y = position_at(x0, y0, vx0, vy0, t)
    assert np.all(x >= RADIUS - 1e-9) and np.all(x <= CANVAS_SIZE - RADIUS + 1e-9)
    assert np.all(y >= RADIUS - 1e-9) and np.all(y <= CANVAS_SIZE - RADIUS + 1e-9)


def test_same_seed_produces_bit_identical_sequence():
    a = generate_sequence(seed=42, n_frames=10)
    b = generate_sequence(seed=42, n_frames=10)
    assert np.array_equal(a["frames"], b["frames"])
    assert np.array_equal(a["positions"], b["positions"])
    assert np.array_equal(a["velocities"], b["velocities"])
    assert np.array_equal(a["initial_state"], b["initial_state"])


def test_different_seeds_produce_different_sequences():
    a = generate_sequence(seed=1, n_frames=10)
    b = generate_sequence(seed=2, n_frames=10)
    assert not np.array_equal(a["initial_state"], b["initial_state"])


def test_stored_positions_recompute_exactly_from_initial_state():
    # The dataset's stored per-frame positions/velocities must match position_at/velocity_at
    # applied to the stored initial_state exactly -- i.e. generate_sequence doesn't do anything
    # beyond what the closed-form functions themselves compute.
    seq = generate_sequence(seed=7, n_frames=15)
    x0, y0, vx0, vy0 = seq["initial_state"]
    t = np.arange(15, dtype=np.float64)
    x, y = position_at(x0, y0, vx0, vy0, t)
    vx, vy = velocity_at(x0, y0, vx0, vy0, t)
    assert np.array_equal(seq["positions"], np.stack([x, y], axis=-1))
    assert np.array_equal(seq["velocities"], np.stack([vx, vy], axis=-1))


def test_generate_dataset_same_master_seed_bit_identical():
    a = generate_dataset(n_sequences=5, n_frames=6, master_seed=0)
    b = generate_dataset(n_sequences=5, n_frames=6, master_seed=0)
    assert np.array_equal(a["frames"], b["frames"])
    assert np.array_equal(a["positions"], b["positions"])
    assert np.array_equal(a["velocities"], b["velocities"])


def test_generate_dataset_sequences_are_independent_not_offset_copies():
    dataset = generate_dataset(n_sequences=5, n_frames=6, master_seed=0)
    initial_states = dataset["initial_states"]
    # No two sequences should share an initial state (would indicate a broken seed-spawn).
    unique_rows = {tuple(row) for row in initial_states}
    assert len(unique_rows) == len(initial_states)


def test_frame_shape_and_dtype():
    seq = generate_sequence(seed=0, n_frames=4)
    assert seq["frames"].shape == (4, CANVAS_SIZE, CANVAS_SIZE)
    assert seq["frames"].dtype == np.uint8

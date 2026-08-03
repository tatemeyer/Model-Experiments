from __future__ import annotations

import pytest
from jepa.train import train_jepa

# Per projects/em-piml/CLAUDE.md's standing convention (mirrored here per issue #67): mark a test
# @pytest.mark.slow if it actually trains/fits a model, so the default `uv run pytest` stays fast.
# These train_jepa calls are brief (20-150 steps) but still real training loops.


@pytest.mark.slow
def test_same_seed_produces_bit_identical_loss_curve():
    history_a: list[float] = []
    train_jepa(steps=20, seed=0, history=history_a)
    history_b: list[float] = []
    train_jepa(steps=20, seed=0, history=history_b)
    assert history_a == history_b


@pytest.mark.slow
def test_different_seeds_produce_different_loss_curves():
    history_a: list[float] = []
    train_jepa(steps=20, seed=0, history=history_a)
    history_b: list[float] = []
    train_jepa(steps=20, seed=1, history=history_b)
    assert history_a != history_b


@pytest.mark.slow
def test_loss_decreases_on_average_across_training():
    # Compares the mean of the first vs. last few steps rather than history[0] vs. history[-1]
    # directly -- single-step loss is noisy (fresh minibatch + fresh mask each step), so a
    # first-vs-last-window average is the robust way to check a real downward trend without
    # flaking on ordinary step-to-step variance.
    history: list[float] = []
    train_jepa(steps=150, seed=0, history=history)
    window = 10
    early_mean = sum(history[:window]) / window
    late_mean = sum(history[-window:]) / window
    assert late_mean < early_mean, (
        f"expected loss to decrease on average: early mean {early_mean:.4f}, "
        f"late mean {late_mean:.4f}"
    )

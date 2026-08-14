from __future__ import annotations

import pytest
import torch
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


@pytest.mark.slow
def test_checkpoint_hook_does_not_perturb_training():
    """The sweep in jepa.momentum_steps_sweep evaluates mid-training. That is only sound if the
    callback cannot change the trajectory. train_jepa seeds torch globally once before model
    construction, then draws batches from an explicit torch.Generator and masks from a numpy
    Generator -- never the global RNG -- so a callback that scribbles all over the global RNG
    should still leave training bit-identical. This asserts exactly that, with a deliberately
    hostile callback."""
    seen: list[int] = []

    def hostile(step: int, encoder) -> None:
        seen.append(step)
        torch.manual_seed(999)  # deliberately corrupt the global RNG
        torch.rand(64)

    hooked: list[float] = []
    train_jepa(
        steps=60,
        seed=0,
        history=hooked,
        checkpoint_steps=(20, 40, 60),
        on_checkpoint=hostile,
    )

    plain: list[float] = []
    train_jepa(steps=60, seed=0, history=plain)

    assert seen == [20, 40, 60]
    assert hooked == plain

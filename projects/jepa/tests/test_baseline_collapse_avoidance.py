"""Locks in Arc 1 Slice 1's baseline result (Task E, issue #69). See
../experiments/001-baseline-collapse-avoidance.md for the full write-up, the pooling-strategy
exploration that led to this specific harness, and how these thresholds were picked.

Two independent findings, not one:
1. Collapse (confirms the issue's hypothesis, with a correction): the full EMA model's
   *effective_rank* stays reliably above the no-EMA ablation's (target encoder trained by
   gradient, no stop-gradient) across seeds -- but only once training runs long enough (this
   project's STEPS=300 default is too short for the gap to appear) and only for effective_rank,
   not embedding_std (which does not reliably separate the two -- see the write-up).
2. Linear-probe R^2 (does *not* confirm the issue's hypothesis): the full model's held-out
   position/velocity probe R^2 is not reliably higher than a random-init-encoder baseline's, at
   this toy architecture's scale -- documented as a negative result, not asserted as a threshold
   crossing that doesn't actually happen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from jepa.bouncing_ball import CANVAS_SIZE, generate_dataset
from jepa.eval import effective_rank, embedding_std, linear_probe_r2
from jepa.models import PatchEncoder
from jepa.train import EMBED_DIM, HIDDEN_DIM, PATCH_SIZE, train_jepa

# train.py's own STEPS default (300) is calibrated for test_train.py's fast collapse-agnostic
# sanity checks, not for this experiment's collapse comparison -- at 300 steps neither model has
# trained long enough for the EMA-vs-no-EMA effective_rank gap to appear (see write-up). 3000 is
# the shortest length at which the gap was reliably observed across seeds 0/1/2.
COLLAPSE_STEPS = 3000

# Held-out probe splits use master seeds offset well clear of any training-pool seed used in this
# test/experiment, so a probe frame is never one the encoder was trained on.
PROBE_TRAIN_SEED_OFFSET = 10_000
PROBE_TEST_SEED_OFFSET = 20_000

# effective_rank threshold separating full from the no-EMA ablation -- picked with margin inside
# the observed seed 0/1/2 ranges (full: 2.35-2.79, no_ema: 1.25-1.46) at COLLAPSE_STEPS; not a
# universal "healthy embedding" constant. embedding_std is deliberately not thresholded here --
# it does not reliably separate the two variants (see write-up's mechanistic diagnosis).
COLLAPSE_RANK_THRESHOLD = 1.8

# Regression floor for the probe-R^2 negative result -- observed seed 0/1/2 range for both full
# and random_init is 0.10-0.98 (see write-up's results table); this just guards against a future
# change silently collapsing the online encoder's probe-recoverable signal to ~chance (R^2 <= 0),
# not an accuracy bar (there isn't a "full beats random_init" bar to clear here -- see below).
PROBE_R2_FLOOR = 0.05


def _probe_frames_and_targets(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = generate_dataset(n_sequences=n, n_frames=1, master_seed=seed)
    frames = torch.from_numpy(dataset["frames"][:, 0]).float().unsqueeze(1) / 255.0
    positions = dataset["positions"][:, 0]
    velocities = dataset["velocities"][:, 0]
    targets = torch.from_numpy(np.concatenate([positions, velocities], axis=-1)).float()
    return frames, targets


def _per_patch_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Every (frame, patch) pair as its own sample, (N * num_patches, embed_dim) -- the direct
    "do patch representations vary meaningfully" collapse question, not diluted by averaging
    content-bearing patches against the ~55/64 pure-background patches this task's frames have
    (mean-pooling was tried first and found to wash out the signal almost entirely -- see
    write-up)."""
    with torch.no_grad():
        tokens = encoder(frames)
    return tokens.reshape(-1, tokens.shape[-1])


def _flattened_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Every patch embedding concatenated per frame, (N, num_patches * embed_dim) -- preserves
    spatial position (unlike mean-pooling) for the linear probe. Needs a probe train set well
    above this dimensionality (num_patches * embed_dim = 2048 here) to avoid the OLS
    overdetermined-regime overfitting this task's D>>N small-sample regime showed early on (see
    write-up)."""
    with torch.no_grad():
        tokens = encoder(frames)
    return tokens.reshape(tokens.shape[0], -1)


def collapse_metrics(encoder: PatchEncoder, seed: int, n_test: int = 300) -> dict[str, float]:
    frames, _ = _probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    embeddings = _per_patch_embeddings(encoder, frames)
    return {
        "embedding_std": embedding_std(embeddings),
        "effective_rank": effective_rank(embeddings),
    }


def probe_r2(
    encoder: PatchEncoder, seed: int, n_train: int = 4000, n_test: int = 300
) -> float:
    train_frames, train_targets = _probe_frames_and_targets(n_train, seed + PROBE_TRAIN_SEED_OFFSET)
    test_frames, test_targets = _probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    train_embeddings = _flattened_embeddings(encoder, train_frames)
    test_embeddings = _flattened_embeddings(encoder, test_frames)
    return linear_probe_r2(train_embeddings, train_targets, test_embeddings, test_targets)


def build_encoder(variant: str, seed: int, steps: int = COLLAPSE_STEPS) -> PatchEncoder:
    """variant: "full" (EMA target), "no_ema" (gradient-trained target, issue #69's ablation), or
    "random_init" (untrained encoder, no train_jepa call at all)."""
    if variant == "full":
        encoder, _, _ = train_jepa(steps=steps, seed=seed, use_ema=True)
    elif variant == "no_ema":
        encoder, _, _ = train_jepa(steps=steps, seed=seed, use_ema=False)
    elif variant == "random_init":
        torch.manual_seed(seed)
        encoder = PatchEncoder(
            image_size=CANVAS_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return encoder


@pytest.mark.slow
def test_full_model_training_is_deterministic_given_a_seed():
    # Determinism must hold before any seed-to-seed comparison is trustworthy (projects/em-piml/
    # CLAUDE.md issue #19's standing rule, mirrored in this project's train.py:56-58).
    history_a: list[float] = []
    train_jepa(steps=30, seed=0, history=history_a)
    history_b: list[float] = []
    train_jepa(steps=30, seed=0, history=history_b)
    assert history_a == history_b


@pytest.mark.slow
def test_no_ema_ablation_training_is_deterministic_given_a_seed():
    # The no-EMA ablation is a new training path (issue #69) -- its own determinism isn't covered
    # by the EMA-path check above (different optimizer param list, different target-encoder
    # branch), so it gets its own check before its numbers are trusted.
    history_a: list[float] = []
    train_jepa(steps=30, seed=0, use_ema=False, history=history_a)
    history_b: list[float] = []
    train_jepa(steps=30, seed=0, use_ema=False, history=history_b)
    assert history_a == history_b


@pytest.mark.slow
def test_full_model_avoids_collapse_and_does_not_reliably_beat_random_init_probe():
    """Two independent findings per seed, from encoders trained once each (not retrained per
    assertion): (1) the full EMA model's effective_rank reliably beats the no-EMA ablation's --
    confirms the issue's collapse-avoidance hypothesis, with effective_rank (not embedding_std) as
    the metric that actually separates them. (2) the full model's held-out probe R^2 does NOT
    reliably beat a random-init encoder's -- a negative result relative to the issue's second
    hypothesis. See ../experiments/001-baseline-collapse-avoidance.md for the full write-up and
    the observed per-seed numbers this locks in."""
    for seed in (0, 1, 2):
        full_encoder = build_encoder("full", seed)
        no_ema_encoder = build_encoder("no_ema", seed)
        random_encoder = build_encoder("random_init", seed)

        full_collapse = collapse_metrics(full_encoder, seed)
        no_ema_collapse = collapse_metrics(no_ema_encoder, seed)
        assert full_collapse["effective_rank"] > COLLAPSE_RANK_THRESHOLD, (
            f"seed {seed}: full model effective_rank {full_collapse['effective_rank']:.3f} did "
            f"not clear {COLLAPSE_RANK_THRESHOLD}"
        )
        assert no_ema_collapse["effective_rank"] < COLLAPSE_RANK_THRESHOLD, (
            f"seed {seed}: no-EMA ablation effective_rank {no_ema_collapse['effective_rank']:.3f} "
            f"did not fall below {COLLAPSE_RANK_THRESHOLD}"
        )

        full_r2 = probe_r2(full_encoder, seed)
        random_r2 = probe_r2(random_encoder, seed)
        # Not asserting full_r2 > random_r2 -- that's the issue's hypothesis, and it does not
        # hold reproducibly (see write-up). Both just need to stay above chance-level R^2.
        assert full_r2 > PROBE_R2_FLOOR, f"seed {seed}: full model probe_r2 {full_r2:.4f} too low"
        assert random_r2 > PROBE_R2_FLOOR, (
            f"seed {seed}: random_init probe_r2 {random_r2:.4f} too low"
        )

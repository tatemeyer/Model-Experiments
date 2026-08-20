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
   position/velocity probe R^2 is not higher than a random-init-encoder baseline's, at this toy
   architecture's scale -- documented as a negative result, not asserted as a threshold crossing
   that doesn't actually happen.

   **Amended by issue #104.** The per-seed probe numbers this originally locked in (0.10-0.98,
   scattered) were produced by a broken least-squares solver and are not real. The negative
   result survives the correction, but its shape changed completely: all three variants score
   0.9763-0.9773, indistinguishable, because the probe is saturated rather than noisy. See
   ../experiments/003-probe-solver-correctness.md.
"""

from __future__ import annotations

import pytest
from jepa.harness import build_encoder, collapse_metrics, probe_r2
from jepa.train import train_jepa

# train.py's own STEPS default (300) is calibrated for test_train.py's fast collapse-agnostic
# sanity checks, not for this experiment's collapse comparison -- at 300 steps neither model has
# trained long enough for the EMA-vs-no-EMA effective_rank gap to appear (see write-up). 3000 is
# the shortest length at which the gap was reliably observed across seeds 0/1/2.
COLLAPSE_STEPS = 3000

# effective_rank threshold separating full from the no-EMA ablation -- picked with margin inside
# the observed seed 0/1/2 ranges (full: 2.35-2.79, no_ema: 1.25-1.46) at COLLAPSE_STEPS; not a
# universal "healthy embedding" constant. embedding_std is deliberately not thresholded here --
# it does not reliably separate the two variants (see write-up's mechanistic diagnosis).
COLLAPSE_RANK_THRESHOLD = 1.8

# Regression floor for the probe-R^2 result. Was 0.05 against an observed 0.10-0.98 range; issue
# #104 showed that range was an artifact of a broken least-squares solver, and under the corrected
# probe every variant lands in 0.9763-0.9773 (see experiments/003-probe-solver-correctness.md).
# 0.95 is therefore a real regression guard rather than a near-vacuous one -- it would have caught
# the #104 defect on its own, which 0.05 did not.
PROBE_R2_FLOOR = 0.95


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
        full_encoder = build_encoder("full", seed, steps=COLLAPSE_STEPS)
        no_ema_encoder = build_encoder("no_ema", seed, steps=COLLAPSE_STEPS)
        random_encoder = build_encoder("random_init", seed, steps=COLLAPSE_STEPS)

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
        # Still not asserting full_r2 > random_r2, but for a sharper reason than the original
        # write-up had: under the corrected probe (issue #104) the two are not merely unordered,
        # they are equal to ~1e-3 because the probe is *saturated* -- an untrained encoder already
        # scores 0.9767. There is no headroom for training to win, so this stays a floor check.
        assert full_r2 > PROBE_R2_FLOOR, f"seed {seed}: full model probe_r2 {full_r2:.4f} too low"
        assert random_r2 > PROBE_R2_FLOOR, (
            f"seed {seed}: random_init probe_r2 {random_r2:.4f} too low"
        )

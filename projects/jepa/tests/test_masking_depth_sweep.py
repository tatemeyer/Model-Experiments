"""Arc 1 Slice 3 (issue #107): masking ratio and predictor depth.

Two things are locked in here. First, that Slice 3's new `train_jepa` parameters are a genuine
no-op at their defaults -- the axes had to become parameters before they could be swept, and a
parameterization that quietly changed the default trajectory would invalidate every number Slices
1 and 2 recorded. Second, the slice's actual finding (see the sweep summary in
../experiments/004-masking-ratio-predictor-depth.md).
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest
import torch
from jepa.harness import collapse_metrics
from jepa.masking import BlockMaskGenerator
from jepa.masking_depth_sweep import MASK_CONFIGS, realized_target_fraction
from jepa.train import (
    NUM_TARGET_BLOCKS,
    PREDICTOR_DEPTH,
    TARGET_SCALE_RANGE,
    train_jepa,
)


def test_slice3_constants_match_the_values_prior_slices_ran_at():
    """The defaults are load-bearing: Slices 1 and 2's recorded numbers were produced at these
    exact values, so changing a constant silently reinterprets every historical row. Pinned
    against `models.Predictor` / `BlockMaskGenerator`'s own defaults rather than repeated
    literals, so the two definitions cannot drift apart unnoticed."""
    generator = BlockMaskGenerator(grid_h=8, grid_w=8)
    assert PREDICTOR_DEPTH == 2
    assert NUM_TARGET_BLOCKS == generator.num_target_blocks
    assert TARGET_SCALE_RANGE == generator.target_scale_range


@pytest.mark.slow
def test_slice3_parameters_are_a_no_op_at_their_defaults():
    """Passing the defaults explicitly must produce a bit-identical trajectory to not passing
    them at all. This is the assertion that makes Slice 3's sweep comparable to Slices 1 and 2:
    the `default` masking cell and `depth=2` row are supposed to *be* the configuration those
    slices ran, not merely resemble it.

    Bit-identical loss history, not just a close final metric -- a divergence anywhere in
    training is what would matter, and it would be invisible in a single end-of-run number."""
    baseline: list[float] = []
    train_jepa(steps=30, seed=0, history=baseline)

    explicit: list[float] = []
    train_jepa(
        steps=30,
        seed=0,
        history=explicit,
        predictor_depth=PREDICTOR_DEPTH,
        num_target_blocks=NUM_TARGET_BLOCKS,
        target_scale_range=TARGET_SCALE_RANGE,
    )
    assert baseline == explicit


# Mean within-cell seed spread measured across the full 27-run grid (see the write-up's variance
# table). It is the yardstick the slice's finding is stated against: both axes moved
# effective_rank by *less* than seeds did. Asserting against a noise scale rather than an absolute
# rank also makes these robust to the torch-version drift the write-up records (~0.06 here, an
# order of magnitude smaller).
SEED_SPREAD = 0.75

# Long enough to be past the regime where every variant is indistinguishable, short enough to keep
# four trainings inside a slow test. The full sweep ran 3000; this reproduces the *direction* of
# the finding, not the surface -- same reduced-budget approach as Slice 2's regression test.
NULL_STEPS = 1000


@pytest.mark.slow
def test_predictor_depth_moves_rank_less_than_seed_noise_does():
    """Slice 3's headline null, stated as something a test can actually fail.

    "No effect" has no threshold, so this asserts the comparative claim the write-up makes
    instead: sweeping predictor depth across its full swept range (1 -> 4) at a fixed seed changes
    `effective_rank` by less than the seed-to-seed spread within a single configuration. If a
    future change made depth genuinely matter, this is what would break.

    Measured at NULL_STEPS/seed 0: depth 1 -> 1.630, depth 4 -> 1.845, a gap of 0.215 against a
    seed spread of 0.75."""
    shallow, _, _ = train_jepa(steps=NULL_STEPS, seed=0, predictor_depth=1)
    deep, _, _ = train_jepa(steps=NULL_STEPS, seed=0, predictor_depth=4)

    gap = abs(
        collapse_metrics(shallow, seed=0)["effective_rank"]
        - collapse_metrics(deep, seed=0)["effective_rank"]
    )
    assert gap < SEED_SPREAD, (
        f"predictor depth 1 vs 4 moved effective_rank by {gap:.3f}, at or above the {SEED_SPREAD} "
        f"seed-noise scale -- Slice 3 concluded this axis is inert, so a real effect here means "
        f"experiments/004-masking-ratio-predictor-depth.md needs revisiting"
    )


@pytest.mark.slow
def test_masking_ratio_moves_rank_less_than_seed_noise_does():
    """The masking half of the same null. Realized masked fraction spans 0.258 -> 0.702 between
    these two configurations -- a wide span, not a marginal one -- and still moves the metric less
    than changing the seed does (measured 0.256 at NULL_STEPS/seed 0).

    Note the write-up's *other* masking finding, that heavy masking leaves the loss still rising,
    is deliberately not asserted: it is clean at 3000 steps (9/9 runs) but its margin at this
    test's reduced budget is ~0.007, too thin to encode without a flaky test."""
    light, _, _ = train_jepa(steps=NULL_STEPS, seed=0, target_scale_range=MASK_CONFIGS["light"])
    heavy, _, _ = train_jepa(steps=NULL_STEPS, seed=0, target_scale_range=MASK_CONFIGS["heavy"])

    gap = abs(
        collapse_metrics(light, seed=0)["effective_rank"]
        - collapse_metrics(heavy, seed=0)["effective_rank"]
    )
    assert gap < SEED_SPREAD, (
        f"masking ratio 0.258 vs 0.702 moved effective_rank by {gap:.3f}, at or above the "
        f"{SEED_SPREAD} seed-noise scale -- Slice 3 concluded this axis is inert for collapse"
    )


@pytest.mark.slow
def test_predictor_depth_changes_the_trajectory():
    """Guards the plumbing rather than the science: if `predictor_depth` were dropped on the way
    to `Predictor`, every depth cell in the sweep would silently be depth 2 and the whole axis
    would read as 'no effect' for the wrong reason. A different depth must at minimum train
    differently."""
    shallow: list[float] = []
    train_jepa(steps=30, seed=0, history=shallow, predictor_depth=1)
    deep: list[float] = []
    train_jepa(steps=30, seed=0, history=deep, predictor_depth=4)
    assert shallow != deep


def test_masking_ratio_knob_changes_the_realized_masked_fraction():
    """The companion plumbing guard for the masking axis, and the reason the sweep reports a
    *realized* fraction: the knob is a per-block scale range, and four blocks are unioned, so the
    quantity that actually varies has to be measured rather than read off the knob."""
    fractions = {name: realized_target_fraction(rng) for name, rng in MASK_CONFIGS.items()}
    assert fractions["light"] < fractions["default"] < fractions["heavy"]
    # Union of overlapping blocks is sublinear in per-block scale: 4 blocks at 30-40% of the grid
    # each would be >100% if they never overlapped, so the realized heavy fraction must be well
    # under 1.0. This is the specific arithmetic that makes the nominal knob misleading.
    assert fractions["heavy"] < 1.0


def test_target_and_context_stay_disjoint_at_every_swept_masking_ratio():
    """I-JEPA's context-minus-target construction must survive the aggressive end of the sweep:
    at `heavy` the target union can swallow most of the grid, and the generator's degenerate-case
    fallback is what keeps the context non-empty. A masking cell that handed the predictor a
    patch it was asked to predict would leak the answer and produce a meaningless rank."""
    rng = np.random.default_rng(0)
    for scale_range in MASK_CONFIGS.values():
        generator = BlockMaskGenerator(
            grid_h=8, grid_w=8, num_target_blocks=NUM_TARGET_BLOCKS, target_scale_range=scale_range
        )
        for _ in range(200):
            context_idx, target_idx = generator.sample(rng)
            assert len(context_idx) > 0
            assert not (set(context_idx.tolist()) & set(target_idx.tolist()))


def test_sweep_axes_include_the_configuration_prior_slices_ran():
    """The sweep must contain the historical operating point, or it has no anchor to compare
    against -- a grid of three new masking ratios with the old one omitted could not tell a real
    effect from a shifted baseline."""
    assert TARGET_SCALE_RANGE in MASK_CONFIGS.values()
    assert MASK_CONFIGS["default"] == TARGET_SCALE_RANGE


def test_collapse_metrics_are_stable_across_torch_thread_counts():
    """`effective_rank` goes through a float32 SVD, and the sweep pins threads to 1 while the test
    suite does not -- so sweep rows and test assertions are only comparable if the metric doesn't
    depend on that choice.

    It doesn't depend on it *materially*, but it is not bit-identical either: BLAS reduction order
    varies with thread count, giving ~5e-7 relative drift. The bound is therefore relative and set
    at 1e-5, which is four orders of magnitude below the between-cell differences this slice has
    to resolve (rank differences of 0.1+ on values of 1-4). Contrast the probe in `eval.py`, which
    needed float64 because its drift *was* the same size as the effect being measured (issue
    #104) -- here it genuinely isn't, so float32 stays."""
    from jepa.eval import effective_rank

    torch.manual_seed(0)
    embeddings = torch.randn(512, 32)
    original = torch.get_num_threads()
    try:
        values = []
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            values.append(effective_rank(embeddings))
    finally:
        torch.set_num_threads(original)
    assert (max(values) - min(values)) / statistics.fmean(values) < 1e-5

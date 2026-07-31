from __future__ import annotations

import numpy as np
from jepa.masking import BlockMaskGenerator


def test_context_and_target_are_disjoint():
    gen = BlockMaskGenerator(grid_h=8, grid_w=8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        context_idx, target_idx = gen.sample(rng)
        assert set(context_idx.tolist()).isdisjoint(target_idx.tolist())


def test_indices_are_within_bounds_and_sorted_unique():
    gen = BlockMaskGenerator(grid_h=8, grid_w=8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        context_idx, target_idx = gen.sample(rng)
        for idx in (context_idx, target_idx):
            assert idx.min() >= 0
            assert idx.max() < gen.num_patches
            assert np.array_equal(idx, np.sort(idx))
            assert len(set(idx.tolist())) == len(idx)


def test_context_is_never_empty():
    gen = BlockMaskGenerator(grid_h=8, grid_w=8)
    rng = np.random.default_rng(0)
    for _ in range(50):
        context_idx, _ = gen.sample(rng)
        assert context_idx.size > 0


def test_target_coverage_matches_configured_ratio_on_average():
    # num_target_blocks * mid-scale should land the target's average patch coverage in the
    # right ballpark -- individual draws are stochastic (blocks can overlap, reducing union
    # size), so this checks the mean over many draws rather than any single sample.
    gen = BlockMaskGenerator(
        grid_h=8, grid_w=8, num_target_blocks=4, target_scale_range=(0.15, 0.2)
    )
    rng = np.random.default_rng(0)
    ratios = [gen.sample(rng)[1].size / gen.num_patches for _ in range(300)]
    mean_ratio = sum(ratios) / len(ratios)
    # 4 blocks at ~0.15-0.2 scale each, before overlap dedup, is up to ~0.6-0.8; overlap only
    # shrinks the union, so a generous [0.2, 0.8] band is the sanity check, not a tight bound.
    assert 0.2 < mean_ratio < 0.8


def test_same_rng_state_produces_bit_identical_masks():
    gen = BlockMaskGenerator(grid_h=8, grid_w=8)
    context_a, target_a = gen.sample(np.random.default_rng(42))
    context_b, target_b = gen.sample(np.random.default_rng(42))
    assert np.array_equal(context_a, context_b)
    assert np.array_equal(target_a, target_b)


def test_different_seeds_produce_different_masks():
    gen = BlockMaskGenerator(grid_h=8, grid_w=8)
    _, target_a = gen.sample(np.random.default_rng(1))
    _, target_b = gen.sample(np.random.default_rng(2))
    assert not np.array_equal(target_a, target_b)

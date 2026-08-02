"""I-JEPA-style multi-block context/target masking over a patch grid (Assran et al., CVPR 2023,
arXiv:2301.08243, section 3.1/3.2). Dataset- and model-agnostic: operates purely on patch-grid
indices, independent of `models.py`'s encoder/predictor and Task A's bouncing-ball frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sample_block(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    scale_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
) -> tuple[int, int, int, int]:
    """One rectangular block (top, left, h, w) in patch-grid units, sized as a random fraction of
    the grid area (scale_range) at a random aspect ratio (log-uniform, matching the paper's own
    sampling), clamped to fit inside the grid."""
    area = grid_h * grid_w
    target_area = area * rng.uniform(*scale_range)
    aspect = np.exp(rng.uniform(np.log(aspect_ratio_range[0]), np.log(aspect_ratio_range[1])))
    h = int(min(max(round(np.sqrt(target_area * aspect)), 1), grid_h))
    w = int(min(max(round(np.sqrt(target_area / aspect)), 1), grid_w))
    top = int(rng.integers(0, grid_h - h + 1))
    left = int(rng.integers(0, grid_w - w + 1))
    return top, left, h, w


def _block_indices(top: int, left: int, h: int, w: int, grid_w: int) -> np.ndarray:
    rows, cols = np.meshgrid(np.arange(top, top + h), np.arange(left, left + w), indexing="ij")
    return (rows * grid_w + cols).reshape(-1)


@dataclass(frozen=True)
class BlockMaskGenerator:
    """Samples `num_target_blocks` target blocks (union of their patches is the prediction
    target) and one large context block with the target union removed (the paper's own
    context-minus-target construction, so the predictor is never handed a patch it must predict).
    """

    grid_h: int
    grid_w: int
    num_target_blocks: int = 4
    target_scale_range: tuple[float, float] = (0.15, 0.2)
    target_aspect_ratio_range: tuple[float, float] = (0.75, 1.5)
    context_scale_range: tuple[float, float] = (0.85, 1.0)

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (context_idx, target_idx), sorted int64 arrays of disjoint patch indices."""
        target_set: set[int] = set()
        for _ in range(self.num_target_blocks):
            block = _sample_block(
                rng,
                self.grid_h,
                self.grid_w,
                self.target_scale_range,
                self.target_aspect_ratio_range,
            )
            target_set.update(_block_indices(*block, self.grid_w).tolist())
        target_idx = np.array(sorted(target_set), dtype=np.int64)

        top, left, h, w = _sample_block(
            rng, self.grid_h, self.grid_w, self.context_scale_range, (1.0, 1.0)
        )
        context_candidate = set(_block_indices(top, left, h, w, self.grid_w).tolist())
        context_set = context_candidate - target_set
        if not context_set:
            # Degenerate case (the sampled context block landed entirely inside the target
            # union) -- fall back to every non-target patch rather than returning an empty
            # context, which downstream encoders/predictors can't handle.
            context_set = set(range(self.num_patches)) - target_set
        context_idx = np.array(sorted(context_set), dtype=np.int64)
        return context_idx, target_idx

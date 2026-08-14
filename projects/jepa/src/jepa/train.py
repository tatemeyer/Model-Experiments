"""Trains Task B's encoder/target-encoder/predictor against Task A's bouncing-ball physics
(Arc 1 Slice 1 Task C, issue #67) -- an I-JEPA loss (predicted vs. stop-gradient target patch
embeddings), no pixel reconstruction, no contrastive negatives, per the mechanism study this
project's CLAUDE.md describes.

Frames are generated procedurally via jepa.bouncing_ball at training time (a pool of independent
single-frame samples, not the .npz Task A's mx-data entry produces) rather than loaded from a
fetched dataset file -- mirrors projects/em-piml's own convention of sampling training data live
rather than depending on a pre-fetched artifact mid-training; Task A's mx-data registration stays
the reusable, versioned reference dataset for external use.
"""

from __future__ import annotations

import numpy as np
import torch

from jepa.bouncing_ball import CANVAS_SIZE, generate_dataset
from jepa.masking import BlockMaskGenerator
from jepa.models import EMATargetEncoder, PatchEncoder, Predictor

PATCH_SIZE = 4
EMBED_DIM = 32
HIDDEN_DIM = 32
PREDICTOR_DIM = 16
EMA_MOMENTUM = 0.996
POOL_SIZE = 512
STEPS = 300
LR = 1e-3
BATCH_SIZE = 32


def _make_frame_pool(pool_size: int, seed: int) -> torch.Tensor:
    """pool_size independent single-frame samples (n_frames=1 per generated sequence -- Task C's
    baseline studies collapse-avoidance on static images, not temporal structure), normalized to
    [0, 1] float32, shape (pool_size, 1, CANVAS_SIZE, CANVAS_SIZE)."""
    dataset = generate_dataset(n_sequences=pool_size, n_frames=1, master_seed=seed)
    frames = dataset["frames"][:, 0]  # (pool_size, H, W) uint8
    return torch.from_numpy(frames).float().unsqueeze(1) / 255.0


def train_jepa(
    steps: int = STEPS,
    seed: int = 0,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    pool_size: int = POOL_SIZE,
    ema_momentum: float = EMA_MOMENTUM,
    history: list[float] | None = None,
    use_ema: bool = True,
) -> tuple[PatchEncoder, EMATargetEncoder | PatchEncoder, Predictor]:
    # seed before model construction, not after (projects/em-piml/CLAUDE.md issue #19's standing
    # rule) -- the frame pool and masking RNG are seeded from the same `seed` value but via their
    # own independent generators (numpy, not torch's global RNG), so this ordering doesn't affect
    # their determinism either way; kept first for consistency with the rule regardless.
    torch.manual_seed(seed)
    encoder = PatchEncoder(
        image_size=CANVAS_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    target_encoder: EMATargetEncoder | PatchEncoder
    if use_ema:
        target_encoder = EMATargetEncoder(encoder)
    else:
        # No-EMA ablation (Arc 1 Slice 1 Task E, issue #69): the target/teacher network is a
        # distinct, independently initialized encoder trained jointly via ordinary gradient
        # descent through the same prediction loss -- no stop-gradient, no momentum update. This
        # is the collapse failure mode EMA + stop-gradient targets exist to prevent in
        # Siamese/BYOL-style architectures (nothing stops both encoders from co-adapting to a
        # trivial constant solution that trivially minimizes the loss).
        target_encoder = PatchEncoder(
            image_size=CANVAS_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        )
    predictor = Predictor(
        embed_dim=EMBED_DIM, num_patches=encoder.num_patches, predictor_dim=PREDICTOR_DIM
    )
    mask_gen = BlockMaskGenerator(grid_h=encoder.grid_size, grid_w=encoder.grid_size)

    frame_pool = _make_frame_pool(pool_size, seed)
    batch_rng = torch.Generator().manual_seed(seed)
    mask_rng = np.random.default_rng(seed)

    trainable_params = list(encoder.parameters()) + list(predictor.parameters())
    if not use_ema:
        trainable_params += list(target_encoder.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    for _ in range(steps):
        batch_idx = torch.randint(0, pool_size, (batch_size,), generator=batch_rng)
        batch = frame_pool[batch_idx]

        context_idx_np, target_idx_np = mask_gen.sample(mask_rng)
        context_idx = torch.from_numpy(context_idx_np).long()
        target_idx = torch.from_numpy(target_idx_np).long()

        online_tokens = encoder(batch)
        context_tokens = online_tokens[:, context_idx, :]
        if use_ema:
            with torch.no_grad():
                target_tokens = target_encoder(batch)[:, target_idx, :]
        else:
            target_tokens = target_encoder(batch)[:, target_idx, :]

        predicted = predictor(context_tokens, context_idx, target_idx)
        loss = ((predicted - target_tokens) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_ema:
            target_encoder.update(encoder, ema_momentum)

        if history is not None:
            history.append(loss.item())

    return encoder, target_encoder, predictor


def main() -> None:
    history: list[float] = []
    train_jepa(history=history)
    print(f"jepa training: initial loss {history[0]:.4f} -> final loss {history[-1]:.4f}")


if __name__ == "__main__":
    main()

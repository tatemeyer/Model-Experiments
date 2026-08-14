"""Shared evaluation harness for this project's experiments: frame/target generation, the two
pooling strategies Slice 1 settled on, collapse metrics, the linear probe, and encoder
construction per variant.

Promoted verbatim out of tests/test_baseline_collapse_avoidance.py (Arc 1 Slice 1, issue #69) so
that offline sweep scripts under src/ can reuse it rather than duplicating it. Behaviour is
unchanged -- the numbers in experiments/001-baseline-collapse-avoidance.md still reproduce.
"""

from __future__ import annotations

import numpy as np
import torch

from jepa.bouncing_ball import CANVAS_SIZE, generate_dataset
from jepa.eval import effective_rank, embedding_std, linear_probe_r2
from jepa.models import PatchEncoder
from jepa.train import EMBED_DIM, HIDDEN_DIM, PATCH_SIZE, train_jepa

# Held-out probe splits use master seeds offset well clear of any training-pool seed, so a probe
# frame is never one the encoder was trained on.
PROBE_TRAIN_SEED_OFFSET = 10_000
PROBE_TEST_SEED_OFFSET = 20_000


def probe_frames_and_targets(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n single-frame samples normalized to [0, 1], plus their [x, y, vx, vy] ground truth."""
    dataset = generate_dataset(n_sequences=n, n_frames=1, master_seed=seed)
    frames = torch.from_numpy(dataset["frames"][:, 0]).float().unsqueeze(1) / 255.0
    positions = dataset["positions"][:, 0]
    velocities = dataset["velocities"][:, 0]
    targets = torch.from_numpy(np.concatenate([positions, velocities], axis=-1)).float()
    return frames, targets


def per_patch_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Every (frame, patch) pair as its own sample, (N * num_patches, embed_dim) -- the direct
    "do patch representations vary meaningfully" collapse question, not diluted by averaging
    content-bearing patches against the ~55/64 pure-background patches this task's frames have."""
    with torch.no_grad():
        tokens = encoder(frames)
    return tokens.reshape(-1, tokens.shape[-1])


def flattened_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Every patch embedding concatenated per frame, (N, num_patches * embed_dim) -- preserves
    spatial position (unlike mean-pooling) for the linear probe."""
    with torch.no_grad():
        tokens = encoder(frames)
    return tokens.reshape(tokens.shape[0], -1)


def collapse_metrics(encoder: PatchEncoder, seed: int, n_test: int = 300) -> dict[str, float]:
    """embedding_std and effective_rank over per-patch embeddings of a held-out frame set."""
    frames, _ = probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    embeddings = per_patch_embeddings(encoder, frames)
    return {
        "embedding_std": embedding_std(embeddings),
        "effective_rank": effective_rank(embeddings),
    }


def probe_r2(encoder: PatchEncoder, seed: int, n_train: int = 4000, n_test: int = 300) -> float:
    """Held-out linear-probe R^2 for position+velocity. Retained for Slice 1's regression test;
    Slice 2 deliberately does not use it (see that slice's design -- random-init sometimes scores
    highest, so it cannot support a conclusion at this scale)."""
    train_frames, train_targets = probe_frames_and_targets(n_train, seed + PROBE_TRAIN_SEED_OFFSET)
    test_frames, test_targets = probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    train_embeddings = flattened_embeddings(encoder, train_frames)
    test_embeddings = flattened_embeddings(encoder, test_frames)
    return linear_probe_r2(train_embeddings, train_targets, test_embeddings, test_targets)


def build_encoder(variant: str, seed: int, steps: int = 3000) -> PatchEncoder:
    """variant: "full" (EMA target), "no_ema" (gradient-trained target), or "random_init"
    (untrained encoder, no train_jepa call at all)."""
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

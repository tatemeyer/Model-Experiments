"""Standalone, dataset-agnostic JEPA components (Assran et al., I-JEPA, CVPR 2023,
arXiv:2301.08243): a small CNN patch encoder, an EMA-updated target encoder, and a predictor that
maps context-patch embeddings to predicted target-patch embeddings. Not wired to a training loop
or Task A's bouncing-ball data here -- see Task C (issue #67).
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class PatchEncoder(nn.Module):
    """Patchifies a (B, C, H, W) image via a strided conv (one embedding per non-overlapping
    patch_size x patch_size patch, ViT-style tokenization but without a transformer body here --
    the "small CNN encoder" the issue asks for), then a small projection head."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int = 1,
        embed_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch_size {patch_size}")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.patchify = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, num_patches, embed_dim), patch order row-major over the grid."""
        feat = self.patchify(x)
        feat = feat.flatten(2).transpose(1, 2)
        return self.proj(feat)


class EMATargetEncoder(nn.Module):
    """Deep-copies `encoder`'s architecture as the target/teacher. Every parameter is frozen
    (requires_grad=False) at construction and `forward` runs under torch.no_grad() -- weights
    move only via `update`'s exponential moving average of the online encoder, never via
    gradient, per I-JEPA's stop-gradient target."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.target = copy.deepcopy(encoder)
        for param in self.target.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target(x)

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        """target <- momentum * target + (1 - momentum) * online, per-parameter."""
        for t_param, o_param in zip(self.target.parameters(), online.parameters(), strict=True):
            t_param.mul_(momentum).add_(o_param, alpha=1 - momentum)


class Predictor(nn.Module):
    """Predicts target-patch embeddings from context-patch embeddings: a shared learnable mask
    token (+ a per-patch-position embedding) stands in for each target patch, a shallow
    self-attention stack lets context and mask tokens exchange information, and only the
    mask-token positions are read out and projected back to embed_dim."""

    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        predictor_dim: int = 32,
        num_heads: int = 2,
        depth: int = 2,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.in_proj = nn.Linear(embed_dim, predictor_dim)
        self.pos_embed = nn.Embedding(num_patches, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(predictor_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 2,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self, context_tokens: torch.Tensor, context_idx: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """context_tokens: (B, N_ctx, embed_dim). context_idx: (N_ctx,), target_idx: (N_tgt,) --
        the same mask applied across the batch (Task C's training loop may resample per call).
        Returns predicted target-patch embeddings, (B, N_tgt, embed_dim)."""
        batch = context_tokens.shape[0]
        ctx = self.in_proj(context_tokens) + self.pos_embed(context_idx).unsqueeze(0)
        n_tgt = target_idx.shape[0]
        mask_tokens = self.mask_token.view(1, 1, -1).expand(batch, n_tgt, -1)
        mask_tokens = mask_tokens + self.pos_embed(target_idx).unsqueeze(0)
        tokens = torch.cat([ctx, mask_tokens], dim=1)
        out = self.blocks(tokens)
        predicted = out[:, ctx.shape[1] :, :]
        return self.out_proj(predicted)

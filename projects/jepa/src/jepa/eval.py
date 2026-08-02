"""Collapse metrics and linear-probe evaluation harness (Arc 1 Slice 1 Task D, issue #68) --
diagnoses whether an encoder's embeddings are non-degenerate (collapse metrics) and whether they
recover the ground-truth generative factors (linear probe), the standard JEPA/SSL evaluation
protocol (Assran et al., I-JEPA, arXiv:2301.08243) this project's Slice 1 success criteria are
checked against. Dataset- and model-agnostic: takes plain tensors, not Task A's generator or
Task B's models directly.
"""

from __future__ import annotations

import torch


def embedding_std(embeddings: torch.Tensor) -> float:
    """Mean per-dimension standard deviation across the batch -- shrinks toward 0 as embeddings
    become more constant, the textbook representation-collapse signature (every input maps to
    ~the same vector). embeddings: (N, D)."""
    flat = embeddings.reshape(embeddings.shape[0], -1)
    return flat.std(dim=0).mean().item()


def effective_rank(embeddings: torch.Tensor) -> float:
    """Roy & Vetterli's effective rank (2007, "The effective rank: A measure of effective
    dimensionality"): exp(entropy of the normalized singular-value spectrum) -- a smooth,
    real-valued generalization of matrix rank. A collapsed embedding matrix concentrates variance
    into one dominant singular value (low entropy, effective rank -> 1); a non-degenerate one
    spreads it across more directions (effective rank -> min(N, D)). embeddings: (N, D)."""
    flat = embeddings.reshape(embeddings.shape[0], -1)
    singular_values = torch.linalg.svdvals(flat)
    normalized = singular_values / singular_values.sum()
    # 0*log(0) := 0 by the entropy definition's limit; clamp avoids a NaN from log(0) on a
    # numerically-zero singular value without changing the analytical result (its own term is
    # already ~0 in the limit).
    entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum()
    return torch.exp(entropy).item()


def _fit_linear_probe(embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Closed-form ordinary-least-squares fit (torch.linalg.lstsq, no new dependency) of targets
    from embeddings, with an intercept column appended. embeddings: (N, D), targets: (N, T) ->
    weights: (D+1, T)."""
    ones = torch.ones(embeddings.shape[0], 1, dtype=embeddings.dtype)
    design = torch.cat([embeddings, ones], dim=1)
    return torch.linalg.lstsq(design, targets).solution


def linear_probe_r2(
    train_embeddings: torch.Tensor,
    train_targets: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_targets: torch.Tensor,
) -> float:
    """Fits a linear probe (OLS, with intercept) on the train split, scores R^2 on the held-out
    test split -- I-JEPA's own "does this representation linearly encode the ground-truth
    generative factors" protocol (LITERATURE.md). Held-out scoring matters specifically because a
    probe with many embedding dimensions relative to sample count can otherwise fit noise in-
    sample. targets: (N, T) for T target dimensions (e.g. T=4 for [x, y, vx, vy], see
    jepa.bouncing_ball's ground truth)."""
    weights = _fit_linear_probe(train_embeddings, train_targets)
    ones = torch.ones(test_embeddings.shape[0], 1, dtype=test_embeddings.dtype)
    design = torch.cat([test_embeddings, ones], dim=1)
    predicted = design @ weights
    ss_res = ((test_targets - predicted) ** 2).sum()
    ss_tot = ((test_targets - test_targets.mean(dim=0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

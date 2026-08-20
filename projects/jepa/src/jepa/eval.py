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


PROBE_ALPHAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
_VALIDATION_FRACTION = 0.2


def _ridge_fit(design: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge solution of the normal equations, (X'X + aI)w = X'y, with the intercept column left
    unpenalized. Solved, not inverted; float64 is the caller's responsibility.

    Why ridge and not `torch.linalg.lstsq` (issue #104): this project's probe design matrix is
    4000x2049 with condition number ~1.9e8, and on it lstsq's default CPU driver (`gelsy`)
    returns a solution whose *training* residual is 15-40x worse than the SVD driver's -- it
    rank-truncates away real signal, scoring held-out R^2 at 0.16-0.66 where a faithful fit of
    the same embeddings scores ~0.977, and not reproducibly. Ridge
    conditions the system at the source instead of relying on a driver's rank heuristic, and is
    also the standard SSL linear-probe protocol (regularized, not bare OLS)."""
    n_features = design.shape[1]
    penalty = torch.full((n_features,), alpha, dtype=design.dtype)
    penalty[-1] = 0.0  # intercept column, appended last by _design_matrix
    gram = design.T @ design + torch.diag(penalty)
    return torch.linalg.solve(gram, design.T @ targets)


def _design_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Flattens to (N, D), casts to float64, and appends the intercept column last.

    float64 is not cosmetic: in float32 the same fit varies by ~4e-7 across BLAS thread counts
    (reduction order), the scale at which issue #104's original 0.9761/0.9763 discrepancy lives.
    In float64 that spread drops to ~1e-12, far below any precision this project reports."""
    flat = embeddings.reshape(embeddings.shape[0], -1).double()
    ones = torch.ones(flat.shape[0], 1, dtype=torch.float64)
    return torch.cat([flat, ones], dim=1)


def _r2(weights: torch.Tensor, design: torch.Tensor, targets: torch.Tensor) -> float:
    predicted = design @ weights
    ss_res = ((targets - predicted) ** 2).sum()
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def select_probe_alpha(design: torch.Tensor, targets: torch.Tensor) -> float:
    """Picks the ridge penalty by held-out R^2 on a validation slice carved off the *end of the
    train split* -- never the test split, so test data stays untouched by model selection. The
    slice is a deterministic tail cut, not a random subsample, so the choice is reproducible."""
    n_validation = max(1, int(round(design.shape[0] * _VALIDATION_FRACTION)))
    if n_validation >= design.shape[0]:  # too few rows to hold anything out
        return PROBE_ALPHAS[0]
    fit_design, fit_targets = design[:-n_validation], targets[:-n_validation]
    val_design, val_targets = design[-n_validation:], targets[-n_validation:]
    # max() keeps the first argmax on ties, so equal-scoring alphas resolve to the smallest.
    return max(
        PROBE_ALPHAS,
        key=lambda alpha: _r2(_ridge_fit(fit_design, fit_targets, alpha), val_design, val_targets),
    )


def linear_probe_r2(
    train_embeddings: torch.Tensor,
    train_targets: torch.Tensor,
    test_embeddings: torch.Tensor,
    test_targets: torch.Tensor,
) -> float:
    """Fits a ridge linear probe on the train split, scores R^2 on the held-out test split --
    I-JEPA's own "does this representation linearly encode the ground-truth generative factors"
    protocol (LITERATURE.md). Held-out scoring matters specifically because a probe with many
    embedding dimensions relative to sample count can otherwise fit noise in-sample. targets:
    (N, T) for T target dimensions (e.g. T=4 for [x, y, vx, vy], see jepa.bouncing_ball's ground
    truth).

    The ridge penalty is chosen per call on a validation slice of the train split; see
    `_ridge_fit` for why this replaced the plain-OLS `torch.linalg.lstsq` fit (issue #104)."""
    train_design = _design_matrix(train_embeddings)
    test_design = _design_matrix(test_embeddings)
    alpha = select_probe_alpha(train_design, train_targets.double())
    weights = _ridge_fit(train_design, train_targets.double(), alpha)
    return _r2(weights, test_design, test_targets.double())

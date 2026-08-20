from __future__ import annotations

import pytest
import torch
from jepa.bouncing_ball import CANVAS_SIZE
from jepa.eval import (
    _design_matrix,
    _r2,
    _ridge_fit,
    effective_rank,
    embedding_std,
    linear_probe_r2,
    select_probe_alpha,
)


def test_embedding_std_lower_for_degenerate_encoder_output():
    torch.manual_seed(0)
    degenerate = torch.ones(64, 32) * 0.5  # every input maps to the same vector
    healthy = torch.randn(64, 32)
    assert embedding_std(degenerate) < embedding_std(healthy)
    assert embedding_std(degenerate) == 0.0


def test_effective_rank_lower_for_degenerate_encoder_output():
    torch.manual_seed(0)
    degenerate = torch.ones(64, 32) * 0.5
    healthy = torch.randn(64, 32)
    assert effective_rank(degenerate) < effective_rank(healthy)
    # A degenerate (constant-row) matrix has exactly one nonzero singular value -> effective
    # rank exactly 1 (entropy of a one-hot distribution is 0, exp(0) = 1).
    assert abs(effective_rank(degenerate) - 1.0) < 1e-4


def test_effective_rank_of_healthy_random_embeddings_approaches_full_rank():
    torch.manual_seed(0)
    n, d = 200, 16
    healthy = torch.randn(n, d)
    # Isotropic random Gaussian data spreads variance close to evenly across all d directions --
    # effective rank should land close to d, not collapse toward 1.
    assert effective_rank(healthy) > d * 0.8


def _synthetic_targets_and_embeddings(
    n_train: int, n_test: int, target_dim: int, embed_dim: int, seed: int, correlated: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n = n_train + n_test
    targets = torch.randn(n, target_dim)
    if correlated:
        projection = torch.randn(target_dim, embed_dim)
        embeddings = targets @ projection + 0.01 * torch.randn(n, embed_dim)
    else:
        embeddings = torch.randn(n, embed_dim)
    return embeddings[:n_train], targets[:n_train], embeddings[n_train:], targets[n_train:]


def test_linear_probe_r2_higher_for_correlated_embeddings_than_noise():
    train_emb_c, train_tgt_c, test_emb_c, test_tgt_c = _synthetic_targets_and_embeddings(
        n_train=200, n_test=50, target_dim=4, embed_dim=32, seed=0, correlated=True
    )
    train_emb_n, train_tgt_n, test_emb_n, test_tgt_n = _synthetic_targets_and_embeddings(
        n_train=200, n_test=50, target_dim=4, embed_dim=32, seed=0, correlated=False
    )
    r2_correlated = linear_probe_r2(train_emb_c, train_tgt_c, test_emb_c, test_tgt_c)
    r2_noise = linear_probe_r2(train_emb_n, train_tgt_n, test_emb_n, test_tgt_n)
    assert r2_correlated > r2_noise
    # The correlated construction (linear projection + small noise) should be recovered nearly
    # exactly by a linear probe; noise embeddings, evaluated held-out, should land near/below 0.
    assert r2_correlated > 0.9
    assert r2_noise < 0.3


# --- issue #104: the probe must be correct and reproducible, not just "a fit" ------------------


def _ill_conditioned_probe_system(
    n_train: int = 4000, n_test: int = 300, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The real thing this project probes: PatchEncoder embeddings of bouncing-ball frames against
    their [x, y, vx, vy] ground truth. Built from the shared harness rather than synthetic
    tensors because the defect in issue #104 only appears on a genuinely ill-conditioned design
    matrix (4000x2049, cond ~1.9e8) -- well-conditioned synthetic data hides it entirely."""
    from jepa.harness import (
        PROBE_TEST_SEED_OFFSET,
        PROBE_TRAIN_SEED_OFFSET,
        flattened_embeddings,
        probe_frames_and_targets,
    )
    from jepa.models import PatchEncoder
    from jepa.train import EMBED_DIM, HIDDEN_DIM, PATCH_SIZE

    torch.manual_seed(seed)
    encoder = PatchEncoder(
        image_size=CANVAS_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    train_frames, train_targets = probe_frames_and_targets(n_train, seed + PROBE_TRAIN_SEED_OFFSET)
    test_frames, test_targets = probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    return (
        flattened_embeddings(encoder, train_frames),
        train_targets,
        flattened_embeddings(encoder, test_frames),
        test_targets,
    )


@pytest.mark.slow
def test_linear_probe_r2_is_reproducible_across_repeated_calls():
    """Issue #104's original symptom: repeated calls on bit-identical inputs returned different
    values (0.9761533737 three times, 0.9763703942 once)."""
    train_emb, train_tgt, test_emb, test_tgt = _ill_conditioned_probe_system()
    values = [linear_probe_r2(train_emb, train_tgt, test_emb, test_tgt) for _ in range(4)]
    assert len(set(values)) == 1, f"probe R^2 not reproducible across calls: {values}"


@pytest.mark.slow
def test_linear_probe_r2_is_stable_across_blas_thread_counts():
    """The same fit must not depend on how many threads BLAS happens to use -- CI runners and dev
    machines differ. float32 drifts ~4e-7 here (reduction order); the float64 solve holds ~1e-12,
    so 1e-9 is a tolerance the implementation meets with margin but a float32 one would not."""
    train_emb, train_tgt, test_emb, test_tgt = _ill_conditioned_probe_system()
    original = torch.get_num_threads()
    try:
        values = []
        for threads in (1, 2, 4):
            torch.set_num_threads(threads)
            values.append(linear_probe_r2(train_emb, train_tgt, test_emb, test_tgt))
    finally:
        torch.set_num_threads(original)
    assert max(values) - min(values) < 1e-9, f"probe R^2 drifts across thread counts: {values}"


@pytest.mark.slow
def test_linear_probe_r2_recovers_position_from_ill_conditioned_embeddings():
    """The substantive half of issue #104, and the assertion that fails against the old
    implementation. On this exact system the previous float32 `torch.linalg.lstsq` fit returned
    0.363583, 0.443417, 0.612194 and 0.368235 on four *identical* calls -- not because the
    representation was bad, but because the default CPU driver
    (`gelsy`) applies an rcond-based rank cut that, at float32 precision on a cond ~1.9e8 matrix,
    truncated away most of the real signal. A faithful fit of the same embeddings scores ~0.98.

    The bound is deliberately far below what the implementation achieves and far above the
    broken value, so it catches the catastrophe without pinning an exact number."""
    train_emb, train_tgt, test_emb, test_tgt = _ill_conditioned_probe_system()
    r2 = linear_probe_r2(train_emb, train_tgt, test_emb, test_tgt)
    assert r2 > 0.9, (
        f"probe R^2 {r2:.4f} -- the issue-#104 rank-truncation regression returns 0.36-0.61"
    )


@pytest.mark.slow
def test_linear_probe_solve_path_does_not_truncate_rank():
    """Guards the solver itself rather than the end-to-end score: at a negligible penalty the
    ridge path must reproduce a true least-squares fit, so any future change that reintroduces a
    rank heuristic (or drops the float64 cast) shows up here as an inflated training residual.
    `gelsd` is the SVD-based reference -- a genuine least-squares solution, not this fitter
    re-tested against itself."""
    train_emb, train_tgt, _, _ = _ill_conditioned_probe_system()
    design = _design_matrix(train_emb)
    targets = train_tgt.double()

    reference = torch.linalg.lstsq(design, targets, driver="gelsd").solution
    reference_residual = ((targets - design @ reference) ** 2).sum().item()
    negligible = ((targets - design @ _ridge_fit(design, targets, 1e-10)) ** 2).sum().item()

    assert negligible < reference_residual * 1.05, (
        f"solve path does not reach the least-squares optimum: {negligible:.4e} vs "
        f"{reference_residual:.4e} (issue #104's failure was 15-40x at float32)"
    )


@pytest.mark.slow
def test_selected_ridge_penalty_beats_unregularized_ols_held_out():
    """The penalty has to earn its place: the selected alpha must score at least as well on the
    held-out split as a bare least-squares fit. (It trades away training residual to do so, which
    is the point of regularizing -- so training residual is the wrong thing to compare.)"""
    train_emb, train_tgt, test_emb, test_tgt = _ill_conditioned_probe_system()
    design, targets = _design_matrix(train_emb), train_tgt.double()
    test_design, test_targets = _design_matrix(test_emb), test_tgt.double()

    ols = torch.linalg.lstsq(design, targets, driver="gelsd").solution
    ridge = _ridge_fit(design, targets, select_probe_alpha(design, targets))
    assert _r2(ridge, test_design, test_targets) >= _r2(ols, test_design, test_targets)

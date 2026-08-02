from __future__ import annotations

import torch
from jepa.eval import effective_rank, embedding_std, linear_probe_r2


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

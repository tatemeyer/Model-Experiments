from __future__ import annotations

import torch
from jepa.models import EMATargetEncoder, PatchEncoder, Predictor


def _state_dicts_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_patch_encoder_output_shape():
    encoder = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    x = torch.randn(5, 1, 32, 32)
    out = encoder(x)
    assert out.shape == (5, 64, 16)  # (32/4)^2 = 64 patches


def test_patch_encoder_rejects_non_divisible_patch_size():
    try:
        PatchEncoder(image_size=32, patch_size=5, embed_dim=16)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-divisible patch_size")


def test_patch_encoder_same_seed_bit_identical_init():
    torch.manual_seed(0)
    a = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    torch.manual_seed(0)
    b = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    assert _state_dicts_equal(a.state_dict(), b.state_dict())


def test_predictor_same_seed_bit_identical_init():
    torch.manual_seed(0)
    a = Predictor(embed_dim=16, num_patches=64, predictor_dim=8, num_heads=2, depth=1)
    torch.manual_seed(0)
    b = Predictor(embed_dim=16, num_patches=64, predictor_dim=8, num_heads=2, depth=1)
    assert _state_dicts_equal(a.state_dict(), b.state_dict())


def test_predictor_output_shape():
    torch.manual_seed(0)
    predictor = Predictor(embed_dim=16, num_patches=64, predictor_dim=8, num_heads=2, depth=1)
    context_tokens = torch.randn(3, 20, 16)
    context_idx = torch.arange(20)
    target_idx = torch.arange(20, 30)
    out = predictor(context_tokens, context_idx, target_idx)
    assert out.shape == (3, 10, 16)


def test_ema_target_encoder_forward_matches_a_manual_copy():
    # allclose, not equal: these are two separate forward passes through separate (but
    # identically-weighted) modules -- CPU intra-op parallelism can reorder floating-point
    # reductions by a ULP or two between the two calls, which is not a correctness bug.
    torch.manual_seed(0)
    online = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    target = EMATargetEncoder(online)
    x = torch.randn(2, 1, 32, 32)
    assert torch.allclose(target(x), online(x), atol=1e-6)


def test_ema_target_encoder_params_are_frozen():
    online = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    target = EMATargetEncoder(online)
    assert all(not p.requires_grad for p in target.parameters())


def test_ema_target_encoder_receives_no_gradient():
    # Mirrors the actual training computation: an online forward pass feeds into a loss that
    # also involves the target encoder's (stop-gradient) output. Only the online encoder's
    # parameters should end up with a non-None .grad after backward().
    online = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    target = EMATargetEncoder(online)
    x = torch.randn(2, 1, 32, 32)

    online_out = online(x)
    target_out = target(x)
    loss = ((online_out - target_out) ** 2).mean()
    loss.backward()

    assert all(p.grad is not None for p in online.parameters())
    assert all(p.grad is None for p in target.parameters())


def test_ema_update_moves_target_toward_online_by_exactly_one_minus_momentum():
    torch.manual_seed(0)
    online = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    target = EMATargetEncoder(online)

    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)  # displace online params by a known, uniform amount

    before = {k: v.clone() for k, v in target.target.state_dict().items()}
    momentum = 0.99
    target.update(online, momentum)
    after = target.target.state_dict()

    for key, online_param in online.state_dict().items():
        expected = momentum * before[key] + (1 - momentum) * online_param
        assert torch.allclose(after[key], expected, atol=1e-6)


def test_ema_update_does_not_touch_online_params():
    torch.manual_seed(0)
    online = PatchEncoder(image_size=32, patch_size=4, embed_dim=16, hidden_dim=16)
    target = EMATargetEncoder(online)
    before = {k: v.clone() for k, v in online.state_dict().items()}
    target.update(online, momentum=0.9)
    after = online.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)

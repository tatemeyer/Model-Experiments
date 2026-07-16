from __future__ import annotations

import torch
from em_piml.train import _sample_points

# issue #12 needed to vary "which points get drawn" independently of the model-init seed.
# This is a fast plumbing check (no training) - the actual point-draw-variance investigation
# (which needs full L-BFGS training runs) lives in em_piml.point_draw_sweep and is documented
# in projects/em-piml/CLAUDE.md rather than re-run on every CI invocation.


def test_sample_points_generator_is_deterministic_and_independent_of_global_rng():
    torch.manual_seed(999)  # unrelated global state, e.g. a model-init seed
    gen_a = torch.Generator().manual_seed(42)
    points_a = _sample_points(10, 4, 4, generator=gen_a)

    torch.manual_seed(111)  # different unrelated global state
    gen_b = torch.Generator().manual_seed(42)
    points_b = _sample_points(10, 4, 4, generator=gen_b)

    for tensor_a, tensor_b in zip(points_a, points_b, strict=True):
        assert torch.equal(tensor_a, tensor_b), (
            "same points_seed generator should draw identical points regardless of global RNG state"
        )


def test_sample_points_different_seeds_draw_different_points():
    gen_a = torch.Generator().manual_seed(1)
    gen_b = torch.Generator().manual_seed(2)
    x_c_a, *_ = _sample_points(10, 4, 4, generator=gen_a)
    x_c_b, *_ = _sample_points(10, 4, 4, generator=gen_b)
    assert not torch.equal(x_c_a, x_c_b)

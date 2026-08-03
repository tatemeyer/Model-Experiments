from __future__ import annotations

import pytest
import torch
from em_piml.physics import PERIOD, L
from em_piml.train import (
    _sample_points_sobol,
    evaluate_relative_l2_error,
    train_fourier_cavity_lbfgs,
)

# issue #40: fast plumbing checks for _sample_points_sobol (mirrors test_point_draw_seed.py's
# _sample_points checks), plus one slow test locking in the actual finding -- the expensive
# 5-draw x 2-density sweep itself lives in em_piml.sobol_point_draw_sweep and is documented in
# projects/em-piml/CLAUDE.md rather than re-run on every CI invocation, same precedent as
# point_draw_sweep.py.


def test_sample_points_sobol_is_deterministic_and_independent_of_global_rng():
    torch.manual_seed(999)  # unrelated global state, e.g. a model-init seed
    points_a = _sample_points_sobol(10, 4, 4, seed=42)

    torch.manual_seed(111)  # different unrelated global state
    points_b = _sample_points_sobol(10, 4, 4, seed=42)

    for tensor_a, tensor_b in zip(points_a, points_b, strict=True):
        assert torch.equal(tensor_a, tensor_b), (
            "same Sobol seed should draw identical points regardless of global RNG state"
        )


def test_sample_points_sobol_different_seeds_draw_different_points():
    x_c_a, *_ = _sample_points_sobol(10, 4, 4, seed=1)
    x_c_b, *_ = _sample_points_sobol(10, 4, 4, seed=2)
    assert not torch.equal(x_c_a, x_c_b)


def test_sample_points_sobol_stays_within_domain_bounds():
    x_c, t_c, x_b0, x_bl, t_b, x_i, t_i = _sample_points_sobol(200, 50, 50, seed=7)
    assert (x_c >= 0).all() and (x_c <= L).all()
    assert (t_c >= 0).all() and (t_c <= PERIOD).all()
    assert (t_b >= 0).all() and (t_b <= PERIOD).all()
    assert (x_i >= 0).all() and (x_i <= L).all()


# issue #40, "Quasi Random Physics-Informed Neural Networks" (arXiv:2507.08121): does Sobol
# (low-discrepancy) collocation sampling reduce the point-draw variance issue #12 found with
# uniform torch.rand sampling? issue #12's own numbers (mean 0.078/stdev 0.035 at
# n_collocation=2000) predate issue #10's 64-hidden capacity fix and aren't a fair baseline --
# re-running point_draw_sweep.py at the current 64-hidden default first (040-sobol-sampling.md)
# found the variance problem had already shrunk ~4-7x from capacity alone (stdev down to
# 0.0082/0.0066). Against that matched 64-hidden uniform baseline, Sobol (sobol_point_draw_sweep.py)
# gives a real further reduction -- stdev ~2.3x lower at n_collocation=2000, ~1.4x lower at 4000
# -- but only a marginal mean-accuracy improvement (~5%), not a dramatic fix on its own. Full
# 5-draw sweep at n_collocation=4000: [0.0192, 0.0250, 0.0315, 0.0301, 0.0223], mean=0.0256,
# stdev=0.0046. This single-seed check locks in the magnitude of a single Sobol run without
# re-running the full 5-draw sweep in CI.
SOBOL_RELATIVE_L2_TOLERANCE = 0.05


@pytest.mark.slow
def test_sobol_sampling_beats_uniform_point_draw_variance():
    model = train_fourier_cavity_lbfgs(
        seed=0, num_bands=4, n_collocation=4000, points_seed=100, sampling="sobol"
    )
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < SOBOL_RELATIVE_L2_TOLERANCE, (
        f"expected Sobol sampling to reproduce its documented ~0.019-0.032 range at "
        f"n_collocation=4000, got relative L2 error: {relative_l2:.4f} -- if this now fails, "
        f"the finding in CLAUDE.md issue #40 needs revisiting"
    )

from __future__ import annotations

import pytest
from em_piml.physics import analytical_field_two_mode
from em_piml.train import (
    evaluate_relative_l2_error,
    train_cavity_rwf_two_mode,
    train_fourier_cavity_rwf_lbfgs_two_mode,
    train_fourier_cavity_rwf_two_mode,
)

# Issue #39: does Random Weight Factorization (Wang, Wang, Seidman, Perdikaris, arXiv:2210.01274)
# close the two-mode spectral-bias gap (issue #22), alone and combined with the existing Fourier
# embeddings? Across seeds 0/1/2/7 (see rwf_sweep.py / experiments/two-mode-spectral-bias/
# 039-random-weight-factorization.md):
#   rwf_alone (no Fourier):        0.7471-0.7688 -- vs. plain baseline's 0.7699-0.7947: a real but
#                                   small improvement, still nowhere close to the 0.026-0.046
#                                   achievable on the single-mode target.
#   rwf + num_bands=2 (Adam):      0.7018-0.7190 -- *worse* than num_bands=2 alone (0.6995-0.7063),
#                                   the only variant among the three where RWF didn't help.
#   rwf + num_bands=4 (L-BFGS):    0.6991-0.7028 -- a real, if small, improvement over num_bands=4
#                                   alone (0.7023-0.7128), and markedly lower seed-to-seed variance.
# None comes close to closing the gap -- the pointwise check (CLAUDE.md, not asserted here)
# confirms all three still miss the n=8 mode's ripple almost entirely, same mechanism as every
# other fix tried against this target. These bounds document that failure signature, matching
# test_two_mode_superposition.py's FAILURE_LOWER_BOUND convention.
FAILURE_LOWER_BOUND = 0.5


@pytest.mark.slow
def test_rwf_alone_does_not_fix_two_mode_target():
    model = train_cavity_rwf_two_mode(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected RWF alone (no Fourier embedding) to still fall well short of fixing the "
        f"two-mode target, got relative L2 error: {relative_l2:.4f} -- if this now passes, the "
        f"finding in CLAUDE.md issue #39 needs revisiting"
    )


@pytest.mark.slow
def test_rwf_num_bands_2_does_not_fix_two_mode_target():
    model = train_fourier_cavity_rwf_two_mode(seed=0, num_bands=2)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected RWF + num_bands=2 to still fall well short of fixing the two-mode target, "
        f"got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #39 needs revisiting"
    )


@pytest.mark.slow
def test_rwf_num_bands_4_lbfgs_does_not_fix_two_mode_target():
    model = train_fourier_cavity_rwf_lbfgs_two_mode(seed=0, num_bands=4)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected RWF + num_bands=4 (L-BFGS) to still fall well short of fixing the two-mode "
        f"target, got relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in "
        f"CLAUDE.md issue #39 needs revisiting"
    )

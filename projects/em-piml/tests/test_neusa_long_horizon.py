from __future__ import annotations

import pytest
from em_piml.physics import PERIOD
from em_piml.train import evaluate_relative_l2_error, train_cavity_neusa_long_horizon

# Issue #36: every prior fix against the long-horizon collapse (issue #23) -- causal reweighting
# (#23), pseudo-sequence tokenization (#30), curriculum training (#32), NTK reweighting (#34) --
# was a modification layered on the same plain coordinate-input MLP (CavityPINN). This tests a
# genuinely different architecture: NeuSA (Bizzi et al., "Neuro-Spectral Architectures for Causal
# Physics-Informed Networks," NeurIPS 2025, arXiv:2509.04966) -- project E(x,t) onto the Dirichlet
# sine basis (satisfies BC by construction), reduce the wave equation to a finite-dimensional ODE
# for the spectral coefficients (IC set exactly via quadrature, not learned), and integrate with a
# small learned correction term via hand-rolled RK4.
#
# Result: essentially solves it, unlike every prior fix. Relative L2 error ~1e-3 to 1e-2 across
# seeds 0/1/2/7 at t_max=5*PERIOD, vs. uniform (0.9225-0.9255), causal (0.9230-0.9251),
# pseudo-sequence (0.9792-1.1015), curriculum (0.8985-0.9141), NTK-reweighted (0.9999-1.1336). See
# experiments/long-horizon-collapse/036-neusa-long-horizon.md for the full derivation, per-seed
# numbers, and pointwise verification that the oscillation is sustained (not the degenerate
# near-constant collapse every plain-MLP fix in this thread showed).
#
# The shipped config trains on a 1-period horizon (train_horizon_periods=1.0, default) while
# evaluating at the full 5-period horizon -- a deliberate, documented deviation from every prior
# long-horizon function's "train == eval horizon" convention, justified by NeuSA's architectural
# extrapolation (BC/IC are exact by construction, not sampled; the learned vector field is
# integrated fresh at eval time regardless of training horizon) and this project's CI-runtime
# budget (training at the naive matching 5-period horizon measured ~5.3s/training-step -- backprop
# through the O(n_steps) sequential RK4 unroll dominates, and n_steps scales with the training
# horizon; 1 period is 5x cheaper and the result above shows it costs no accuracy).
SUCCESS_UPPER_BOUND = 0.05  # ~5-50x margin above the observed ~1e-3 to 1e-2 range
T_MAX = 5.0 * PERIOD  # 5 periods of this project's fundamental mode


@pytest.mark.slow
def test_neusa_solves_long_horizon_collapse():
    """Unlike every prior fix in this thread, NeuSA should essentially solve the long-horizon
    target -- relative L2 error close to zero, not merely an improvement over ~0.9-1.1 failure."""
    model = train_cavity_neusa_long_horizon(seed=0)
    relative_l2 = evaluate_relative_l2_error(model, t_max=T_MAX)
    assert relative_l2 < SUCCESS_UPPER_BOUND, (
        f"expected NeuSA to essentially solve the long-horizon target, got relative L2 error: "
        f"{relative_l2:.4f} -- if this regressed, the finding in CLAUDE.md issue #36 needs "
        f"revisiting"
    )


def test_neusa_long_horizon_is_deterministic():
    """Same seed -> bit-identical result, per this project's standing determinism rule (see
    CLAUDE.md's 'seed the RNG before constructing the model' pitfall)."""
    model_a = train_cavity_neusa_long_horizon(seed=0, steps=5)
    model_b = train_cavity_neusa_long_horizon(seed=0, steps=5)
    params_a = [p.detach().clone() for p in model_a.parameters()]
    params_b = [p.detach().clone() for p in model_b.parameters()]
    assert len(params_a) == len(params_b)
    assert all(
        (pa == pb).all() for pa, pb in zip(params_a, params_b)
    ), "same seed produced different parameters -- determinism broken"

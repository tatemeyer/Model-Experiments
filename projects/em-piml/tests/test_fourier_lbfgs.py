from __future__ import annotations

from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs

# NOT the standard 0.1 baseline bar (see test_fourier_embedding.py) — L-BFGS does not fully
# resolve the num_bands=4 instability, it only partially improves on Adam's total failure
# (~1.0 relative L2). This asserts that partial improvement, honestly, not a "fixed it" claim.
# See projects/em-piml/CLAUDE.md for the full investigation.
PARTIAL_IMPROVEMENT_BOUND = 0.95


def test_lbfgs_partially_but_not_fully_fixes_num_bands_4_instability():
    """num_bands=4 collapses under Adam (~1.0-1.04 relative L2, see CLAUDE.md).

    L-BFGS consistently converges to ~0.8-0.86 instead — meaningfully better than Adam's
    total failure, but nowhere near the 0.1 bar the num_bands=2 variant meets.
    """
    model = train_fourier_cavity_lbfgs(seed=0, num_bands=4)
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < PARTIAL_IMPROVEMENT_BOUND, (
        f"expected L-BFGS to at least partially improve on Adam's ~1.0 total failure, "
        f"got relative L2 error: {relative_l2:.4f}"
    )

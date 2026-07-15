from __future__ import annotations

from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs

# Not the standard 0.1 baseline bar (see test_fourier_embedding.py). A denser fixed
# collocation/boundary/initial set (2000/400/400, up from 200/64/64) resolved most of the
# num_bands=4 instability from issue #6 — order-of-magnitude better than the ~0.8 plateau at
# 200 points — but results are noisy across seeds/density levels (observed 0.065-0.104 across
# seeds 0/1/2/7 at this density), not reliably under 0.1. See projects/em-piml/CLAUDE.md.
DENSER_COLLOCATION_BOUND = 0.15


def test_denser_collocation_set_substantially_improves_num_bands_4_lbfgs():
    """At 200 points, L-BFGS plateaus at ~0.8 relative L2 (issue #6). At 2000, ~0.065-0.104."""
    model = train_fourier_cavity_lbfgs(seed=0, num_bands=4)
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < DENSER_COLLOCATION_BOUND, (
        f"expected the denser collocation set to substantially improve on the 200-point "
        f"~0.8 plateau, got relative L2 error: {relative_l2:.4f}"
    )

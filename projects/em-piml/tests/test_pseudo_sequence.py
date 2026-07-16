from __future__ import annotations

import pytest
from em_piml.train import evaluate_relative_l2_error, train_pseudo_sequence_cavity

# Issue #20: does PINNsFormer-style pseudo-sequence tokenization (Zhao et al., ICLR 2024,
# arXiv:2307.11833) beat the raw-coordinate baseline (0.026-0.046, see test_baseline_cavity.py)?
#
# No. Across seeds 0/1/2/7, relative L2 error lands at 0.958-1.383 -- the model's own PDE
# residual/BC/IC training loss converges to near-zero (verified directly, not just inferred from
# this bound), yet the resulting field does not match the analytical solution. This was
# extensively re-checked before concluding it's a real finding rather than a bug: the per-position
# Jacobian-diagonal derivative extraction (_sequence_derivative in train.py) was validated against
# finite differences; more L-BFGS iterations, denser collocation sets, per-step Adam resampling
# (rules out overfitting a fixed point set), and rescaling `dt` to the problem's actual timescale
# were all tried and none closed the gap -- see projects/em-piml/CLAUDE.md for the full writeup.
# This bound is a regression/reproducibility check on that documented negative result, not an
# accuracy target: ~1.5x margin above the observed worst case (1.383).
NEGATIVE_RESULT_BOUND = 2.0


@pytest.mark.slow
def test_pseudo_sequence_does_not_beat_baseline():
    """Documents issue #20's negative result: trains fine by its own loss, wrong global solution."""
    model = train_pseudo_sequence_cavity(seed=0)
    relative_l2 = evaluate_relative_l2_error(model)
    assert relative_l2 < NEGATIVE_RESULT_BOUND, (
        f"relative L2 error {relative_l2:.4f} exceeds even the documented negative-result range "
        f"(0.958-1.383 across seeds 0/1/2/7) -- something changed, re-verify against CLAUDE.md"
    )

from __future__ import annotations

import pytest
from em_piml.dielectric import (
    EPS_1,
    EPS_2,
    PERIOD,
    analytical_field_dielectric,
    verify_reference_solution,
)
from em_piml.train import evaluate_relative_l2_error, train_dielectric_cavity

# issue #46: reduced but real, actually-run capacity/step budget (see the experiment write-up's
# "Scope reduction" section) -- 600 steps instead of the project's usual 4000, chosen to keep this
# a genuinely fast regression check rather than reproduce the exploratory sweep's full runtime.
STEPS = 600


def test_reference_solution_satisfies_bcs_and_transmission_conditions():
    """Locks in the closed-form (given numerically-solved OMEGA) reference solution: continuous
    at both boundaries, continuous value+slope at the interface, and a genuine curvature kink of
    exactly EPS_2/EPS_1 -- the "numerically verified reference solution" issue #46 asks for."""
    diagnostics = verify_reference_solution()
    assert diagnostics["curvature_jump_ratio"] == pytest.approx(EPS_2 / EPS_1, abs=1e-6)


@pytest.mark.slow
def test_capacity_reduces_relative_l2_error():
    """Headline finding: unlike issue #25's negative capacity result on the global two-mode
    target, widening hidden (same architecture family, same Adam optimizer, no L-BFGS/SOAP) gives
    a real drop in relative L2 error on this local dielectric-interface kink target. Same seed,
    same reduced steps=600 budget for both -- capacity is the only variable."""
    small = train_dielectric_cavity(hidden=16, seed=0, steps=STEPS)
    large = train_dielectric_cavity(hidden=256, seed=0, steps=STEPS)
    small_err = evaluate_relative_l2_error(
        small, field_fn=analytical_field_dielectric, t_max=PERIOD
    )
    large_err = evaluate_relative_l2_error(
        large, field_fn=analytical_field_dielectric, t_max=PERIOD
    )
    assert large_err < small_err, (
        f"expected hidden=256 (relative_l2={large_err:.4f}) to beat hidden=16 "
        f"(relative_l2={small_err:.4f}) -- if this now fails, the capacity-helps finding in "
        f"experiments/046-dielectric-interface-capacity.md needs revisiting"
    )

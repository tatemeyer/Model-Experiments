from __future__ import annotations

import pytest
import torch
from em_piml.model import PirateNetCavityPINN
from em_piml.physics import analytical_field_two_mode
from em_piml.train import evaluate_relative_l2_error, train_piratenets_two_mode


def test_shallow_at_init_matches_paper_eq_4_8():
    """At construction, every block's alpha=0 (a pure identity map), so the whole network must
    reduce to a linear readout of the random Fourier embedding alone (arXiv:2402.00326 eq. 4.8) --
    a fast, exact structural check independent of any training run."""
    torch.manual_seed(0)
    model = PirateNetCavityPINN(hidden=16, num_blocks=3)
    assert all(block.alpha.item() == 0.0 for block in model.blocks)

    x = torch.rand(20, 1)
    t = torch.rand(20, 1)
    with torch.no_grad():
        phi = model._embed(x, t)
        expected = model.out(phi)
        actual = model(x, t)
    assert torch.equal(actual, expected)

# Issue #41: does a PirateNets-style adaptive-residual architecture (Wang, Li, Chen, Perdikaris,
# arXiv:2402.00326) close the two-mode spectral-bias gap (issue #22)? Across seeds 0/1/2/7 at a
# CI-budget-scaled config (num_blocks=2, steps=1000 -- see piratenets_sweep.py /
# experiments/two-mode-spectral-bias/041-piratenets.md): 0.7278-0.7407 -- a real improvement over
# the plain baseline (0.7699-0.7947) but does not beat either existing Fourier-embedding fix
# (num_bands=2: 0.6995-0.7063; num_bands=4 L-BFGS/SOAP: 0.7023-0.7128). A pointwise check (not
# asserted here, see CLAUDE.md) confirms the same missing-n=8-mode failure as every prior fix in
# this thread. This bound documents that failure signature, matching
# test_two_mode_superposition.py's/test_rwf_two_mode.py's FAILURE_LOWER_BOUND convention.
FAILURE_LOWER_BOUND = 0.5


@pytest.mark.slow
def test_piratenets_does_not_fix_two_mode_target():
    model = train_piratenets_two_mode(seed=0, num_blocks=2, steps=1000)
    relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
    assert relative_l2 > FAILURE_LOWER_BOUND, (
        f"expected PirateNets to still fall well short of fixing the two-mode target, got "
        f"relative L2 error: {relative_l2:.4f} -- if this now passes, the finding in CLAUDE.md "
        f"issue #41 needs revisiting"
    )

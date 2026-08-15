"""Locks in Arc 1 Slice 2's finding (issue #97). See
../experiments/002-momentum-collapse-boundary.md for the full write-up and the measured surface.

The finding is the opposite of what the slice was set up to expect. There is no collapse boundary
at the *low*-momentum end: effective_rank is a flat, healthy plateau from momentum 0.0 through
0.999 (3.15-4.03 at 6000 steps across seeds). The boundary sits at the *high* end, where the
target barely moves and the prediction task degenerates.

Deliberately reduced step budget relative to the exploratory sweep's 6000: this reproduces the
direction of the finding as a regression check, not the full momentum x steps surface.
"""

from __future__ import annotations

import pytest
from jepa.harness import collapse_metrics
from jepa.train import train_jepa

# Shortest budget in the swept grid where both gaps are unambiguous at this test's seed.
# Measured at seed 0 (see the write-up's surface table): at 300 steps the two momenta are
# indistinguishable (effective_rank 1.616 vs 1.618 -- independently re-confirming Slice 1's
# "invisible at 300 steps"), so that is the floor. At 1000 the rank margin is 0.491
# (1.965 vs 1.474) and the loss margin is ~300x (0.31478 vs 0.00101) -- decisive, and three
# minutes rather than the nine 3000 steps would cost for no added certainty.
BOUNDARY_STEPS = 1000

# Slice 1's operating point -- the known-healthy reference.
HEALTHY_MOMENTUM = 0.996

# The top of the swept grid: over BOUNDARY_STEPS the target retains almost all of its initial
# weights, so the online encoder is chasing a nearly frozen target.
FROZEN_MOMENTUM = 0.9999

# Ceiling separating a degenerate prediction task from a real one. Observed at BOUNDARY_STEPS
# across seeds 0/1/2: momentum 0.9999 lands at 0.00046-0.00101, momentum 0.996 at 0.102-0.315 --
# two orders of magnitude apart, so this constant has enormous margin on both sides. It is not a
# universal "healthy loss" value.
DEGENERATE_LOSS_CEILING = 0.05


@pytest.mark.slow
def test_frozen_target_momentum_collapses_and_its_loss_goes_degenerate():
    """Headline finding, both halves.

    (1) Collapse: at momentum 0.9999 the target barely moves, and effective_rank falls well below
    Slice 1's 0.996 operating point -- in the full sweep it drops below even an untrained
    random-init encoder (1.88-2.00 vs 2.44-2.93).

    (2) Mechanism: the accompanying loss is *lower*, not higher. A nearly-frozen target is trivial
    to predict, so the prediction task itself degenerates and stops supplying learning signal.
    That inversion -- collapse presenting as anomalously LOW loss -- is the same signature
    projects/em-piml's long-horizon-collapse thread documents for its own trivial solution, and is
    why this test asserts a loss *ceiling* for the collapsed variant rather than a floor.

    Same seed and step budget for both, so momentum is the only variable and neither assertion
    depends on a drifting threshold for the healthy side."""
    healthy_history: list[float] = []
    healthy, _, _ = train_jepa(
        steps=BOUNDARY_STEPS,
        seed=0,
        use_ema=True,
        ema_momentum=HEALTHY_MOMENTUM,
        history=healthy_history,
    )
    frozen_history: list[float] = []
    frozen, _, _ = train_jepa(
        steps=BOUNDARY_STEPS,
        seed=0,
        use_ema=True,
        ema_momentum=FROZEN_MOMENTUM,
        history=frozen_history,
    )

    healthy_rank = collapse_metrics(healthy, seed=0)["effective_rank"]
    frozen_rank = collapse_metrics(frozen, seed=0)["effective_rank"]
    assert frozen_rank < healthy_rank, (
        f"expected momentum={FROZEN_MOMENTUM} (effective_rank={frozen_rank:.3f}) to collapse "
        f"below momentum={HEALTHY_MOMENTUM} (effective_rank={healthy_rank:.3f}) -- if this now "
        f"fails, the boundary in experiments/002-momentum-collapse-boundary.md needs revisiting"
    )

    assert frozen_history[-1] < DEGENERATE_LOSS_CEILING, (
        f"expected momentum={FROZEN_MOMENTUM}'s prediction task to be degenerate "
        f"(final loss {frozen_history[-1]:.5f} below {DEGENERATE_LOSS_CEILING}) -- a rise here "
        f"means the frozen-target mechanism described in the write-up no longer holds"
    )
    assert healthy_history[-1] > DEGENERATE_LOSS_CEILING, (
        f"expected momentum={HEALTHY_MOMENTUM} to retain a non-degenerate prediction task "
        f"(final loss {healthy_history[-1]:.5f} above {DEGENERATE_LOSS_CEILING})"
    )

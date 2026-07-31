"""Generates the golden-value fixtures device-abstraction Slice 2 (issue #59) checks bit-for-bit
CPU-default behavior against. Run once, from pre-threading `main`, before any device= threading
touches train.py/model.py -- NOT part of the test suite itself (no pytest import), and not re-run
as part of CI. See test_device_threading_bit_exact.py for the assertions that consume these files.

One function per distinct optimizer path (Adam, L-BFGS, SOAP), per the device-abstraction Arc
Charter Sec5's "at minimum, one per distinct optimizer path" bar. Small step/point counts are
deliberate -- this only needs to catch an RNG-stream/op-order regression, not produce an accurate
trained model, so it stays fast enough to run unmarked (not @pytest.mark.slow).
"""

from __future__ import annotations

from pathlib import Path

import torch
from em_piml.train import (
    train_cavity_baseline,
    train_fourier_cavity_lbfgs,
    train_fourier_cavity_soap,
)

FIXTURE_DIR = Path(__file__).parent

# Kept in sync with test_device_threading_bit_exact.py -- same call, same kwargs, must round-trip
# bit-for-bit through torch.save/torch.load for the golden-value comparison to mean anything.
CALLS = {
    "adam": lambda: train_cavity_baseline(
        steps=5, seed=0, n_collocation=20, n_boundary=8, n_initial=8
    ),
    "lbfgs": lambda: train_fourier_cavity_lbfgs(
        seed=0,
        num_bands=4,
        outer_steps=1,
        max_iter=3,
        n_collocation=50,
        n_boundary=10,
        n_initial=10,
        points_seed=1,
    ),
    "soap": lambda: train_fourier_cavity_soap(
        seed=0, num_bands=4, steps=3, n_collocation=50, n_boundary=10, n_initial=10
    ),
}


def main() -> None:
    for name, call in CALLS.items():
        model = call()
        torch.save(model.state_dict(), FIXTURE_DIR / f"{name}.pt")
        print(f"wrote {name}.pt ({sum(p.numel() for p in model.parameters())} params)")


if __name__ == "__main__":
    main()

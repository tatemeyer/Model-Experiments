from __future__ import annotations

from pathlib import Path

import torch
from em_piml.train import (
    train_cavity_baseline,
    train_fourier_cavity_lbfgs,
    train_fourier_cavity_soap,
)

# device-abstraction Arc, Slice 2 (training-loop-threading, issue #59): the CPU default path must
# stay bit-for-bit identical to pre-threading `main`. Golden fixtures were generated once, from
# pre-threading code, by tests/fixtures/device_threading_golden/generate.py -- see that file's
# docstring. One function per distinct optimizer path (Adam, L-BFGS, SOAP), per the Arc Charter
# Sec5 minimum. Calls here must match generate.py's CALLS dict exactly (same kwargs, same steps)
# for the comparison to mean anything.

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "device_threading_golden"


def _assert_state_dict_bit_exact(actual: dict[str, torch.Tensor], golden_path: Path) -> None:
    expected = torch.load(golden_path, weights_only=True)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), f"{golden_path.name}: {key} diverged"


def test_adam_path_bit_exact_on_cpu_default():
    model = train_cavity_baseline(steps=5, seed=0, n_collocation=20, n_boundary=8, n_initial=8)
    _assert_state_dict_bit_exact(model.state_dict(), FIXTURE_DIR / "adam.pt")


def test_lbfgs_path_bit_exact_on_cpu_default():
    model = train_fourier_cavity_lbfgs(
        seed=0,
        num_bands=4,
        outer_steps=1,
        max_iter=3,
        n_collocation=50,
        n_boundary=10,
        n_initial=10,
        points_seed=1,
    )
    _assert_state_dict_bit_exact(model.state_dict(), FIXTURE_DIR / "lbfgs.pt")


def test_soap_path_bit_exact_on_cpu_default():
    model = train_fourier_cavity_soap(
        seed=0, num_bands=4, steps=3, n_collocation=50, n_boundary=10, n_initial=10
    )
    _assert_state_dict_bit_exact(model.state_dict(), FIXTURE_DIR / "soap.pt")

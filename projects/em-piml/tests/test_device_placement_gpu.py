"""Device-abstraction Arc, Slice 3 (issue #60): the assertions that need *actual* GPU hardware.

Slice 1's `test_device_selection.py` covers the resolver's logic by monkeypatching
`torch.cuda.is_available()`, so it stays unmarked and fast. Nothing there can tell you whether a
tensor really lands on an accelerator -- a mocked "GPU present" branch returns the right
`torch.device` object on a machine with no GPU at all. These tests close that gap, and therefore
carry `@pytest.mark.gpu`: excluded from the default run and from CI (`ubuntu-latest` has no GPU),
run explicitly with `uv run pytest -m gpu` on a machine that has one.

They also implicitly cover the packaging half of issue #60. A default PyPI resolve on Windows
installs a torch wheel with no CUDA runtime, so `torch.cuda.is_available()` is False and every
test here skips -- the module-level skip is what a broken `[tool.uv.sources]` change would look
like from here.
"""

from __future__ import annotations

import pytest
import torch
from em_piml.device import resolve_device
from em_piml.train import evaluate_relative_l2_error, train_cavity_long_horizon

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def test_resolved_cuda_device_actually_places_tensors_on_the_gpu():
    """The claim Slice 1 structurally could not make: a tensor created on the resolved device
    reports CUDA residency, on real hardware."""
    device = resolve_device("cuda")
    assert device.type == "cuda"
    tensor = torch.zeros(8, device=device)
    assert tensor.is_cuda
    # An actual kernel, not just an allocation -- allocation alone would pass even if compute
    # were broken for this architecture (e.g. a wheel built without this sm_ target).
    assert torch.equal((tensor + 1).sum().cpu(), torch.tensor(8.0))


def test_trained_model_parameters_live_on_the_gpu():
    """`--device cuda` has to place the *model*, not just ad-hoc tensors -- this is the
    end-to-end version of issue #60's "verify GPU device placement", routed through a real
    `train_*` entry point rather than a hand-built tensor.

    Deliberately tiny (`steps=2`): this asserts placement, not convergence. The wall-clock
    CPU-vs-GPU comparison lives in `em_piml.device_timing`, run offline."""
    model = train_cavity_long_horizon(steps=2, seed=0, device="cuda")
    devices = {parameter.device.type for parameter in model.parameters()}
    assert devices == {"cuda"}, f"model parameters ended up on {devices}"


def test_gpu_training_and_eval_produce_a_finite_error():
    """Guards against the failure mode that placement checks alone miss: tensors on the right
    device, kernels running, and the numbers silently garbage (NaN/inf from an unsupported
    kernel path). Asserts finiteness and a loose sanity bound, not a specific value -- CUDA and
    CPU RNG streams differ, so a cross-device numerical match is not expected here (see
    `train_fourier_cavity_lbfgs`'s note that the `points_seed` guarantee holds within a device,
    not across devices)."""
    model = train_cavity_long_horizon(steps=50, seed=0, device="cuda")
    error = evaluate_relative_l2_error(model, device="cuda")
    assert torch.isfinite(torch.tensor(error)), f"relative L2 error was {error}"
    assert error >= 0.0

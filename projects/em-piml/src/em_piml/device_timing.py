"""CPU-vs-GPU wall-clock comparison for device-abstraction Slice 3 (issue #60).

Offline script, not imported by tests -- run it explicitly on a machine with a GPU:
`uv run python -m em_piml.device_timing`. Mirrors the other `*_sweep.py` modules here in shape
(prints a table plus results.csv-ready rows) rather than inventing a new reporting format.

Times two variants deliberately, because on a Turing-generation consumer GPU they are expected to
land on opposite sides of the question: an FP32 long-horizon Adam run and an FP64 L-BFGS run. The
GTX 1660 Ti has no tensor cores and runs FP64 at ~1/32 of its FP32 rate, so "the GPU is slower for
this workload" is a legitimate outcome to record, not a failure to explain away.
"""

from __future__ import annotations

import time

import torch

from em_piml.physics import PERIOD
from em_piml.train import (
    evaluate_relative_l2_error,
    train_cavity_long_horizon,
    train_fourier_cavity_lbfgs_fp64,
)

HORIZON_PERIODS = 5.0


def _sync(device: torch.device) -> None:
    # CUDA kernels are queued asynchronously: without this the timer would stop when the work was
    # *submitted*, not when it finished, making the GPU look arbitrarily fast.
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device: torch.device) -> tuple[float, object]:
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return time.perf_counter() - start, result


def cuda_context_init_seconds() -> float:
    """Cost of standing up the CUDA context, measured separately and reported separately.

    It is a real, once-per-process cost that a short run genuinely pays, but folding it into a
    training time would misattribute a fixed startup charge to per-step throughput."""
    start = time.perf_counter()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    return time.perf_counter() - start


def fp32_long_horizon(device: str, steps: int = 4000) -> tuple[float, float]:
    """FP32 long-horizon Adam variant (issue #23's `train_cavity_long_horizon`)."""
    resolved = torch.device(device)
    elapsed, model = _time(
        lambda: train_cavity_long_horizon(steps=steps, seed=0, device=device), resolved
    )
    error = evaluate_relative_l2_error(
        model, t_max=HORIZON_PERIODS * PERIOD, device=device, dtype=torch.float32
    )
    return elapsed, error


def fp64_lbfgs(device: str, outer_steps: int = 50) -> tuple[float, float]:
    """FP64 L-BFGS variant (issue #38's `train_fourier_cavity_lbfgs_fp64`)."""
    resolved = torch.device(device)
    elapsed, model = _time(
        lambda: train_fourier_cavity_lbfgs_fp64(seed=0, outer_steps=outer_steps, device=device),
        resolved,
    )
    error = evaluate_relative_l2_error(model, device=device, dtype=torch.float64)
    return elapsed, error


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "em_piml.device_timing: no CUDA device available -- this script exists to compare "
            "against one. Nothing to do on a CPU-only machine."
        )

    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major}{minor}"
    print(f"device: {torch.cuda.get_device_name(0)} ({capability})")
    print(f"CUDA context init: {cuda_context_init_seconds():.3f}s (paid once per process)\n")

    rows: list[tuple[str, str, float, float]] = []
    for label, fn in (("fp32_long_horizon", fp32_long_horizon), ("fp64_lbfgs", fp64_lbfgs)):
        for device in ("cpu", "cuda"):
            elapsed, error = fn(device)
            rows.append((label, device, elapsed, error))
            print(f"{label:>18} {device:>5}: {elapsed:8.2f}s  relative_l2={error:.6f}", flush=True)

    print("\n=== speedup (cpu_time / cuda_time; < 1.0 means the GPU is slower) ===")
    for label in ("fp32_long_horizon", "fp64_lbfgs"):
        cpu = next(r[2] for r in rows if r[0] == label and r[1] == "cpu")
        cuda = next(r[2] for r in rows if r[0] == label and r[1] == "cuda")
        print(f"{label:>18}: {cpu / cuda:.2f}x")

    print("\n=== results.csv rows ===")
    for label, device, elapsed, error in rows:
        print(f"60,{label},{device},wall_clock_seconds,{elapsed:.3f}")
        print(f"60,{label},{device},relative_l2,{error:.6f}")


if __name__ == "__main__":
    main()

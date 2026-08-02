from __future__ import annotations

import statistics

import torch

from em_piml.physics import analytical_field_two_mode
from em_piml.train import (
    evaluate_relative_l2_error,
    train_cavity_rwf_two_mode,
    train_fourier_cavity_rwf_lbfgs_two_mode,
    train_fourier_cavity_rwf_two_mode,
)

# Single-threaded for the same reason as num_bands_sweep.py/point_draw_sweep.py: many sequential
# full training runs on a shared/multi-tenant CPU box thrash badly under torch's default intra-op
# threading.
torch.set_num_threads(1)

# issue #39: does Random Weight Factorization (RWFLinear, arXiv:2210.01274) close the two-mode
# spectral-bias gap that raising num_bands (issue #25) only partially closed? Three variants,
# each reusing the exact shipped recipe of the two-mode function it's a drop-in replacement for --
# weight parameterization is the only variable in each comparison:
#   rwf_alone:      train_cavity_rwf_two_mode
#     vs. train_cavity_two_mode (plain, ~0.77-0.79)
#   rwf_num_bands2: train_fourier_cavity_rwf_two_mode
#     vs. train_fourier_cavity_two_mode (~0.70-0.71)
#   rwf_num_bands4: train_fourier_cavity_rwf_lbfgs_two_mode
#     vs. train_fourier_cavity_lbfgs_two_mode (~0.70-0.71)
SEEDS = (0, 1, 2, 7)


def sweep() -> dict[str, list[float]]:
    results: dict[str, list[float]] = {"rwf_alone": [], "rwf_num_bands2": [], "rwf_num_bands4": []}
    for seed in SEEDS:
        alone_model = train_cavity_rwf_two_mode(seed=seed)
        alone_err = evaluate_relative_l2_error(alone_model, field_fn=analytical_field_two_mode)
        results["rwf_alone"].append(alone_err)
        print(f"  rwf_alone      seed={seed}: relative_l2={alone_err:.4f}", flush=True)

        nb2_model = train_fourier_cavity_rwf_two_mode(seed=seed, num_bands=2)
        nb2_err = evaluate_relative_l2_error(nb2_model, field_fn=analytical_field_two_mode)
        results["rwf_num_bands2"].append(nb2_err)
        print(f"  rwf_num_bands2 seed={seed}: relative_l2={nb2_err:.4f}", flush=True)

        nb4_model = train_fourier_cavity_rwf_lbfgs_two_mode(seed=seed, num_bands=4)
        nb4_err = evaluate_relative_l2_error(nb4_model, field_fn=analytical_field_two_mode)
        results["rwf_num_bands4"].append(nb4_err)
        print(f"  rwf_num_bands4 seed={seed}: relative_l2={nb4_err:.4f}", flush=True)
    return results


def main() -> None:
    results = sweep()
    print("---")
    for variant, errors in results.items():
        print(
            f"{variant}: {[round(e, 4) for e in errors]} "
            f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
        )


if __name__ == "__main__":
    main()

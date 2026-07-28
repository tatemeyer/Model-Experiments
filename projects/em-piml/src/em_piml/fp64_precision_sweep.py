from __future__ import annotations

import statistics
import time

import torch

from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs_fp64

# Single-threaded for the same reason as num_bands_sweep.py/point_draw_sweep.py: many sequential
# full training runs on a shared/multi-tenant CPU box thrash badly under torch's default intra-op
# threading.
torch.set_num_threads(1)

# issue #38: "FP64 is All You Need" (arXiv:2505.10949) argues L-BFGS's convergence test firing
# early under FP32 explains PINN "failure modes" previously attributed to genuine local optima.
# Two configs, both num_bands=4, matching issue #38's success criteria exactly:
# (a) "original": issue #6's actual pre-density-fix, pre-capacity-fix setup (32-hidden,
#     n_collocation=200 -- NOT issue #8's already-density-fixed 2000-point default) that produced
#     the ~0.79-0.88 FP32 plateau (006-lbfgs-optimizer.md / 008-denser-collocation.md's "200
#     (original)" row: 0.822, 0.851).
# (b) "shipped": issue #10's currently-shipped config (64-hidden, n_collocation=2000/400/400),
#     documented FP32 range 0.018-0.041 (010-network-capacity.md).
# WARNING: do not run this casually -- at the full outer_steps=50/max_iter=50 budget (unchanged
# from the FP32 baselines; a reduced budget was empirically found to invalidate the result
# entirely, see experiments/num-bands-gap/038-fp64-precision.md), the "original" config takes
# ~400-410s/seed and the "shipped" config takes ~2300-2450s/seed on this project's dev hardware --
# the full 8-run sweep below costs several hours single-threaded. Run seeds/configs in parallel
# background processes if re-deriving this, as was done to produce the numbers in that file.
SEEDS = (0, 1, 2, 7)
CONFIGS = {
    "original_32hidden_200pt": dict(hidden=32, n_collocation=200, n_boundary=64, n_initial=64),
    "shipped_64hidden_2000pt": dict(hidden=64, n_collocation=2000, n_boundary=400, n_initial=400),
}


def sweep() -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    for name, kwargs in CONFIGS.items():
        errors = []
        for seed in SEEDS:
            t0 = time.time()
            model = train_fourier_cavity_lbfgs_fp64(seed=seed, num_bands=4, **kwargs)
            err = evaluate_relative_l2_error(model, dtype=torch.float64)
            errors.append(err)
            print(
                f"  fp64 {name} seed={seed}: relative_l2={err:.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )
        results[name] = errors
    return results


def main() -> None:
    results = sweep()
    for name, errors in results.items():
        print(
            f"{name}: {[round(e, 4) for e in errors]} "
            f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
        )


if __name__ == "__main__":
    main()

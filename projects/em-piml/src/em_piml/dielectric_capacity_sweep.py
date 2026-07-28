from __future__ import annotations

import statistics

import torch

from em_piml.dielectric import PERIOD, X_INT, analytical_field_dielectric
from em_piml.train import evaluate_relative_l2_error, train_dielectric_cavity

# Single-threaded for the same reason as num_bands_sweep.py/point_draw_sweep.py: many sequential
# full training runs thrash badly under torch's default intra-op threading on a shared CPU box.
torch.set_num_threads(1)

# issue #46: does network capacity help resolve a spatially localized derivative (curvature) kink
# at a fixed dielectric interface, in contrast to its established irrelevance to the *global*
# spectral-bias gap in issue #25? num_layers held fixed at the project's standard baseline depth
# (train_cavity_baseline's 3).
#
# Scope reduction (documented tradeoff, same precedent as 010-network-capacity.md's note on
# sandbox CPU oversubscription): this sandbox was running many concurrent agent sessions during
# this issue, inflating a single 4000-step/32-hidden run from the baseline's documented ~35s to
# several minutes even single-threaded, and a full 5-width x 4-seed x 4000-step sweep would not
# complete in a reasonable session. Cut to 3 capacities (smallest, a middle value, largest) x 2
# seeds x 600 steps -- small enough to run synchronously in a few minutes per run, still enough to
# show a monotonic trend across capacity and get a real (if noisier, less converged) multi-seed
# comparison. STEPS/HIDDEN_VALUES/SEEDS are left as easy knobs: rerunning with STEPS=4000,
# HIDDEN_VALUES=(16,32,64,128,256), SEEDS=(0,1,2,7) reproduces the project's usual full-budget
# convention whenever more compute/time is available (see the experiment write-up's leads).
HIDDEN_VALUES = (16, 64, 256)
SEEDS = (0, 1)
STEPS = 600
POINTWISE_HIDDEN_VALUES = (16, 256)  # smallest/largest capacity, per issue #46's >=2 requirement


def sweep() -> dict[int, list[float]]:
    results: dict[int, list[float]] = {}
    for hidden in HIDDEN_VALUES:
        errors = []
        for seed in SEEDS:
            model = train_dielectric_cavity(hidden=hidden, seed=seed, steps=STEPS)
            err = evaluate_relative_l2_error(
                model, field_fn=analytical_field_dielectric, t_max=PERIOD
            )
            errors.append(err)
            print(f"  hidden={hidden} seed={seed}: relative_l2={err:.4f}", flush=True)
        results[hidden] = errors
    return results


def pointwise_error_by_distance(
    model: torch.nn.Module, seed: int = 123, n_points: int = 2000, n_bins: int = 8
) -> list[tuple[float, float, float]]:
    # Bins by |x - X_INT| (distance from the interface), not raw x -- the question this answers
    # is "does error concentrate at the interface," so distance-from-interface is the natural
    # binning axis, not raw position along the cavity.
    torch.manual_seed(seed)
    x = torch.rand(n_points, 1)
    t = torch.rand(n_points, 1) * PERIOD
    with torch.no_grad():
        predicted = model(x, t)
        true = analytical_field_dielectric(x, t)
    abs_err = (predicted - true).abs().squeeze(-1)
    dist = (x.squeeze(-1) - X_INT).abs()
    max_dist = dist.max().item()
    bin_edges = [i * max_dist / n_bins for i in range(n_bins + 1)]
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (dist >= lo) & (dist <= hi if i == n_bins - 1 else dist < hi)
        mean_err = abs_err[in_bin].mean().item() if in_bin.any() else float("nan")
        rows.append((lo, hi, mean_err))
    return rows


def main() -> None:
    results = sweep()
    print("--- capacity sweep: relative L2 error ---")
    for hidden, errors in results.items():
        print(
            f"hidden={hidden}: {[round(e, 4) for e in errors]} "
            f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
        )

    print("--- pointwise |error| vs. distance from interface (seed=0) ---")
    for hidden in POINTWISE_HIDDEN_VALUES:
        model = train_dielectric_cavity(hidden=hidden, seed=0, steps=STEPS)
        rows = pointwise_error_by_distance(model)
        print(f"hidden={hidden}:")
        for lo, hi, mean_err in rows:
            print(f"  dist in [{lo:.3f}, {hi:.3f}]: mean|err|={mean_err:.4f}")


if __name__ == "__main__":
    main()

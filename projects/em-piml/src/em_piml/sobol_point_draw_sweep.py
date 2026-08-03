from __future__ import annotations

import statistics
from pathlib import Path

import torch
from mx_viz.io import save_results

from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs

# Repo-root-relative -- see point_draw_sweep.py's own note; same invocation convention.
OUTPUT_PATH = Path(".outputs/em-piml/sobol_point_draw_sweep.json")

# Single-threaded for the same reason as point_draw_sweep.py/num_bands_sweep.py.
torch.set_num_threads(1)

# issue #40: does Sobol (low-discrepancy quasi-random) collocation sampling reduce the
# point-draw variance point_draw_sweep.py found (issue #12)? Exact same benchmark shape --
# model-init seed held fixed, only the point-draw seed varies across draws at each fixed
# density -- with sampling="sobol" the only variable relative to that sweep.
DENSITIES = (2000, 4000)
MODEL_SEED = 0
POINT_DRAW_SEEDS = (100, 101, 102, 103, 104)


def sweep() -> dict[int, list[float]]:
    results: dict[int, list[float]] = {}
    for n_collocation in DENSITIES:
        errors = []
        for points_seed in POINT_DRAW_SEEDS:
            model = train_fourier_cavity_lbfgs(
                seed=MODEL_SEED,
                n_collocation=n_collocation,
                n_boundary=400,
                n_initial=400,
                points_seed=points_seed,
                sampling="sobol",
            )
            relative_l2 = evaluate_relative_l2_error(model)
            errors.append(relative_l2)
            print(
                f"  n_collocation={n_collocation} points_seed={points_seed}: "
                f"relative_l2={relative_l2:.4f}",
                flush=True,
            )
        results[n_collocation] = errors
    return results


def main() -> None:
    results = sweep()
    for n_collocation, errors in results.items():
        spread = max(errors) - min(errors)
        print(
            f"n_collocation={n_collocation}: {[round(e, 4) for e in errors]} "
            f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f}, "
            f"spread={spread:.4f})"
        )

    save_results(
        OUTPUT_PATH,
        {
            str(n_collocation): dict(zip(map(str, POINT_DRAW_SEEDS), errors, strict=True))
            for n_collocation, errors in results.items()
        },
        metadata={"title": "issue #40: Sobol point-draw variance (num_bands=4 L-BFGS)"},
    )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

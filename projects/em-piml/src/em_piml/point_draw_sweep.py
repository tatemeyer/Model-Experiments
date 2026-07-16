from __future__ import annotations

import statistics

import torch

from em_piml.train import evaluate_relative_l2_error, train_fourier_cavity_lbfgs

# Single-threaded: this sweep runs many sequential full training runs, and on a
# multi-tenant/shared CPU box, torch's default intra-op multithreading (grabbing all cores per
# process) causes severe thread-oversubscription thrashing when multiple such processes run
# concurrently - empirically 10-40x slower than single-threaded in that regime. One thread per
# process is faster here even though each individual run "could" use more cores in isolation.
torch.set_num_threads(1)

# issue #12: is PR #9's density non-monotonicity about point *count* or about *which* points get
# drawn? Model-init seed held fixed; only points_seed (independent of model init, see
# train_fourier_cavity_lbfgs) varies across draws at each fixed density.
DENSITIES = (2000, 4000)
MODEL_SEED = 0
POINT_DRAW_SEEDS = (100, 101, 102, 103, 104)


def sweep() -> dict[int, list[float]]:
    # n_boundary/n_initial held at PR #9's shipped 400/400 (not scaled with n_collocation) so
    # this matches "PR #9's shipped config" per issue #12's constraint - n_collocation and
    # points_seed are the only variables.
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


if __name__ == "__main__":
    main()

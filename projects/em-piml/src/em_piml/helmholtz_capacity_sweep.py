from __future__ import annotations

import statistics

import torch

from em_piml.train import evaluate_relative_l2_error_helmholtz, train_helmholtz_mode

# Single-threaded for the same reason as point_draw_sweep.py/num_bands_sweep.py: many sequential
# full training runs on a shared/multi-tenant CPU box thrash badly under torch's default intra-op
# threading.
torch.set_num_threads(1)

# issue #43: does network capacity (width, and depth) help close the spectral-bias gap on a
# time-independent Helmholtz eigenvalue waveguide-mode target, isolated from the long-horizon
# collapse mechanism (no time dimension, no causality)? mode_order=1 (easy: the fundamental,
# already well within reach of a small MLP) vs. mode_order=16 (hard: a genuinely high spatial
# frequency, chosen higher than the two-mode-spectral-bias thread's n=8 to be unambiguously past
# what a small plain MLP can track) are the two "mode orders" the issue asks for.
WIDTHS = (16, 32, 64, 128, 256)
MODE_ORDERS = {"easy": 1, "hard": 16}
SEEDS = (0, 1, 2, 7)
NUM_LAYERS = 3

# Depth sub-sweep (issue: "and depth if it changes the picture"), run only at the hard mode order
# where width alone leaves headroom to see whether depth moves the number -- at a couple of widths
# to keep this bounded.
DEPTH_WIDTHS = (32, 64)
DEPTHS = (2, 3, 4)


def sweep_width() -> dict[str, dict[int, list[float]]]:
    results: dict[str, dict[int, list[float]]] = {name: {} for name in MODE_ORDERS}
    for name, mode_order in MODE_ORDERS.items():
        for hidden in WIDTHS:
            errors = []
            for seed in SEEDS:
                model = train_helmholtz_mode(
                    mode_order=mode_order, hidden=hidden, num_layers=NUM_LAYERS, seed=seed
                )
                err = evaluate_relative_l2_error_helmholtz(model, mode_order=mode_order)
                errors.append(err)
                print(
                    f"  mode={name}(n={mode_order}) hidden={hidden} seed={seed}: "
                    f"relative_l2={err:.4f}",
                    flush=True,
                )
            results[name][hidden] = errors
    return results


def sweep_depth() -> dict[int, dict[int, list[float]]]:
    mode_order = MODE_ORDERS["hard"]
    results: dict[int, dict[int, list[float]]] = {}
    for hidden in DEPTH_WIDTHS:
        results[hidden] = {}
        for num_layers in DEPTHS:
            errors = []
            for seed in SEEDS:
                model = train_helmholtz_mode(
                    mode_order=mode_order, hidden=hidden, num_layers=num_layers, seed=seed
                )
                err = evaluate_relative_l2_error_helmholtz(model, mode_order=mode_order)
                errors.append(err)
                print(
                    f"  mode=hard(n={mode_order}) hidden={hidden} num_layers={num_layers} "
                    f"seed={seed}: relative_l2={err:.4f}",
                    flush=True,
                )
            results[hidden][num_layers] = errors
    return results


def main() -> None:
    print("--- width sweep ---")
    width_results = sweep_width()
    for name, by_width in width_results.items():
        print(f"mode={name}")
        for hidden, errors in by_width.items():
            print(
                f"  hidden={hidden}: {[round(e, 4) for e in errors]} "
                f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
            )

    print("--- depth sub-sweep (hard mode only) ---")
    depth_results = sweep_depth()
    for hidden, by_depth in depth_results.items():
        print(f"hidden={hidden}")
        for num_layers, errors in by_depth.items():
            print(
                f"  num_layers={num_layers}: {[round(e, 4) for e in errors]} "
                f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import statistics

import torch

from em_piml.physics import analytical_field_two_mode
from em_piml.train import (
    evaluate_relative_l2_error,
    train_fourier_cavity_lbfgs_two_mode,
    train_fourier_cavity_soap_two_mode,
)

# Single-threaded for the same reason as point_draw_sweep.py: many sequential full training runs
# on a shared/multi-tenant CPU box thrash badly under torch's default intra-op threading.
torch.set_num_threads(1)

# issue #25: does raising num_bands close the two-mode spectral-bias gap issue #22 found (plain
# Adam/32-hidden num_bands=2 partially helps at ~0.70, still far from fixed)? num_bands=4 is the
# first value whose basis frequencies {pi,2pi,4pi,8pi} include the 8*pi the n=8 mode needs (see
# CLAUDE.md); 6/8 test whether more headroom beyond that helps further or is just noise. Plain
# Adam destabilizes at num_bands>=4 on this target the same way issue #4 found on the single-mode
# baseline (see num_bands_probe results in CLAUDE.md), so this sweep uses the L-BFGS/SOAP recipes
# issues #10/#11 already validated for exactly this instability, rather than re-deriving a fix.
NUM_BANDS_VALUES = (2, 4, 6, 8)
SEEDS = (0, 1, 2, 7)


def sweep() -> dict[str, dict[int, list[float]]]:
    results: dict[str, dict[int, list[float]]] = {"lbfgs": {}, "soap": {}}
    for num_bands in NUM_BANDS_VALUES:
        lbfgs_errors = []
        soap_errors = []
        for seed in SEEDS:
            lbfgs_model = train_fourier_cavity_lbfgs_two_mode(seed=seed, num_bands=num_bands)
            lbfgs_err = evaluate_relative_l2_error(lbfgs_model, field_fn=analytical_field_two_mode)
            lbfgs_errors.append(lbfgs_err)
            print(
                f"  lbfgs num_bands={num_bands} seed={seed}: relative_l2={lbfgs_err:.4f}",
                flush=True,
            )

            soap_model = train_fourier_cavity_soap_two_mode(seed=seed, num_bands=num_bands)
            soap_err = evaluate_relative_l2_error(soap_model, field_fn=analytical_field_two_mode)
            soap_errors.append(soap_err)
            print(
                f"  soap  num_bands={num_bands} seed={seed}: relative_l2={soap_err:.4f}", flush=True
            )
        results["lbfgs"][num_bands] = lbfgs_errors
        results["soap"][num_bands] = soap_errors
    return results


def main() -> None:
    results = sweep()
    for optimizer_name, by_num_bands in results.items():
        print(f"--- {optimizer_name} ---")
        for num_bands, errors in by_num_bands.items():
            print(
                f"num_bands={num_bands}: {[round(e, 4) for e in errors]} "
                f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
            )


if __name__ == "__main__":
    main()

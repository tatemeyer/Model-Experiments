from __future__ import annotations

import statistics

import torch

from em_piml.physics import analytical_field_two_mode
from em_piml.train import evaluate_relative_l2_error, train_piratenets_two_mode

# Single-threaded for the same reason as rwf_sweep.py/num_bands_sweep.py.
torch.set_num_threads(1)

# issue #41: does a PirateNets-style adaptive-residual architecture (arXiv:2402.00326) close the
# two-mode spectral-bias gap (issue #22) that a plain width increase (issue #10) never tried for
# this target? num_blocks=2/steps=1000 is a reduced budget from the paper's own deeper/longer
# defaults (num_blocks=4/steps=4000 measured ~767-800s/seed -- see experiments/
# two-mode-spectral-bias/041-piratenets.md for the full timing note and the partial
# full-budget numbers gathered as a supplementary data point).
SEEDS = (0, 1, 2, 7)


def sweep() -> list[float]:
    errors = []
    for seed in SEEDS:
        model = train_piratenets_two_mode(seed=seed, num_blocks=2, steps=1000)
        relative_l2 = evaluate_relative_l2_error(model, field_fn=analytical_field_two_mode)
        errors.append(relative_l2)
        print(f"  seed={seed}: relative_l2={relative_l2:.4f}", flush=True)
    return errors


def main() -> None:
    errors = sweep()
    print(
        f"piratenets_two_mode: {[round(e, 4) for e in errors]} "
        f"(mean={statistics.mean(errors):.4f}, stdev={statistics.pstdev(errors):.4f})"
    )


if __name__ == "__main__":
    main()

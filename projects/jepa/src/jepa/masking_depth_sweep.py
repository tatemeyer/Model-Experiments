"""Arc 1 Slice 3 (issue #107): do masking ratio or predictor depth move the collapse boundary?

Slice 2 (issue #97) swept EMA momentum and found it near-irrelevant -- a flat healthy plateau
from 0.0 to 0.999, with stop-gradient rather than smoothing doing the work. Masking ratio and
predictor depth are the two levers Arc 1 named and never varied. This sweep varies them jointly,
so an interaction (e.g. a deeper predictor rescuing an aggressive masking ratio) is visible rather
than assumed away by holding one fixed.

`effective_rank` is the primary metric, deliberately. Issue #104 established that `probe_r2` is
saturated on this task -- an untrained encoder scores 0.9767 against a ~0.978 ceiling -- so it
cannot separate variants and is not collected here.

Offline exploratory script, deliberately out of the default test suite (precedent: Slice 2's
momentum_steps_sweep.py and em-piml's point_draw_sweep.py). Resumable: rows are flushed per run,
and re-invoking skips any (depth, mask, seed) already recorded.

Run: uv run python -m jepa.masking_depth_sweep
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from jepa.bouncing_ball import CANVAS_SIZE
from jepa.harness import collapse_metrics
from jepa.masking import BlockMaskGenerator
from jepa.train import NUM_TARGET_BLOCKS, PATCH_SIZE, train_jepa

# Matches Slice 2's finding that single-threaded wins at this model size (torch's intra-op
# threading loses to overhead on a model this small).
torch.set_num_threads(1)

# Predictor depth. 2 is the shipped default every prior slice ran at; 1 is the shallowest stack
# that still has a transformer block; 4 doubles it. Width (`predictor_dim`) is deliberately held
# fixed -- the issue names depth, and varying both would confound them.
DEPTHS = (1, 2, 4)

# Masking ratio, varied through the target-block scale range at a fixed block count. The nominal
# knob and the realized masked fraction are different numbers (four sampled blocks overlap), so
# `realized_target_fraction` below measures what actually happened rather than trusting the knob.
MASK_CONFIGS: dict[str, tuple[float, float]] = {
    "light": (0.05, 0.10),
    "default": (0.15, 0.20),  # the value every prior slice ran at
    "heavy": (0.30, 0.40),
}

SEEDS = (0, 1, 2)
STEPS = 3000  # where Slice 1 found the collapse gap becomes visible at all

# Same confound guard Slice 2 needed: a configuration can show a healthy-looking rank simply by
# never having trained. Rank alone cannot distinguish "learned a rich representation" from "barely
# moved from initialization"; the loss trend can.
SLOPE_WINDOW = 0.1

RESULTS_PATH = Path(".outputs/jepa/masking_depth_sweep_rows.csv")
FIELDNAMES = (
    "depth",
    "mask",
    "seed",
    "steps",
    "effective_rank",
    "embedding_std",
    "final_loss",
    "loss_slope",
    "seconds",
)


def realized_target_fraction(scale_range: tuple[float, float], samples: int = 2000) -> float:
    """Mean fraction of the patch grid actually landing in the target set, over `samples` draws.

    Reported alongside every masking result because the requested per-block scale is not the
    masked fraction: NUM_TARGET_BLOCKS blocks are sampled independently and their union is the
    target, so overlap makes the realized fraction sublinear in the knob."""
    grid = CANVAS_SIZE // PATCH_SIZE
    generator = BlockMaskGenerator(
        grid_h=grid,
        grid_w=grid,
        num_target_blocks=NUM_TARGET_BLOCKS,
        target_scale_range=scale_range,
    )
    rng = np.random.default_rng(0)
    fractions = [len(generator.sample(rng)[1]) / generator.num_patches for _ in range(samples)]
    return statistics.fmean(fractions)


def _open_results() -> tuple[csv.DictWriter, object]:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_PATH.exists()
    handle = RESULTS_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()
        handle.flush()
    return writer, handle


def _load_rows() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "depth": int(raw["depth"]),
                "mask": raw["mask"],
                "seed": int(raw["seed"]),
                "steps": int(raw["steps"]),
                "effective_rank": float(raw["effective_rank"]),
                "embedding_std": float(raw["embedding_std"]),
                "final_loss": float(raw["final_loss"]),
                "loss_slope": float(raw["loss_slope"]),
                "seconds": float(raw["seconds"]),
            }
            for raw in csv.DictReader(handle)
        ]


def _loss_slope(history: list[float]) -> float:
    """Mean loss over the last SLOPE_WINDOW of training minus the mean over the window before it.
    Negative means the loss was still falling when training stopped."""
    window = max(1, int(len(history) * SLOPE_WINDOW))
    recent = history[-window:]
    earlier = history[-2 * window : -window]
    if not earlier:
        return 0.0
    return statistics.fmean(recent) - statistics.fmean(earlier)


def run() -> list[dict]:
    rows = _load_rows()
    done = {(row["depth"], row["mask"], row["seed"]) for row in rows}
    writer, handle = _open_results()
    try:
        for depth in DEPTHS:
            for mask_name, scale_range in MASK_CONFIGS.items():
                for seed in SEEDS:
                    if (depth, mask_name, seed) in done:
                        continue
                    history: list[float] = []
                    started = time.perf_counter()
                    encoder, _, _ = train_jepa(
                        steps=STEPS,
                        seed=seed,
                        history=history,
                        predictor_depth=depth,
                        target_scale_range=scale_range,
                    )
                    elapsed = time.perf_counter() - started
                    metrics = collapse_metrics(encoder, seed)
                    row = {
                        "depth": depth,
                        "mask": mask_name,
                        "seed": seed,
                        "steps": STEPS,
                        "effective_rank": metrics["effective_rank"],
                        "embedding_std": metrics["embedding_std"],
                        "final_loss": history[-1],
                        "loss_slope": _loss_slope(history),
                        "seconds": elapsed,
                    }
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    print(
                        f"depth={depth} mask={mask_name:>7} seed={seed}: "
                        f"eff_rank={row['effective_rank']:.4f} "
                        f"emb_std={row['embedding_std']:.4f} "
                        f"loss={row['final_loss']:.5f} slope={row['loss_slope']:+.2e} "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )
    finally:
        handle.close()
    return rows


def report(rows: list[dict]) -> None:
    print("\n=== realized target fraction per masking config ===")
    for name, scale_range in MASK_CONFIGS.items():
        print(f"{name:>7} scale={scale_range}: {realized_target_fraction(scale_range):.3f}")

    print("\n=== effective_rank, mean over seeds (rows: depth, cols: mask) ===")
    header = "depth".rjust(6) + "".join(name.rjust(10) for name in MASK_CONFIGS)
    print(header)
    for depth in DEPTHS:
        cells = []
        for name in MASK_CONFIGS:
            vals = [r["effective_rank"] for r in rows if r["depth"] == depth and r["mask"] == name]
            cells.append(f"{statistics.fmean(vals):.3f}".rjust(10) if vals else "-".rjust(10))
        print(str(depth).rjust(6) + "".join(cells))

    ranks = [r["effective_rank"] for r in rows]
    print(f"\nspread across every cell: {min(ranks):.3f} - {max(ranks):.3f}")
    print("\n=== results.csv rows ===")
    for r in rows:
        variant = f"depth{r['depth']}_mask_{r['mask']}"
        print(f"107,{variant},{r['seed']},effective_rank,{r['effective_rank']:.4f}")


def main() -> None:
    report(run())


if __name__ == "__main__":
    main()

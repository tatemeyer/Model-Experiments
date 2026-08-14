"""Arc 1 Slice 2: where does representation collapse appear on the EMA-momentum axis, and how
does that boundary move with training length?

Slice 1 (issue #69) measured one point -- momentum 0.996, 3000 steps. It also found the gap is
invisible at 300 steps, so momentum and training length cannot be studied independently.

Offline exploratory script, deliberately out of the default test suite (precedent:
em-piml's point_draw_sweep.py). ~3.1 hours: 21 runs at ~8.8 min each.

Run: uv run python -m jepa.momentum_steps_sweep
"""

from __future__ import annotations

import statistics

import torch

from jepa.harness import build_encoder, collapse_metrics
from jepa.train import train_jepa

# Single-threaded is measurably faster here, not just conventional: 23.9s vs 39.1s for 300 steps
# on 12 cores (torch's intra-op threading loses to overhead at this model size).
torch.set_num_threads(1)

# Spans both degenerate ends deliberately. At 0.0 the target is the online encoder copied every
# step (stop-gradient, zero smoothing) -- distinct from use_ema=False, where the target is a
# separate network trained by gradient descent. At 0.9999 the target barely moves from its random
# initialization. 0.996 is Slice 1's operating point, included for direct comparability.
MOMENTA = (0.0, 0.9, 0.99, 0.996, 0.999, 0.9999)
SEEDS = (0, 1, 2)
CHECKPOINTS = (300, 1000, 3000, 6000)
MAX_STEPS = max(CHECKPOINTS)

# Fraction of history used for the loss-slope statistic. The high-momentum confound this exists
# for: Slice 1 measured random-init effective_rank at 2.44-2.93, ABOVE the trained model's
# 2.35-2.79, so a healthy-looking rank at momentum 0.9999 could equally mean "never moved".
# Rank alone cannot tell those apart; the loss trend can.
SLOPE_WINDOW = 0.1


def _loss_slope(history: list[float], upto: int) -> float:
    """Mean loss over the last SLOPE_WINDOW of the first `upto` steps minus the mean over the
    window immediately before it. Negative means the loss is still falling."""
    window = max(1, int(upto * SLOPE_WINDOW))
    recent = history[upto - window : upto]
    earlier = history[max(0, upto - 2 * window) : upto - window]
    if not earlier:
        return 0.0
    return statistics.mean(recent) - statistics.mean(earlier)


def _run(variant: str, momentum: float | None, seed: int, rows: list[dict]) -> None:
    """One training run to MAX_STEPS, recording metrics at every checkpoint."""
    history: list[float] = []

    def record(step: int, encoder) -> None:
        metrics = collapse_metrics(encoder, seed)
        row = {
            "variant": variant,
            "momentum": momentum,
            "seed": seed,
            "steps": step,
            "effective_rank": metrics["effective_rank"],
            "embedding_std": metrics["embedding_std"],
            "final_loss": history[step - 1],
            "loss_slope": _loss_slope(history, step),
        }
        rows.append(row)
        shown = "n/a" if momentum is None else f"{momentum:g}"
        print(
            f"  {variant:>11} m={shown:>7} seed={seed} steps={step:>4}: "
            f"rank={row['effective_rank']:.3f} std={row['embedding_std']:.4f} "
            f"loss={row['final_loss']:.5f} slope={row['loss_slope']:+.6f}",
            flush=True,
        )

    train_jepa(
        steps=MAX_STEPS,
        seed=seed,
        history=history,
        use_ema=(variant == "full"),
        ema_momentum=momentum if momentum is not None else 0.996,
        checkpoint_steps=CHECKPOINTS,
        on_checkpoint=record,
    )


def main() -> None:
    rows: list[dict] = []

    print("--- momentum sweep (EMA) ---", flush=True)
    for momentum in MOMENTA:
        for seed in SEEDS:
            _run("full", momentum, seed, rows)

    print("--- no-EMA control ---", flush=True)
    for seed in SEEDS:
        _run("no_ema", None, seed, rows)

    print("--- random-init control (untrained) ---", flush=True)
    for seed in SEEDS:
        encoder = build_encoder("random_init", seed)
        metrics = collapse_metrics(encoder, seed)
        rows.append(
            {
                "variant": "random_init",
                "momentum": None,
                "seed": seed,
                "steps": 0,
                "effective_rank": metrics["effective_rank"],
                "embedding_std": metrics["embedding_std"],
                "final_loss": float("nan"),
                "loss_slope": float("nan"),
            }
        )
        print(
            f"  random_init seed={seed}: rank={metrics['effective_rank']:.3f} "
            f"std={metrics['embedding_std']:.4f}",
            flush=True,
        )

    print("\n--- effective_rank summary (mean over seeds) ---", flush=True)
    print(f"{'variant/momentum':>18} " + " ".join(f"{c:>8}" for c in CHECKPOINTS), flush=True)
    for momentum in MOMENTA:
        cells = []
        for checkpoint in CHECKPOINTS:
            values = [
                r["effective_rank"]
                for r in rows
                if r["variant"] == "full"
                and r["momentum"] == momentum
                and r["steps"] == checkpoint
            ]
            cells.append(f"{statistics.mean(values):>8.3f}" if values else f"{'-':>8}")
        print(f"{momentum:>18g} " + " ".join(cells), flush=True)
    for checkpoint in CHECKPOINTS:
        values = [
            r["effective_rank"]
            for r in rows
            if r["variant"] == "no_ema" and r["steps"] == checkpoint
        ]
        if values:
            print(f"  no_ema @{checkpoint:>5}: {statistics.mean(values):.3f}", flush=True)
    random_values = [r["effective_rank"] for r in rows if r["variant"] == "random_init"]
    if random_values:
        print(f"  random_init:    {statistics.mean(random_values):.3f}", flush=True)

    print("\n--- loss evidence (mean over seeds, disambiguates the high-momentum confound) ---",
          flush=True)
    for momentum in MOMENTA:
        final = [
            r["final_loss"]
            for r in rows
            if r["variant"] == "full" and r["momentum"] == momentum and r["steps"] == MAX_STEPS
        ]
        slope = [
            r["loss_slope"]
            for r in rows
            if r["variant"] == "full" and r["momentum"] == momentum and r["steps"] == MAX_STEPS
        ]
        if final:
            print(
                f"  m={momentum:<8g} final_loss={statistics.mean(final):.5f} "
                f"slope={statistics.mean(slope):+.6f}",
                flush=True,
            )

    print("\n--- results.csv rows ---", flush=True)
    for row in rows:
        if row["momentum"] is None:
            variant_label = row["variant"]
            params = f'"{{""steps"":{row["steps"]}}}"'
        else:
            variant_label = f'{row["variant"]}_m{row["momentum"]:g}'
            params = f'"{{""momentum"":{row["momentum"]},""steps"":{row["steps"]}}}"'
        for metric in ("effective_rank", "embedding_std"):
            print(
                f"ISSUE,002-momentum-collapse-boundary,{variant_label},"
                f"{row['seed']},{metric},{row[metric]:.4f},{params},2026-08-14",
                flush=True,
            )


if __name__ == "__main__":
    main()

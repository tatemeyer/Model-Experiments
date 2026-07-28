from __future__ import annotations

import statistics

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_KINDS = ("box", "bar")


def plot_sweep_comparison(
    results: dict[str, dict[str, float]],
    kind: str = "box",
    ylabel: str = "relative L2 error",
    title: str | None = None,
) -> Figure:
    """Compare a metric across variants, each with one or more per-seed values.

    `results` shape matches point_draw_sweep.py's/num_bands_sweep.py's existing
    in-memory sweep-result dicts (variant name -> {seed: value}) directly, and the
    JSON format written/read by `mx_viz.io.save_results`/`load_results`.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")

    variants = list(results.keys())
    fig = Figure(figsize=(max(4.0, 1.2 * len(variants)), 4.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    if kind == "box":
        data = [list(results[v].values()) for v in variants]
        ax.boxplot(data, showmeans=True)
    else:
        means = [statistics.mean(results[v].values()) for v in variants]
        stdevs = [
            statistics.pstdev(results[v].values()) if len(results[v]) > 1 else 0.0
            for v in variants
        ]
        ax.bar(range(len(variants)), means, yerr=stdevs, capsize=4)

    ax.set_xticks(range(1, len(variants) + 1) if kind == "box" else range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig

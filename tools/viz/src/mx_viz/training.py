from __future__ import annotations

from collections.abc import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_loss_curve(
    histories: dict[str, Sequence[float]],
    log_scale: bool = True,
    title: str | None = None,
) -> Figure:
    """Plot one or more named loss-history series over training step.

    Multiple series overlay on one axes (e.g. uniform vs. causal vs. curriculum
    loss for a direct comparison) -- this project's experiments are almost always
    controlled A/B/C comparisons, not single runs viewed in isolation.
    """
    if not histories:
        raise ValueError("histories must contain at least one named series")

    fig = Figure(figsize=(6.0, 4.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for name, history in histories.items():
        ax.plot(range(len(history)), history, label=name)

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    if log_scale:
        ax.set_yscale("log")
    if len(histories) > 1:
        ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig

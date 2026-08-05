from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# Takes plain numpy arrays, not model objects -- keeps this library framework-agnostic
# (no torch/JAX dependency here; CONVENTIONS.md allows either per-project). Callers
# evaluate their model over a grid/slice themselves (e.g. em_piml.train's
# evaluate_field_grid/evaluate_field_slice) and pass the resulting arrays in.


def plot_field_heatmap(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    predicted: np.ndarray,
    true: np.ndarray,
    title: str | None = None,
) -> Figure:
    """3-panel heatmap: predicted field, true field, and |predicted - true|, over a
    2D (x, t) grid. predicted/true/x_grid/t_grid must all share the same 2D shape
    (e.g. from np.meshgrid)."""
    fig = Figure(figsize=(13.0, 4.0))
    FigureCanvasAgg(fig)

    error = np.abs(predicted - true)
    vmax = max(float(np.abs(predicted).max()), float(np.abs(true).max()))
    panels = [
        ("predicted", predicted, "RdBu_r", vmax),
        ("true", true, "RdBu_r", vmax),
        ("|error|", error, "viridis", None),
    ]

    for i, (name, data, cmap, vlim) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i)
        vmin = -vlim if vlim is not None else None
        im = ax.pcolormesh(x_grid, t_grid, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vlim)
        ax.set_xlabel("x")
        if i == 1:
            ax.set_ylabel("t")
        ax.set_title(name)
        fig.colorbar(im, ax=ax)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_field_slice(
    t_values: Sequence[float],
    predicted: Sequence[float],
    true: Sequence[float],
    x: float | None = None,
    title: str | None = None,
) -> Figure:
    """Predicted vs. true field, line plot across t_values at a fixed x -- the
    plot-equivalent of this project's many hand-tabulated pointwise-check tables
    (e.g. em-piml issues #23/#30/#32/#34's "does it collapse over time" checks)."""
    fig = Figure(figsize=(6.0, 4.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(t_values, true, label="true", linewidth=2)
    ax.plot(t_values, predicted, label="predicted", linestyle="--", marker="o", markersize=3)
    ax.set_xlabel("t")
    ax.set_ylabel(f"E(x={x}, t)" if x is not None else "E(x, t)")
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_field_frame(
    x: np.ndarray,
    true_slice: np.ndarray,
    predicted_slice: np.ndarray,
    t_value: float | None = None,
    y_lim: tuple[float, float] | None = None,
) -> Figure:
    """One frame of a true/predicted/|error| comparison at a fixed t -- a new function distinct
    from plot_field_heatmap (which already plots the full (x, t) comparison as one static image
    with t as an axis; animating *that* would mean either a meaningless progressive row-reveal
    or silently building a different figure -- see issue #62). This is that different figure:
    a per-t 1D line-plot frame, assembled over the time axis by render_field_frames + open_gif.
    y_lim fixes the true/predicted panels' y-axis across frames so a GIF doesn't visually jitter.
    """
    fig = Figure(figsize=(13.0, 4.0))
    FigureCanvasAgg(fig)
    error = np.abs(predicted_slice - true_slice)
    panels = [("true", true_slice), ("predicted", predicted_slice), ("|error|", error)]

    for i, (name, data) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i)
        ax.plot(x, data)
        ax.set_xlabel("x")
        if i == 1:
            ax.set_ylabel("E(x)")
        if y_lim is not None and name != "|error|":
            ax.set_ylim(y_lim)
        ax.set_title(name)

    if t_value is not None:
        fig.suptitle(f"t={t_value:.4f}")
    fig.tight_layout()
    return fig


def render_field_frames(
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    predicted: np.ndarray,
    true: np.ndarray,
) -> list[Figure]:
    """Build one plot_field_frame per time step from an evaluate_field_grid-shaped array set,
    stepping over grid_t's time axis -- pass the result to mx_viz.animate.open_gif.

    Assumes torch.meshgrid(..., indexing="xy")'s convention (see em_piml.train.
    evaluate_field_grid): axis 0 of grid_x/grid_t/predicted/true is t, axis 1 is x, matching
    plot_field_heatmap's own input shape so the same evaluate_field_grid() call feeds both.
    """
    x = grid_x[0, :]
    n_t = grid_t.shape[0]
    y_lim = (
        float(min(true.min(), predicted.min())),
        float(max(true.max(), predicted.max())),
    )
    return [
        plot_field_frame(x, true[i, :], predicted[i, :], t_value=float(grid_t[i, 0]), y_lim=y_lim)
        for i in range(n_t)
    ]

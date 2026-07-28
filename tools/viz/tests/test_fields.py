from __future__ import annotations

import numpy as np
from mx_viz.fields import plot_field_heatmap, plot_field_slice


def test_heatmap_has_three_panels():
    n = 5
    x = np.linspace(0, 1, n)
    t = np.linspace(0, 1, n)
    grid_x, grid_t = np.meshgrid(x, t)
    predicted = np.sin(grid_x)
    true = np.sin(grid_x) + 0.01

    fig = plot_field_heatmap(grid_x, grid_t, predicted, true)

    # 3 data panels + 3 colorbars (fig.colorbar adds its own Axes per panel)
    assert len(fig.axes) == 6
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["predicted", "true", "|error|"]


def test_slice_plots_predicted_and_true():
    t_values = [0.0, 0.5, 1.0]
    predicted = [1.0, 0.0, -1.0]
    true = [1.0, 0.1, -0.9]

    fig = plot_field_slice(t_values, predicted, true, x=0.5)

    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None
    assert ax.get_ylabel() == "E(x=0.5, t)"


def test_savefig_produces_nonempty_file(tmp_path):
    fig = plot_field_slice([0.0, 1.0], [1.0, 0.0], [1.0, 0.1])
    path = tmp_path / "out.png"
    fig.savefig(path)
    assert path.exists()
    assert path.stat().st_size > 0

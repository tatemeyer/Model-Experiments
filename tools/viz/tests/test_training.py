from __future__ import annotations

import pytest
from mx_viz.training import plot_loss_curve


def test_single_series_plots_one_line():
    fig = plot_loss_curve({"uniform": [1.0, 0.5, 0.25, 0.1]})
    ax = fig.axes[0]
    assert len(ax.lines) == 1


def test_multiple_series_plot_multiple_lines_with_legend():
    fig = plot_loss_curve({"uniform": [1.0, 0.5], "causal": [1.0, 0.6]})
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None


def test_single_series_has_no_legend():
    fig = plot_loss_curve({"uniform": [1.0, 0.5]})
    ax = fig.axes[0]
    assert ax.get_legend() is None


def test_empty_histories_raises():
    with pytest.raises(ValueError):
        plot_loss_curve({})


def test_log_scale_default():
    fig = plot_loss_curve({"a": [1.0, 0.5, 0.1]})
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"


def test_log_scale_can_be_disabled():
    fig = plot_loss_curve({"a": [1.0, 0.5, 0.1]}, log_scale=False)
    ax = fig.axes[0]
    assert ax.get_yscale() == "linear"

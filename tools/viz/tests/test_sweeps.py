from __future__ import annotations

import pytest
from mx_viz.sweeps import plot_sweep_comparison


def _sample_results():
    return {
        "uniform": {"0": 0.92, "1": 0.93, "2": 0.91},
        "causal": {"0": 0.90, "1": 0.94, "2": 0.89},
    }


def test_box_plot_has_one_box_per_variant():
    fig = plot_sweep_comparison(_sample_results(), kind="box")
    ax = fig.axes[0]
    assert len(ax.get_xticklabels()) == 2


def test_bar_plot_has_one_bar_per_variant():
    fig = plot_sweep_comparison(_sample_results(), kind="bar")
    ax = fig.axes[0]
    assert len(ax.patches) == 2


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        plot_sweep_comparison(_sample_results(), kind="pie")


def test_title_and_ylabel_applied():
    fig = plot_sweep_comparison(_sample_results(), title="demo", ylabel="metric")
    ax = fig.axes[0]
    assert ax.get_title() == "demo"
    assert ax.get_ylabel() == "metric"


def test_savefig_produces_nonempty_file(tmp_path):
    fig = plot_sweep_comparison(_sample_results())
    path = tmp_path / "out.png"
    fig.savefig(path)
    assert path.exists()
    assert path.stat().st_size > 0

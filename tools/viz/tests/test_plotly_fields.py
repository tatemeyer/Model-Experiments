from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")  # requires the mx-viz[3d] extra; skip cleanly without it

from mx_viz.plotly_fields import (  # noqa: E402
    export_html,
    export_png,
    plot_isosurface,
    plot_streamtube,
    plot_surface,
    plot_volume,
    render_orbit_gif,
)


def _make_surface_grid():
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 1.0, 6)
    grid_x, grid_y = np.meshgrid(x, y)
    z = np.sin(grid_x) * np.cos(grid_y)
    return grid_x, grid_y, z


def _make_volume_grid():
    x, y, z = np.mgrid[0:1:5j, 0:1:5j, 0:1:5j]
    value = np.sin(x) * np.cos(y) * z
    return x, y, z, value


def test_plot_surface_builds_one_trace():
    grid_x, grid_y, z = _make_surface_grid()
    fig = plot_surface(grid_x, grid_y, z, title="demo")
    assert len(fig.data) == 1
    assert fig.data[0].type == "surface"


def test_plot_isosurface_builds_one_trace():
    x, y, z, value = _make_volume_grid()
    fig = plot_isosurface(x, y, z, value, isomin=0.0, isomax=1.0)
    assert len(fig.data) == 1
    assert fig.data[0].type == "isosurface"


def test_plot_volume_builds_one_trace():
    x, y, z, value = _make_volume_grid()
    fig = plot_volume(x, y, z, value)
    assert len(fig.data) == 1
    assert fig.data[0].type == "volume"


def test_plot_streamtube_builds_one_trace():
    x, y, z, value = _make_volume_grid()
    u, v, w = value, value, value
    fig = plot_streamtube(x, y, z, u, v, w)
    assert len(fig.data) == 1
    assert fig.data[0].type == "streamtube"


def test_export_png_writes_nonempty_file(tmp_path):
    grid_x, grid_y, z = _make_surface_grid()
    fig = plot_surface(grid_x, grid_y, z)
    path = tmp_path / "surface.png"

    export_png(fig, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_render_orbit_gif_writes_nonempty_file(tmp_path):
    grid_x, grid_y, z = _make_surface_grid()
    fig = plot_surface(grid_x, grid_y, z)
    path = tmp_path / "orbit.gif"

    render_orbit_gif(fig, path, n_frames=3)

    assert path.exists()
    assert path.stat().st_size > 0


def test_render_orbit_gif_rejects_zero_frames(tmp_path):
    grid_x, grid_y, z = _make_surface_grid()
    fig = plot_surface(grid_x, grid_y, z)
    with pytest.raises(ValueError):
        render_orbit_gif(fig, tmp_path / "empty.gif", n_frames=0)


def test_export_html_has_csp_blocking_remote_requests(tmp_path):
    grid_x, grid_y, z = _make_surface_grid()
    fig = plot_surface(grid_x, grid_y, z)
    path = tmp_path / "surface.html"

    export_html(fig, path)

    html = path.read_text(encoding="utf-8")
    assert path.stat().st_size > 0
    assert "Content-Security-Policy" in html
    assert "default-src 'self'" in html
    # Confirms the CSP is neutralizing a real reference, not decorating an already-clean file --
    # plotly.js's full bundle hardcodes this host for its (here unused) mapbox trace types.
    assert "cartodb-basemaps" in html or "openstreetmap" in html


def test_export_html_preserves_mit_license_header(tmp_path):
    grid_x, grid_y, z = _make_surface_grid()
    fig = plot_surface(grid_x, grid_y, z)
    path = tmp_path / "surface.html"

    export_html(fig, path)

    html = path.read_text(encoding="utf-8")
    assert "Licensed under the MIT license" in html

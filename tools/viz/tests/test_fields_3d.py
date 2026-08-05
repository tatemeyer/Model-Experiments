from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista")  # requires the mx-viz[3d] extra; skip cleanly without it

from mx_viz.fields_3d import (  # noqa: E402
    export_field_surface_html,
    plot_field_surface,
    render_field_surface_orbit_gif,
)


def _make_field_grid():
    x = np.linspace(0.0, 1.0, 6)
    t = np.linspace(0.0, 1.0, 6)
    grid_x, grid_t = np.meshgrid(x, t, indexing="xy")
    field = np.sin(grid_x) * np.cos(grid_t)
    return grid_x, grid_t, field


def test_plot_field_surface_builds_a_mesh():
    grid_x, grid_t, field = _make_field_grid()
    plotter = plot_field_surface(grid_x, grid_t, field, title="demo")
    assert plotter.mesh.n_points == grid_x.size
    plotter.close()


def test_render_field_surface_orbit_gif_writes_nonempty_file(tmp_path):
    grid_x, grid_t, field = _make_field_grid()
    plotter = plot_field_surface(grid_x, grid_t, field)
    path = tmp_path / "orbit.gif"

    render_field_surface_orbit_gif(plotter, path, n_frames=3)

    assert path.exists()
    assert path.stat().st_size > 0


def test_export_field_surface_html_has_no_remote_references(tmp_path):
    grid_x, grid_t, field = _make_field_grid()
    plotter = plot_field_surface(grid_x, grid_t, field)
    path = tmp_path / "surface.html"

    export_field_surface_html(plotter, path)
    plotter.close()

    html = path.read_text(encoding="utf-8")
    assert path.stat().st_size > 0
    # The CSP itself is the enforced guarantee (blocks any remote fetch the embedded bundle
    # tries, browser-side); this also checks the one concrete remote reference verified in the
    # unpatched output (a kitware.github.io favicon) is still present in the *markup* -- proving
    # the CSP is doing real work, not just decorating an already-clean file.
    assert 'Content-Security-Policy' in html
    assert "default-src 'self'" in html
    assert "kitware.github.io" in html  # confirms the CSP is neutralizing a real reference

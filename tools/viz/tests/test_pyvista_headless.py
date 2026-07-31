from __future__ import annotations

import numpy as np
import pytest

# field-visualization Arc, Slice 1 (issue #58): PyVista is an optional `mx-viz[3d]` dependency,
# not part of the base install -- skip (not fail) when it isn't synced, so this file doesn't
# change the required-dependency footprint of the default `uv run pytest` collection. CI opts
# into it via `uv sync --all-extras` (ci.yml), which is what actually exercises this test there.
pyvista = pytest.importorskip("pyvista")


def test_offscreen_plotter_renders_a_screenshot_with_no_display():
    # off_screen=True is the whole point of this Slice: no X server, no window, still produces a
    # real rendered image -- the thing #58 exists to prove works in this repo's actual CI runner.
    plotter = pyvista.Plotter(off_screen=True, window_size=[64, 64])
    try:
        mesh = pyvista.Sphere()
        plotter.add_mesh(mesh)
        image = plotter.screenshot(return_img=True)
    finally:
        plotter.close()

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3  # (height, width, channels)
    assert image.shape[0] > 0 and image.shape[1] > 0
    # A crashed/blank offscreen context typically screenshots as uniformly one color (e.g. all
    # black) -- a rendered sphere against the default background should have real variation.
    assert image.std() > 0

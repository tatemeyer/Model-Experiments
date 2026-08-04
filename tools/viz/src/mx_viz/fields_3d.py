from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv

# A strict CSP, not string-stripping: PyVista's Plotter.export_html embeds the vtk.js viewer
# bundle inline (self-contained for the actual scene data), but its own runtime also injects a
# handful of <link rel="icon"> tags pointing at a remote kitware.github.io favicon -- verified
# empirically against real export_html output, not documented anywhere in PyVista's own docs
# (the same "self-contained was asserted, never verified" trap issue #62/#63 both flag). Rather
# than pattern-matching and removing that one specific minified call site (fragile across vtk.js
# versions, and blind to any other remote reference a future version might add), a CSP meta tag
# is inserted that makes the browser itself refuse *any* remote network request the embedded
# bundle might make -- enforced by the rendering browser, not by exhaustively grepping a
# minified third-party bundle for every way a URL could be constructed.
_SELF_CONTAINED_CSP = (
    '<meta http-equiv="Content-Security-Policy" '
    "content=\"default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:;\">"
)


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "3D field rendering requires the 'mx-viz[3d]' extra (pyvista + trame). Install "
            "with `uv sync --all-extras` (see tools/viz/CLAUDE.md)."
        ) from exc
    return pv


def plot_field_surface(
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    field: np.ndarray,
    title: str | None = None,
    cmap: str = "RdBu_r",
) -> pv.Plotter:
    """Rotatable 3D surface (x, t axes, field value as height) via PyVista.

    field is a single 2D (x, t)-shaped array -- caller picks which panel to render (true,
    predicted, or |error|; see em_piml.train.evaluate_field_grid). A 3-in-one-scene layout
    isn't attempted here: unlike plot_field_heatmap's flat 2D image, three overlaid 3D surfaces
    would occlude each other, so this stays one field per call, same as plot_field_frame's
    per-panel design. Always off-screen (headless rendering; see tools/viz/CLAUDE.md) --
    caller/test decides what to do with the resulting Plotter (screenshot, open_gif orbit,
    export_field_surface_html).
    """
    pv = _import_pyvista()
    z = np.zeros_like(grid_x)
    grid = pv.StructuredGrid(grid_x, grid_t, z)
    grid["field"] = np.asarray(field).ravel(order="F")
    warped = grid.warp_by_scalar("field")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(warped, scalars="field", cmap=cmap)
    plotter.add_axes()
    if title:
        plotter.add_title(title)
    return plotter


def render_field_surface_orbit_gif(
    plotter: pv.Plotter,
    path: str | Path,
    n_frames: int = 36,
    degrees_per_frame: float = 10.0,
) -> None:
    """Orbit the camera around a plot_field_surface Plotter and write the frames as a GIF.

    A camera orbit, not a time-stepping animation: unlike mx_viz.fields.render_field_frames
    (where t must step frame-by-frame because the 2D per-frame comparison has no other way to
    show time), this 3D surface already encodes t as a spatial axis (same rationale as issue
    #62's correction of the original "animate plot_field_heatmap" framing) -- rotating the
    static surface is the non-redundant way to animate it. MP4 (open_movie) isn't offered here,
    same deferral as mx_viz.animate.open_gif's module docstring (unverified imageio-ffmpeg
    license).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.open_gif(str(path))
    for _ in range(n_frames):
        plotter.camera.azimuth += degrees_per_frame
        plotter.write_frame()
    plotter.close()


def export_field_surface_html(plotter: pv.Plotter, path: str | Path) -> None:
    """Export a plot_field_surface Plotter as self-contained HTML.

    See _SELF_CONTAINED_CSP above for why this isn't just a bare plotter.export_html() call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.export_html(str(path))
    html = path.read_text(encoding="utf-8")
    if "<head>" not in html:
        raise ValueError(f"expected a <head> tag in exported HTML, none found in {path}")
    html = html.replace("<head>", f"<head>\n    {_SELF_CONTAINED_CSP}", 1)
    path.write_text(html, encoding="utf-8")

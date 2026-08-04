from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Same mitigation as mx_viz.fields_3d's PyVista export, and for the same reason: verified
# empirically against real write_html(include_plotlyjs=True) output (not documented anywhere in
# Plotly's own "self-contained" framing), the full plotly.js bundle -- embedded regardless of
# which trace types a given figure actually uses -- hardcodes remote map-tile/icon hosts
# (openstreetmap.org, mapbox.com, cartocdn.com, unpkg.com/maki) for its choropleth/mapbox trace
# types. None of Isosurface/Volume/Streamtube/Surface trigger that code path, but the strings and
# a couple of live `href`s (carto.com, plotly.com attribution links) still ship in the bundle. A
# CSP blocking any remote request is more robust than trying to strip specific bundle internals
# (fragile across plotly.js versions, and blind to any future reference).
_SELF_CONTAINED_CSP = (
    '<meta http-equiv="Content-Security-Policy" '
    "content=\"default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:;\">"
)


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly rendering requires the 'mx-viz[3d]' extra (plotly + kaleido). Install with "
            "`uv sync --all-extras` (see tools/viz/CLAUDE.md)."
        ) from exc
    return go


def _make_figure(trace, title: str | None) -> go.Figure:
    go = _import_plotly()
    fig = go.Figure(data=[trace])
    if title:
        fig.update_layout(title=title)
    return fig


def plot_surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str | None = None
) -> go.Figure:
    """Interactive Plotly Surface (z(x, y)) -- a lighter-weight sibling to mx_viz.fields_3d.
    plot_field_surface (no PyVista/VTK needed at render time, though both share the mx-viz[3d]
    extra)."""
    go = _import_plotly()
    return _make_figure(go.Surface(x=x, y=y, z=z), title)


def plot_isosurface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    value: np.ndarray,
    isomin: float | None = None,
    isomax: float | None = None,
    title: str | None = None,
) -> go.Figure:
    """Interactive Plotly Isosurface over a 3D scalar field (x/y/z/value all the same shape,
    typically from np.meshgrid + a scalar function -- flattened here since Isosurface expects
    1D coordinate arrays)."""
    go = _import_plotly()
    return _make_figure(
        go.Isosurface(
            x=np.ravel(x),
            y=np.ravel(y),
            z=np.ravel(z),
            value=np.ravel(value),
            isomin=isomin,
            isomax=isomax,
        ),
        title,
    )


def plot_volume(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    value: np.ndarray,
    opacity: float = 0.1,
    surface_count: int = 17,
    title: str | None = None,
) -> go.Figure:
    """Interactive Plotly Volume rendering over a 3D scalar field. opacity/surface_count default
    to Plotly's own documented starting point for a translucent volume render."""
    go = _import_plotly()
    return _make_figure(
        go.Volume(
            x=np.ravel(x),
            y=np.ravel(y),
            z=np.ravel(z),
            value=np.ravel(value),
            opacity=opacity,
            surface_count=surface_count,
        ),
        title,
    )


def plot_streamtube(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    title: str | None = None,
) -> go.Figure:
    """Interactive Plotly Streamtube over a 3D vector field (u, v, w)."""
    go = _import_plotly()
    return _make_figure(
        go.Streamtube(
            x=np.ravel(x),
            y=np.ravel(y),
            z=np.ravel(z),
            u=np.ravel(u),
            v=np.ravel(v),
            w=np.ravel(w),
        ),
        title,
    )


def export_png(fig: go.Figure, path: str | Path, width: int = 800, height: int = 600) -> None:
    """PNG export -- the primary, CI-checked, PR-facing deliverable. GitHub renders an inline
    image from a PR comment/body upload; it does not render an attached self-contained HTML file
    at all (see export_html's docstring), so this is the actual review-facing path, not HTML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, height=height)


def render_orbit_gif(
    fig: go.Figure,
    path: str | Path,
    n_frames: int = 12,
    radius: float = 1.8,
    z_eye: float = 1.25,
    fps: float = 8.0,
) -> None:
    """Orbit the camera around a 3D Plotly figure and assemble the frames into a GIF via Pillow
    (same "rotate rather than time-step" rationale as mx_viz.fields_3d.
    render_field_surface_orbit_gif -- these trace types have no time axis of their own to step
    over). Requires kaleido (like export_png) to rasterize each frame."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        fig.update_layout(
            scene_camera={
                "eye": {"x": radius * np.cos(theta), "y": radius * np.sin(theta), "z": z_eye}
            }
        )
        png_bytes = fig.to_image(format="png")
        images.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    duration_ms = round(1000 / fps)
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def export_html(fig: go.Figure, path: str | Path) -> None:
    """Export self-contained HTML -- a secondary, local-inspection-only artifact (GitHub doesn't
    render an attached HTML file in a PR at all; export_png above is the actual PR-facing
    deliverable). See _SELF_CONTAINED_CSP above for why a bare
    write_html(include_plotlyjs=True) isn't self-contained on its own."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True)
    html = path.read_text(encoding="utf-8")
    if "<head>" not in html:
        raise ValueError(f"expected a <head> tag in exported HTML, none found in {path}")
    html = html.replace("<head>", f"<head>\n    {_SELF_CONTAINED_CSP}", 1)
    path.write_text(html, encoding="utf-8")

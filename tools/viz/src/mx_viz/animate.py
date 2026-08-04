from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from PIL import Image

# GIF-only: MP4 export would need imageio-ffmpeg's bundled FFmpeg binaries, whose LGPL/GPL
# license text doesn't travel with the PyPI wheel in a form this Design's license-file-not-
# just-metadata check can verify (checked against imageio-ffmpeg's own repo/README -- neither
# documents which FFmpeg build/license applies or ships the corresponding license file). Per
# issue #62's own stated fallback, GIF-only is acceptable rather than shipping an unverified
# redistribution. Uses Pillow (already an indirect matplotlib dependency) rather than adding a
# new one -- see tools/viz/CLAUDE.md.


def _figure_to_image(fig: Figure) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    return Image.frombuffer("RGBA", (width, height), buf, "raw", "RGBA", 0, 1).convert("RGB")


def open_gif(frames: list[Figure], path: str | Path, fps: float = 10.0) -> None:
    """Assemble already-rendered matplotlib Figures into an animated GIF, in list order.

    Each frame must already have an Agg canvas (e.g. via FigureCanvasAgg(fig), as every
    mx_viz.fields figure-constructing function already does). See
    mx_viz.fields.render_field_frames for the field-comparison-over-time use this backs.
    """
    if not frames:
        raise ValueError("open_gif requires at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [_figure_to_image(fig) for fig in frames]
    duration_ms = round(1000 / fps)
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)

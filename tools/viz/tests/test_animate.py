from __future__ import annotations

import numpy as np
import pytest
from mx_viz.animate import open_gif
from mx_viz.fields import plot_field_frame


def test_open_gif_writes_nonempty_file(tmp_path):
    x = np.linspace(0, 1, 5)
    frames = [
        plot_field_frame(x, np.sin(x + phase), np.sin(x + phase) + 0.01, t_value=phase)
        for phase in (0.0, 0.5, 1.0)
    ]
    path = tmp_path / "frames.gif"

    open_gif(frames, path, fps=5)

    assert path.exists()
    assert path.stat().st_size > 0


def test_open_gif_rejects_empty_frame_list(tmp_path):
    with pytest.raises(ValueError):
        open_gif([], tmp_path / "empty.gif")


def test_open_gif_creates_parent_dirs(tmp_path):
    x = np.linspace(0, 1, 3)
    frames = [plot_field_frame(x, np.sin(x), np.sin(x))]
    path = tmp_path / "nested" / "dir" / "frames.gif"

    open_gif(frames, path)

    assert path.exists()

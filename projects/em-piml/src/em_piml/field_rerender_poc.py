from __future__ import annotations

from pathlib import Path

from mx_viz import io as viz_io
from mx_viz.animate import open_gif
from mx_viz.fields import plot_field_heatmap, render_field_frames

from em_piml.train import (
    evaluate_relative_l2_error,
    save_field_grid_artifact,
    train_cavity_baseline,
)

# Manual, one-off proof-of-concept for field-visualization Slice 5 (issue #64) -- not part of
# automated CI (a one-off proof that persist -> render works end-to-end on real data isn't a
# standing check CI should re-run on every PR; see the issue's Rev-C corrected justification).
# Run by hand: `uv run python -m em_piml.field_rerender_poc`
ARTIFACT_PATH = Path(".outputs/em-piml/baseline_cavity_field.npz")
HEATMAP_PATH = Path(".outputs/em-piml/baseline_cavity_heatmap.png")
GIF_PATH = Path(".outputs/em-piml/baseline_cavity_frames.gif")

# The baseline cavity mode (issue #2) -- chosen over a long-horizon-collapse or two-mode
# experiment specifically for being fast (~35s) and easy to visually sanity-check (a single clean
# standing wave, no collapse dynamics to interpret), matching the issue's own "or a simpler one
# chosen for being easy to visually sanity-check" option.
SEED = 0


def main() -> None:
    print(f"Training baseline cavity PINN (seed={SEED})...")
    model = train_cavity_baseline(seed=SEED)
    relative_l2 = evaluate_relative_l2_error(model)
    print(f"  relative L2 error (held-out): {relative_l2:.4f}")

    print(f"Persisting field artifact to {ARTIFACT_PATH}...")
    save_field_grid_artifact(model, str(ARTIFACT_PATH))

    print(f"Loading artifact back from disk ({ARTIFACT_PATH}, allow_pickle=False)...")
    data = viz_io.load_field_artifact(ARTIFACT_PATH)
    viz_io.validate_field_artifact(data)

    print(f"Rendering static heatmap comparison to {HEATMAP_PATH}...")
    fig = plot_field_heatmap(
        data["grid_x"],
        data["grid_t"],
        data["predicted"],
        data["true"],
        title="baseline cavity: persisted-artifact re-render",
    )
    fig.savefig(HEATMAP_PATH, dpi=150)

    print(f"Rendering per-frame animated comparison to {GIF_PATH}...")
    frames = render_field_frames(data["grid_x"], data["grid_t"], data["predicted"], data["true"])
    open_gif(frames, GIF_PATH, fps=10)

    print("Done:")
    for path in (ARTIFACT_PATH, HEATMAP_PATH, GIF_PATH):
        print(f"  {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

# tools/viz (mx-viz)

Repo-wide plotting toolkit for research experiments (`CONVENTIONS.md`:
matplotlib-backed). See `tools/README.md` for the top-level "what
exists" summary and CLI usage; this file covers `mx-viz`-specific
implementation conventions a future session working in this package
needs.

## matplotlib-2D vs. PyVista-3D split

- `mx_viz.fields`, `mx_viz.training`, `mx_viz.sweeps` — matplotlib
  (`Figure` + `FigureCanvasAgg`, never `pyplot`, so headless by
  construction). Base dependencies only (`matplotlib`, `numpy`); every
  workspace member that imports plain `mx-viz` gets these for free.
- `mx_viz.fields_3d` — PyVista. Behind the `mx-viz[3d]` extra
  (`pyvista`, `plotly`, `trame`/`trame-vtk`/`trame-vuetify`,
  `imageio`). Guards its own import (`_import_pyvista()` raises a
  clear `ImportError` pointing at the extra) so importing `mx_viz.
  fields_3d` itself never requires PyVista to be installed — only
  calling its functions does.
- `mx_viz.plotly_fields` (issue #63) — Plotly, same `[3d]` extra.

**Never add `mx-viz[3d]` as a default/non-optional dependency of any
workspace member.** `ci.yml` runs `uv sync --all-packages
--all-extras`, syncing every member's *default* deps — pulling in
PyVista/VTK (~150-250MB) as a default dependency of e.g. `em-piml`
would defeat the whole point of the extra being optional (issue #58's
CI-cost problem this extra exists to solve). Depend on bare `mx-viz`
unless the consumer specifically needs `mx_viz.fields_3d`/
`mx_viz.plotly_fields`, and even then, add `mx-viz[3d]` only to that
consumer's own `pyproject.toml`, never widen the base package's own
`dependencies`.

Install the extra: `uv sync --all-extras` (repo-wide) or `uv sync
--package mx-viz --extra 3d` (this package only).

## Headless rendering

PyVista/VTK's off-screen renderer needs a real OpenGL context.
- **In code**: always construct `pv.Plotter(off_screen=True)` — never
  the interactive default.
- **In CI**: `ubuntu-latest` has neither a GPU nor a software OpenGL
  driver by default; `ci.yml` installs `libgl1 libglx-mesa0
  libxrender1 libxext6 xvfb` and wraps both test steps in `xvfb-run -a`
  (issue #58 — the "no Xvfb/OSMesa build needed" claim from PyVista's
  own docs didn't hold up against this actual runner).
- **Locally on Windows** (this repo's primary dev machine per
  `CONVENTIONS.md`'s compute-assumption entry): PyVista renders
  off-screen fine without Xvfb (real Windows OpenGL), so `uv run
  pytest tools/viz` works directly — Xvfb is a Linux/CI-only need.

## `mx_viz.feed` — why a publishing concern lives in a plotting package

`feed.py` (issue #112) renders a project's `results.csv` as JSONL for the
Parallax cockpit. It plots nothing and imports neither matplotlib nor
numpy, so the placement is a judgement call worth recording rather than
re-litigating:

- This package already owns results **I/O** (`mx_viz.io` persists and
  loads sweep results) and already owns "render results for a consumer" —
  the consumer is usually a human looking at a PNG, and here it is a
  cockpit reading a feed. Same job, different renderer.
- The alternative was a fourth workspace package for ~40 lines, which
  root `CLAUDE.md`'s "no scaffolding beyond what the intent requires"
  argues against more strongly than this placement argues for a new home.

**Its record schema is not free to change on taste.** Parallax's
`parse_metrics` keeps only a record's *string* fields as dimensions and
drops numeric ones, and groups series by (metric, dimensions). So
emitting `seed` as a string scatters a variant's runs into one-point
series and destroys the spread a null result lives in; emitting `issue`
or `steps` as numbers deletes them from the cockpit entirely. Read
`feed.py`'s module docstring before touching which columns are stringified,
and note that `tools/viz/tests/test_feed.py` mirrors the consumer's
grouping precisely so a schema regression fails there rather than in a
chart nobody double-checks.

## Field-artifact / rendering pipeline

`em_piml.train.evaluate_field_grid` → `em_piml.train.
save_field_grid_artifact` → `mx_viz.io.save_field_artifact` (`.npz`,
`allow_pickle=False` enforced on load) → `mx_viz.io.
load_field_artifact` → `mx_viz.fields.render_field_frames` (2D,
per-t-step) or `mx_viz.fields_3d.plot_field_surface` (3D, one field
array per call — see its docstring for why three overlaid 3D surfaces
aren't attempted). `uv run mx-viz field <artifact>` validates/
summarizes an artifact from the CLI.

## GIF-only: no MP4/`open_movie` yet

Both `mx_viz.animate.open_gif` (2D, via Pillow — already an indirect
matplotlib dependency, no new one added) and `mx_viz.fields_3d.
render_field_surface_orbit_gif` (3D, via PyVista's own `Plotter.
open_gif`/`write_frame`, backed by plain `imageio`) ship GIF export
only. MP4 (`open_movie`) would need `imageio-ffmpeg`'s bundled FFmpeg
binaries; checked against imageio-ffmpeg's own repo/README (issue #62)
and neither documents which FFmpeg build/license applies or ships the
corresponding license file where this Design's license-file-not-just-
metadata check could verify it. Per issue #62's own stated fallback,
GIF-only ships now; MP4 is revisitable if a verifiably-licensed FFmpeg
source is found later.

## Self-contained HTML isn't automatic — verify, don't trust the docs

Both PyVista's `Plotter.export_html` and (per issue #63) Plotly's
`write_html(include_plotlyjs=True)` are documented as "self-contained,"
but neither claim was true by inspection alone:

- PyVista: `export_html` embeds the vtk.js viewer bundle inline for
  the actual scene, but the bundle's own runtime also injects `<link
  rel="icon">` tags pointing at a remote `kitware.github.io` favicon —
  verified empirically against real output, not documented anywhere
  in PyVista's docs. `mx_viz.fields_3d.export_field_surface_html`
  doesn't try to pattern-match and strip that one specific minified
  call site (fragile across vtk.js versions); it inserts a strict
  Content-Security-Policy meta tag instead, so the *browser* refuses
  any remote request the embedded bundle makes, known or not.
- Plotly: see `mx_viz.plotly_fields` (issue #63) for the equivalent
  verified-not-assumed check.

If you touch either export path, re-verify against the actual emitted
file (grep for `http(s)://` usage as an actual `src`/`href` sink, not
just any string occurrence — copyright/doc-link comments inside a
minified bundle are inert and will false-positive a naive text scan)
rather than trusting a library's own "self-contained" claim.

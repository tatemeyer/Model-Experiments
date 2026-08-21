# tools/

Internal tooling: deterministic scripts/CLIs that agents call instead of
re-doing the same work freeform each time (see `CLAUDE.md` principle 2).
Check here before writing a one-off script for something a tool already
does.

## Available tools

- **`datasets/`** — `mx-data`: repo-wide dataset registry, fetch, and
  checksum verification. Register a dataset once as a `.toml` file under
  `tools/datasets/registry/`, then any project can run:
  ```
  uv run mx-data list
  uv run mx-data fetch <name>
  uv run mx-data verify [name]
  ```
  Datasets land in `.data/<name>/` at the repo root (gitignored, shared
  across projects — never duplicate a dataset per-project). See
  `CONVENTIONS.md` for the "mx-data is the only sanctioned path" rule.

  A `kind = "generator"` entry's `command` runs as a plain subprocess —
  it only sees packages if the workspace venv already has them (`uv sync
  --all-packages` must have run, and the generator must be invoked via
  `uv run mx-data fetch ...` so PATH resolves into `.venv`). A generator
  needing e.g. `numpy` should rely on some workspace member already
  depending on it rather than adding deps to `mx-datasets` itself.

- **`viz/`** — `mx-viz`: a plotting library (matplotlib-backed, see
  `CONVENTIONS.md`) for visualizing research experiments and results —
  field comparisons, training loss curves, and multi-variant/multi-seed
  sweep comparisons. Add it as a workspace dependency
  (`[tool.uv.sources] mx-viz = { workspace = true }`, see
  `projects/em-piml/pyproject.toml`) and import directly:
  ```python
  from mx_viz.fields import plot_field_heatmap, plot_field_slice
  from mx_viz.training import plot_loss_curve
  from mx_viz.sweeps import plot_sweep_comparison
  from mx_viz.io import save_results, load_results
  ```
  `fields`/`training`/`sweeps` take plain numpy arrays / float sequences
  / dicts, not model objects — framework-agnostic on purpose (works for
  torch, JAX, or anything else; the caller does the "evaluate my model"
  step, e.g. `em_piml.train.evaluate_field_grid`/`evaluate_field_slice`).
  `save_results`/`load_results` persist a sweep's `{variant: {seed:
  value}}` results as JSON (schema: `{"metadata": {...}, "results":
  {...}}`), and `save_field_artifact`/`load_field_artifact` persist a
  target/predicted field evaluation as a plain-array `.npz` (schema:
  `x, t, grid_x, grid_t, true, predicted, schema_version`;
  `load_field_artifact` enforces `allow_pickle=False`) — both so results
  outlive the training process and can be re-plotted later. Write these
  to `.outputs/<project>/` at the repo root (gitignored, mirrors
  `.data/`'s pattern, never commit generated plots/results). Two CLI
  verbs exist for the two cases with a natural persisted artifact:
  ```
  uv run mx-viz sweep .outputs/em-piml/some_sweep.json --out sweep.png [--kind box|bar]
  uv run mx-viz field .outputs/em-piml/some_field.npz
  ```
  A third verb publishes rather than plots. `mx_viz.feed` projects a project's
  tidy long-format `results.csv` into the JSONL metrics feed `parallax.yaml`
  declares (issue #112) — same results, rendered for a machine consumer instead
  of for a reader:
  ```
  uv run mx-viz feed projects/jepa/results.csv --out projects/jepa/results.jsonl
  ```
  The `.jsonl` is **checked in** (unlike `.outputs/`) so the declared feed is
  backed on a fresh clone, and it is a *derived projection* — `results.csv`
  stays the record of truth, and each project's `test_results_feed.py` fails CI
  if the two drift. Re-run this after appending rows to a `results.csv`.
  The record schema is dictated by the consumer's semantics, not chosen: see
  `feed.py`'s module docstring before changing which columns are emitted as
  strings, because that choice is what decides whether seeds group into one
  series with spread or scatter into one-point series.
  `em_piml.train.save_field_grid_artifact(model, path)` wraps
  `evaluate_field_grid` to produce a field artifact. Loss-curve plots
  stay library-only (no loss-history checkpointing exists in this repo
  yet) — call `plot_loss_curve` directly from a training script or
  research session.

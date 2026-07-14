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

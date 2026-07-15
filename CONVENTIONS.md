# Conventions

A living record of repo-wide decisions. Entries are dated; when a
convention changes, add a new entry rather than silently editing history
— the point is to track *why* the current best practice was adopted, so
it can be revisited when a better one emerges (see `CLAUDE.md` principle
3: conventions should grow with SOTA, not calcify).

## 2026-07-14 — Python dependency management: uv workspace

The repo root `pyproject.toml` is a virtual `uv` workspace (no
`[project]` table of its own). Every tool under `tools/` and every
project under `projects/` is a workspace member with its own
`pyproject.toml`, sharing one `uv.lock` at the root. Run `uv sync
--all-packages` to install everything; `uv run <cmd>` to execute inside
the workspace env.

Why: `uv` is currently the fastest and most widely adopted Python
dependency manager, and its workspace feature is built for exactly this
shape — one repo, many independently-versioned Python packages.

## 2026-07-14 — Linting: ruff

`ruff check .` (config lives in the root `pyproject.toml`,
`line-length = 100`). No separate formatter/linter stack — ruff covers
both lint rules and (via `ruff format`, not yet wired into CI) formatting.

## 2026-07-14 — Testing: pytest, colocated

Tests live next to the code they test (`tools/<name>/tests/`,
`projects/<name>/tests/`), not in a top-level `tests/` tree. `uv run
pytest` from the repo root discovers all of them (see `testpaths` in the
root `pyproject.toml`).

## 2026-07-14 — Datasets: mx-data is the only sanctioned path

Any dataset — downloaded or simulator-generated — must be registered as
a `.toml` entry in `tools/datasets/registry/` and fetched via `mx-data
fetch <name>`. Do not hand-roll a `curl`/`wget`/download script inside a
project; add a registry entry instead. This keeps data fetching
reusable across projects, checksummed, and out of git (see
`tools/README.md`).

## 2026-07-14 — Compute assumption: modest, mostly free

Default assumption for anything trained/simulated in this repo: CPU
primarily, with an optional single consumer GPU (currently a GTX 1660
Ti — Turing architecture, CUDA-capable, no tensor cores) and free-tier
cloud only (Colab/Kaggle-class, no paid rented compute). Don't default
to multi-GPU, large-batch, or paid-cloud-only designs; note explicitly
in a project's `CLAUDE.md` if it needs more than this.

## 2026-07-14 — ML framework default: PyTorch

Unless a project's issue says otherwise, default to PyTorch. Given the
compute assumption above (CPU + a non-tensor-core consumer GPU + free
notebook tiers), PyTorch's ecosystem maturity and lower-friction CUDA/CPU
path outweigh JAX's functional-transform advantages for now. Revisit
per-project if a project's research question specifically benefits from
JAX (e.g. needing to differentiate through a JAX-backed simulator).

## 2026-07-14 — Branch naming: prefix by kind

`main` is trunk. Everything else is prefixed by kind:
`feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`,
`experiment/<slug>` (research spikes that may never merge). The GitHub
rulesets in `.github/SETUP.md` target these patterns directly.

## 2026-07-14 — Third-party GitHub Actions are pinned by commit SHA

Not a moving version tag (`@v4`) — a pinned commit with a version
comment (`@<sha> # v7.0.0`), kept current via `.github/dependabot.yml`'s
`github-actions` ecosystem. GitHub-authored actions (`actions/*`) get
the same treatment for consistency. Prompted by the March 2026
`trivy-action` incident, where a compromised maintainer force-pushed
version tags to redirect them at malicious commits — a SHA pin can't be
silently repointed that way.

## 2026-07-14 — Documentation is agent-first, not human-first

Don't write doc-comments, docstrings, or comments explaining what code
does in areas a human doesn't plan to read or edit by hand — write only
what a future agent needs to avoid re-deriving context (non-obvious
invariants, why a workaround exists), and no more. Verbose human-oriented
prose belongs only in places a human actually maintains directly.

## 2026-07-15 — Small, well-vetted optimizer packages beyond `torch.optim` are acceptable when literature-justified

Default is still `torch.optim` only — don't add an optimizer dependency
on a whim. But when a specific research result names an optimizer not in
`torch.optim` (e.g. Khodakarami et al. on SOAP/SS-Broyden resolving PINN
spectral-bias instability, `projects/em-piml/CLAUDE.md` issue #11) and a
small, actively-maintained, narrowly-scoped PyPI package implements it
faithfully (checked: recent releases, real usage/stars, CI, doesn't drag
in unrelated heavy deps), adopting it is preferable to hand-rolling the
algorithm or vendoring an un-packaged reference implementation. Document
the specific tradeoff (what, why trusted, what it costs) in the
project's own `CLAUDE.md` each time — this convention doesn't pre-approve
any specific package going forward, it just establishes the bar is
"justified by literature + vetted for trust," not "never a new dep."

## 2026-07-15 — Testing: fast by default, `slow` marker for model training

Full `uv run pytest` runtime kept growing (~90s -> ~2:06 -> ~2:50 across
PRs #5/#7/#9) as `projects/em-piml/tests/` accumulated tests that each
actually train a PINN (35-100s+ apiece). Default `uv run pytest` (no
args) must stay a fast, routine command for iterating on non-training
code — it now excludes anything marked `slow` via `addopts = "-m 'not
slow'"` in the root `pyproject.toml`, and completes in well under 30
seconds.

Mark a test `@pytest.mark.slow` (registered in `[tool.pytest.ini_options]
markers`) if it actually trains/fits a model (as opposed to pure-Python
logic, CLI plumbing, or fixture-based tests like
`tools/datasets/tests/test_cli.py`, which stay fast and unmarked). Run
the full suite, slow tests included, with `uv run pytest -m slow`
(slow-only) or `uv run pytest -o addopts=""` (everything, overriding the
default exclusion). CI (`.github/workflows/ci.yml`) runs both: a
"Test (fast)" step (the default, unmarked command) and a "Test (slow)"
step (`uv run pytest -m slow`) — so slow tests keep running in CI even
though they're excluded from the local default.

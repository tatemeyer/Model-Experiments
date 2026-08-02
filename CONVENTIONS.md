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
spectral-bias instability, `projects/em-piml/experiments/num-bands-gap/
011-soap-optimizer.md`) and a small, actively-maintained, narrowly-scoped
PyPI package implements it faithfully (checked: recent releases, real
usage/stars, CI, doesn't drag in unrelated heavy deps), adopting it is
preferable to hand-rolling the algorithm or vendoring an un-packaged
reference implementation. Document the specific tradeoff (what, why
trusted, what it costs) in the project's own experiment write-up each
time (see the "Project experiment logs" convention below for where that
lives) — this convention doesn't pre-approve any specific package going
forward, it just establishes the bar is "justified by literature +
vetted for trust," not "never a new dep."

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

## 2026-07-27 — Project experiment logs: forest not monolith, plus structured data

`projects/em-piml/CLAUDE.md` grew append-only from 74 to 1394 lines
across 13 days as each new experiment appended a full write-up section
and never removed one — every session working in that project loaded
all of it regardless of relevance, and the growth was accelerating with
no natural ceiling. Once a project accumulates more than a handful of
experiments, apply the same "forest not monolith" principle (root
`CLAUDE.md` principle 1) one level deeper, inside the project itself:

- **`projects/<name>/experiments/`** — one Markdown file per experiment
  (`NNN-slug.md`, issue number + short slug), containing that
  experiment's full write-up (mechanism, results, diagnosis, leads).
  Group experiments that target the same underlying question into a
  subfolder (`experiments/<thread>/`) once a second one follows up on
  the first; a standalone question stays a top-level file. See
  `projects/em-piml/experiments/TEMPLATE.md` for the expected shape and
  placement rule.
- **`projects/<name>/results.csv`** — every experiment's numeric
  results in tidy long format (one row per issue/variant/seed/metric
  datapoint, a flexible JSON `params` column for whatever was swept).
  Machine-readable by design — this is the data contract any
  visualization/analysis tooling should read from, not a Markdown table
  scraped out of prose.
- **`projects/<name>/LITERATURE.md`** — a paper registry (one row per
  paper cited or tested, with a verdict), so "have we already tried
  this" is answerable by scanning a table instead of grepping every
  experiment file.
- The project's own `CLAUDE.md` shrinks to stable spec (the problem,
  model, verification approach, data source — things every session
  needs) plus an experiment index (one line per experiment, linking
  out) and an actively-maintained "open leads" section — never a
  per-experiment write-up appended in place.

Why: the growth pattern is structural (an issue closes, its findings get
recorded, the project accumulates them indefinitely) and will recur in
every project that runs more than a few experiments, not just em-piml —
adopt this layout from the start rather than waiting for the same
1000+-line file to happen again.

## 2026-07-27 — Test filenames are unique repo-wide, not just per-directory

Test files across the whole workspace share one basename namespace: no
`tests/` directory here has an `__init__.py`, so pytest's rootless
import mode requires every test *filename* to be unique repo-wide, not
just per-directory (`tools/viz/tests/test_cli.py` collided with
`tools/datasets/tests/test_cli.py` and had to be renamed
`test_sweep_cli.py`). Check `uv run pytest --collect-only -q` (or just
grep existing `tests/` dirs) for a name collision before adding a new
`test_<name>.py`, rather than adding `__init__.py` files repo-wide to
work around it.

## 2026-07-27 — Plotting: matplotlib (`tools/viz`, `mx-viz`)

As research output grew past what fits in a `print()`-and-hand-transcribe
loop (this project's many pointwise-check markdown tables, now living
under `projects/em-piml/experiments/` per the entry above), a shared
plotting toolkit (`tools/viz`) was added. matplotlib is the default
plotting library for this repo — no plotting library existed before
this entry, so this is a first adoption, not a switch.

Why matplotlib over plotly/bokeh/altair: this repo reports findings as
static content embedded in PR bodies and experiment-log markdown, not a
hosted/interactive dashboard, so static PNG output is the better fit;
it has a trivial headless-CI story (no browser/JS runtime — build
`Figure`/`Axes` objects directly via the object-oriented API rather
than importing `pyplot`, which avoids global backend state and works
identically in an interactive session or CI test collection); and it
carries no GPU/CUDA-adjacent dependency risk. `mx-viz`'s own plotting
functions are framework-agnostic (numpy arrays / float sequences /
dicts in, not model objects) so they aren't tied to PyTorch even though
every current caller is.

Note: `mx_viz.io`'s JSON schema (`{variant: {seed: value}}`) and the
"Project experiment logs" entry above's `results.csv` (tidy long format,
one row per issue/variant/seed/metric) are two independent data
contracts that currently don't read from each other — a natural
follow-up is teaching `mx-viz` to plot directly from `results.csv`
rows instead of requiring its own JSON export per sweep.

## 2026-07-30 — Testing: `gpu` marker for hardware-gated tests

The `device-abstraction` Arc (`docs/design/specs/2026-07-28-em-piml-modernization/`)
introduces a second, orthogonal test-exclusion axis alongside the
existing `slow` marker (2026-07-15 entry, above): hardware availability,
not runtime. A test that needs actual GPU hardware isn't slow in the
sense that entry means (it may run in well under a second on hardware
that has a GPU) — it's simply impossible to run correctly on
`ubuntu-latest`, which has none. Overloading `slow` for this would let a
`gpu`-marked test get selected by `uv run pytest -m slow` on a GPU-less
runner and fail there, not skip.

Mark a test `@pytest.mark.gpu` (registered in `[tool.pytest.ini_options]
markers`, root `pyproject.toml`) if it requires actual GPU hardware to
run correctly — as opposed to a test that merely exercises
device-selection *logic* (e.g. `projects/em-piml/tests/test_device_selection.py`,
which stays unmarked and fast, since it mocks
`torch.cuda.is_available()` rather than needing a real GPU). Root
`pyproject.toml`'s `addopts` is `-m 'not slow and not gpu'` — both
markers are excluded from the default `uv run pytest` command.
`.github/workflows/ci.yml`'s "Test (slow)" step runs `uv run pytest -m
'slow and not gpu'`, so GPU-gated tests stay excluded from CI even
though `slow` ones run there. Run GPU-gated tests explicitly, on a
machine that actually has one, with `uv run pytest -m gpu` (GPU tests
only) or `uv run pytest -o addopts=""` (everything, overriding both
exclusions) — **this supersedes the 2026-07-15 entry's documented
`uv run pytest -m slow` and `uv run pytest -o addopts=""` commands**:
the former now needs `-m 'slow and not gpu'` to avoid also selecting
GPU-gated tests on a machine without a GPU (a bare `-m` on the command
line overrides `addopts` rather than ANDing with it); the latter is
unchanged, since it already runs everything.

## 2026-07-30 — Multi-project gitops: branch naming, labels, worktrees

With a second project (`projects/jepa`) now onboarded alongside
`em-piml`, formalizing patterns that were already emerging informally
(confirmed by `git branch -a` history, e.g. `feat/em-piml-ntk-reweighted-
long-horizon`, `docs/em-piml-arc-charters-tier1`) rather than inventing
new ones:

- **Branch naming**: `<kind>/<project>-<slug>` (e.g.
  `feat/jepa-research-scaffold`) when a branch is scoped to one
  `projects/<name>/` or `tools/<name>/`; plain `<kind>/<slug>` (no
  project segment) when the change is cross-cutting (root config, CI, a
  shared tool, this entry itself). No ruleset change needed — the
  `feature-branches` ruleset already globs `feat/**` etc., which matches
  the project-scoped form for free. `.github/SETUP.md`'s "Branch naming
  convention" section is updated to state this explicitly.
- **Project labels**: `project:<name>` (`project:em-piml`,
  `project:jepa`, `project:shared` for cross-cutting work) exist as
  real GitHub labels, applied manually by whoever files the Issue/PR —
  not by an auto-labeler Action. Two projects don't yet justify building
  path-mapping automation; add it only if manual mislabeling actually
  becomes recurring, the same trigger-based-adoption logic as the
  "Project experiment logs" entry above.
- **Worktrees**: one mechanism, not two. A session working on a given
  project uses Claude Code's own worktree isolation (already gitignored
  via `.claude/`, see PR #14) and checks out/creates that project's
  branch inside it — project scoping lives in the branch name, worktree
  scoping lives in the session-isolation mechanism, and the two are
  orthogonal by design. Don't build a second, project-keyed worktree
  layout. Caveat worth remembering: each worktree needs its own `uv sync
  --all-packages` (a separate `.venv`, not shared across worktrees).
- ~~**CI path-scoping**~~ — filed as issue #71, superseded by the
  2026-07-31 entry below.

## 2026-07-31 — CI: slow-test step scoped to touched projects/tools

Implements issue #71. `.github/workflows/ci.yml`'s `verify` job stays one
required check (job name unchanged, per `.github/SETUP.md`'s Rulesets
section) and Lint/`Test (fast)` stay workspace-wide — only `Test (slow)`
is scoped. A new `Determine slow-test scope` step
(`.github/scripts/slow_test_scope.sh`) diffs the PR's base/head SHAs (or
`before`/`after` on a push to `main`), and:

- If every changed file lives under `projects/<name>/` or
  `tools/<name>/`, runs `uv run pytest -m 'slow and not gpu'` scoped to
  just those top-level directories (e.g. a `projects/em-piml/**`-only PR
  runs only `em-piml`'s slow tests).
- Otherwise — any root-level/shared file changed (`pyproject.toml`,
  `uv.lock`, `.github/workflows/ci.yml` itself, docs, etc.), or the
  base/head SHA can't be resolved — falls back to running every
  project's slow tests, the safe default.

Deliberately *not* an `on.pull_request.paths` trigger filter: a required
check that never runs for a non-matching PR blocks merge forever ("waiting
for status"). The scoping happens inside the always-running `verify` job
instead. `actions/checkout` now uses `fetch-depth: 0` so both diff
endpoints are resolvable locally — cheap at this repo's current size.

Verification note: GitHub resolves a `pull_request`-triggered workflow's
own file changes against that same PR's subsequent runs, but a *second*,
independently-branched PR only sees a workflow change once it's on
`main` — so the "scoped to one project" branch of this logic can't be
exercised live by a PR that also modifies `ci.yml` itself (this one
necessarily does, which is exactly why it takes the fallback path). Verified via local dry-run of the script against real and synthetic
file-change lists (all three branches: single-project, multi-project, and
root-file-triggers-fallback) plus a direct `uv run pytest --collect-only`
scoped to `projects/em-piml` confirming only its 19 slow tests are
selected. Live end-to-end confirmation of the scoped path is the first
project-only PR opened after this merges.

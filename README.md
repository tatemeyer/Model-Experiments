# Model-Experiments

ML research experiments, organized one subtree per project under
`projects/`: currently physics-informed ML for electromagnetics
(`projects/em-piml/`) and Joint-Embedding Predictive Architecture research
(`projects/jepa/`). Each project has its own scoped `CLAUDE.md`,
`pyproject.toml`, and research arc.

Development is driven by **Bitter Lesson Engineering (BLE)** / **Intent
Engineering**: intent is filed as a GitHub Issue (desired end state + how
to verify it, not implementation steps), Claude Code implements against
that intent and opens a PR, and CI is the source of truth for "done." See
[`CLAUDE.md`](./CLAUDE.md) for the full loop and the autonomy-label rules
Claude Code sessions follow in this repo, and
[`CONVENTIONS.md`](./CONVENTIONS.md) for current technical decisions
(package manager, linter, ML framework, compute assumptions, etc).

## Projects

- [`projects/em-piml/`](./projects/em-piml/) — physics-informed ML for
  electromagnetics. A verified baseline PINN solving the 1D
  perfect-electric-conductor cavity wave equation against a closed-form
  solution, building toward the project's real research interest:
  tokenization/embedding schemes for PIML.
- [`projects/jepa/`](./projects/jepa/) — Joint-Embedding Predictive
  Architecture research (I-JEPA/V-JEPA family). Studies what prevents
  representation collapse at toy scale, using a procedurally generated
  bouncing-ball environment with exact closed-form ground truth for
  linear-probe evaluation.
- [`projects/nemotron-asr/`](./projects/nemotron-asr/) — CPU-only speech
  recognition with NVIDIA Nemotron 3.5 ASR (0.6B cache-aware
  FastConformer-RNNT), evaluated as a dictation front-end for `TTUI` and
  `Parallax`. An integration study rather than a training study: measures
  real-time factor and WER on the target CPU instead of trusting vendor
  GPU benchmarks.

## Layout

- `projects/<name>/` — one subtree per experiment, each with its own
  scoped `CLAUDE.md`.
- `tools/` — internal automation shared across projects: `mx-data` (dataset
  registry/fetch/verify) and `mx-viz` (plotting/rendering for experiment
  results). See [`tools/README.md`](./tools/README.md).
- Single `uv` workspace rooted at this `pyproject.toml` — `uv sync
  --all-packages` installs everything, `uv run pytest` runs every
  project's tests.

## Contributing an experiment

1. File an Issue using the **Intent** template — describe the desired
   end state and how it can be verified, not the implementation steps.
2. Label it with an autonomy level (`autonomy:safe`, `autonomy:review`,
   or `autonomy:human`).
3. Claude Code (or a human) implements it and opens a PR.
4. CI verifies it. `autonomy:safe` PRs merge themselves on green CI;
   everything else waits for review.

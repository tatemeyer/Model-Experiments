---
title: "Arc Charter — device-abstraction"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-30-device-abstraction
slice: 2026-07-30-charter
revision: B
status: proposed
date: 2026-07-30
related-arcs: [jax-migration, cloud-compute-ops]
supersedes: null
superseded-by: null
---

# Arc Charter — device-abstraction

See `docs/design/README.md` for what Design/Arc/Slice, revisions, gates, and
Change Orders mean, and the parent Design Charter
(`docs/design/specs/2026-07-28-em-piml-modernization/2026-07-28-foundation/2026-07-28-charter/README.md`,
currently Rev-F) for this Design's overall scope, non-negotiable
constraints, and cross-Arc dependencies. This document assumes both and
doesn't redefine them — it scopes `device-abstraction` specifically.

## 1. Purpose & why now

A repo-wide grep found no `.cuda()` call, no CLI/env flag for device
selection, and no threaded `device=` parameter anywhere in
`projects/em-piml/src/em_piml/` today. **Correcting Rev-A's overstatement**
(Technical feasibility gate finding, see §12): two call sites,
`model.py:73` and `model.py:160`, already use the correct
`device=x.device` inheritance idiom — the gap is that nothing upstream of
those two sites ever threads a *chosen* device in, not that the idiom is
entirely absent. This remains the one clear blocking prerequisite named in
the parent Design Charter's §6 dependency table: both `jax-migration` and
`cloud-compute-ops` are individually incomplete without it. Confirmed as
its own standalone Arc (not folded into `foundation`) as part of the
usage-constraint/dev-time sequencing breakdown adopted 2026-07-30 (parent
Charter §6/§11).

## 2. Relationship to the Design Charter

This Arc Charter inherits every constraint in the parent Design Charter,
especially:

- §5's compute-stays-CPU-primarily/free-tier-cloud-only constraint.
- §5's CI-stays-CPU-only-**and-unchanged**-unless-explicitly-justified
  constraint. **Rev-A dropped the "and unchanged" half** (Technical
  feasibility gate finding, see §12) — this Arc does propose and justify
  two changes to shared CI/test-configuration surface (§5, §6 below); that
  is the "explicitly justified" exception the parent constraint requires,
  not a silent violation of it.
- §7's CI-implications split: CPU-path correctness (defaults to CPU,
  doesn't error absent an accelerator) is CI-verifiable and should be;
  actual GPU/TPU-selection correctness requires hardware CI doesn't have,
  and should be verified manually/interactively with the result recorded
  in an experiment write-up.
- §6's sequencing caveat: this Arc's natural PyTorch-idiom implementation
  (`.to(device)` calls, `device=` threaded through tensor-construction call
  sites) does not structurally carry into JAX's functional device-placement
  model — treat it as an interim measure for the current codebase, not code
  `jax-migration` inherits wholesale.
- **§5's `autonomy:review` carve-out for any PR touching a CI workflow file
  or GitHub Actions secret/Environment** (Security gate finding, see §12) —
  omitted from Rev-A's inheritance list entirely, and directly relevant
  here: §6's Slice 1 modifies `.github/workflows/ci.yml`.

This document adds only what's specific to `device-abstraction`; it doesn't
re-litigate the parent's constraints.

## 3. Scope

- Add real device selection to `projects/em-piml/src/em_piml/`'s training
  and evaluation code. **The primary, CI-testable mechanism is a function
  keyword** — `device: torch.device | str | None = None` threaded through
  training/evaluation call sites — **not a CLI flag** (Goal-delivery gate
  finding, see §12): `projects/em-piml/` has no `argparse` anywhere and no
  `[project.scripts]` entry today (unlike `tools/datasets`/`tools/viz`,
  both of which do), every existing test imports library functions
  directly, and this Arc's own cited precedent — the FP64 experiment
  (`train_fourier_cavity_lbfgs_fp64`, `evaluate_relative_l2_error(...,
  dtype=torch.float64)`) — is itself a function-keyword pattern, not a CLI
  flag. A CLI/env-var resolver may be added as a thin wrapper over the
  keyword, but the keyword is the mechanism Slice 1's tests actually
  exercise.
- **Package home: project-scoped** (`projects/em-piml/src/em_piml/`), not a
  `tools/<name>` package — device selection here is core training-loop
  logic specific to em-piml's own experiments, not a reusable utility other
  projects would import.
- Implementation idiom: PyTorch's (`device=` threaded through
  tensor-construction call sites, extending the `device=x.device` pattern
  already present at `model.py:73`/`:160`) — the same shape this repo
  already used for `dtype` in the FP64 precision experiment.
- **Out of scope for this Arc:**
  - Any JAX-idiom device placement (`jax.device_put`) — that belongs to
    `jax-migration`'s own Arc Charter once it exists.
  - **TPU/XLA backends** (Goal-delivery gate finding, see §12): this Arc
    commits to `torch.cuda.is_available()`/`torch.device`, which covers
    CUDA only. PyTorch TPU access requires the separate `torch_xla`
    package. `cloud-compute-ops`'s Arc Charter — which names Kaggle TPU
    (~20 TPU-hrs/week) and Google TRC among its ranked providers — must
    not assume this Arc's device abstraction unblocks TPU workloads; it
    unblocks CUDA (GPU) only.

## 4. Reconciliation with existing `CONVENTIONS.md` entries

**Omitted entirely from Rev-A** (Convention-alignment gate finding, see
§12) — the same class of gap the parent Design Charter's own
Convention-alignment gate caught for its optimizer-default entry.

**Testing: fast by default, `slow` marker for model training**
(`CONVENTIONS.md:104-123`, 2026-07-15): default `uv run pytest` excludes
anything marked `slow` via `addopts = "-m 'not slow'"` in root
`pyproject.toml:16`; CI runs both a fast step (default) and a slow step
(`uv run pytest -m slow`). This entry models a single exclusion axis. This
Arc introduces a second, orthogonal one — hardware availability, not
runtime — and needs its own marker rather than overloading `slow` (parent
§7/§17 already required this in general terms; this section makes the
specific mechanism explicit). Once §6's Slice 1 lands (which registers the
marker and edits `addopts`/`ci.yml`, see §5), record a new dated
`CONVENTIONS.md` entry describing the `gpu` marker's exclusion semantics
alongside the existing `slow` one — not as a side effect of this Arc
Charter reaching Rev-0, but as part of Slice 1's own PR, since the marker
doesn't mean what this Charter claims until that entry (and the code it
describes) exists.

**Compute assumption** (`CONVENTIONS.md:44-51`, 2026-07-14): "CPU
primarily, with an optional single consumer GPU **(currently a GTX 1660
Ti — Turing architecture, CUDA-capable, no tensor cores)** ... and
free-tier cloud only." **Rev-A's §6 paraphrase dropped the parenthetical**
(Cost/compute-budget gate finding, see §12) — the specific GPU named here
is the fact that actually decides this Arc's feasibility and performance
expectations (§5, §6). This entry doesn't need a new dated entry of its
own; it needs to be read in full, which Rev-B now does.

## 5. Non-negotiable constraints

- **Default behavior vs. explicit request are different guarantees, not
  one** (Goal-delivery gate finding, see §12):
  - The *default* (no device specified) resolves to CPU and never errors,
    regardless of what hardware is present.
  - An *explicit* device request (`device="cuda"`, or the CLI/env
    equivalent if built) that cannot be honored **must raise a clear,
    actionable error — never silently fall back to CPU**. Rev-A's single
    "must not error/degrade when no accelerator is present" bullet
    conflated these two cases; on the owner's own Windows environment (see
    §6's Slice 3 discussion), an explicit `--device cuda` request currently
    fails `torch.cuda.is_available()` for a packaging reason, not a
    hardware-absence reason — silently downgrading that to a CPU run and
    recording it as a GPU result would be a research-provenance failure of
    the kind parent §7 already flags for the JAX port.
  - The resolved device is printed/logged and recorded in
    `projects/em-piml/results.csv`'s `params` column (per
    `CONVENTIONS.md:125-164`'s data-contract convention) for any run that
    isn't the plain CPU default.
- **The CPU default path must stay bit-for-bit identical to pre-Arc
  behavior** (Goal-delivery gate finding, see §12), matching this repo's
  own existing precedent at `train.py:79-88` (`generator=None`/
  `dtype=torch.float32` "kept as the default so existing callers/tests are
  bit-for-bit unaffected"). §6's Slice 2 verification is a same-seed,
  bit-identical-output comparison against pre-Arc `main` for the CPU
  default path specifically — not a loose threshold assertion (the
  existing suite's `relative_l2 < 0.1`-style checks would not catch a
  shifted RNG stream).
- **`torch.Generator` is device-bound** (Technical feasibility gate
  finding, see §12): `train.py:357` and `:434` construct
  `torch.Generator().manual_seed(points_seed)` — a CPU generator — passed
  into `torch.rand(..., generator=generator, ...)` calls
  (`train.py:89-94`). Once a non-CPU device is actually selected, these
  sites need `torch.Generator(device=...)`, which draws a **different**
  random sequence for the same seed. This only affects non-default
  (non-CPU) runs — it does not conflict with the bit-exactness bullet
  above, which is scoped to the CPU default path — but it means seed
  reproducibility is not guaranteed *across* devices, only *within* one;
  §6's Slice 2 write-up must state this explicitly rather than implying
  full seed portability.
- **`gpu`-marked tests require two specific, named edits, not a marker
  declaration alone** (Convention-alignment gate finding, see §12): root
  `pyproject.toml:13-16` currently registers only `slow` and sets
  `addopts = "-m 'not slow'"`. A `@pytest.mark.gpu` marker with no further
  change would still run under the default `-m 'not slow'` selection (and
  fail on a GPU-less CI runner) or under `-m slow` if also marked `slow`.
  Making the exclusion real requires: (a) registering `gpu` in
  `pyproject.toml`'s `markers` list, (b) changing `addopts` to `-m 'not
  slow and not gpu'`, and (c) changing `.github/workflows/ci.yml`'s slow
  step to `-m 'slow and not gpu'`. These three edits are Slice 1's own
  scope (§6), not deferred to Slice 3.
- Any test requiring actual GPU hardware carries `@pytest.mark.gpu`, lives
  colocated in `projects/em-piml/tests/` per `CONVENTIONS.md`'s testing
  entry (no top-level `tests/` tree), and is excluded from the default
  fast/slow CI split by the mechanism above.
- **This Arc's third-party-dependency footprint is a packaging change, not
  a package addition** (Technical feasibility / License-compliance gate
  findings, see §12): `uv.lock:1094-1111` locks `torch` 2.13.0 from
  `pypi.org`, whose CUDA-runtime dependencies (`cuda-bindings`,
  `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton`, etc.) all carry
  `marker = "sys_platform == 'linux'"` — the Windows wheel ships with no
  CUDA runtime at all, a gap `projects/em-piml/CLAUDE.md`'s own "Known
  deferred items" section already records as unresolved. Making CUDA
  actually usable on the owner's Windows environment requires switching
  `torch`'s source to the PyTorch CUDA wheel index
  (`[[tool.uv.index]]`/`[tool.uv.sources]`), not adding a new PyPI package.
  If a Slice does this: (a) the index must be pinned so only `torch`
  resolves from it (`explicit = true` on the `[[tool.uv.index]]` entry,
  per this repo's existing supply-chain posture —
  `CONVENTIONS.md:69-78`'s SHA-pinned-Actions entry, motivated by the
  `trivy-action` incident — extended here to package indices); (b) the
  NVIDIA CUDA EULA / cuDNN SLA governing the pulled redistributables gets
  the same license-file-not-just-metadata check the parent Design Charter
  already applies to every other dependency (parent §5); (c) the resulting
  `uv.lock` diff carries `autonomy:review`, same as a CI-workflow diff.
- **A diff to root `pyproject.toml`'s `[tool.pytest.ini_options]`
  (markers or `addopts`) is treated identically to a CI-workflow-file diff
  for autonomy-labeling purposes** (Security gate finding, see §12) — it
  silently changes what auto-merge's green check *means* even though it
  isn't literally a workflow file, which is a gap in the parent Design
  Charter's carve-out worth raising there as a future Change Order once it
  freezes, not just noting here.
- **The device string, if exposed via CLI/env, is validated against an
  explicit allowlist** (`cpu`, `cuda`, `cuda:<n>`) and rejected loudly
  otherwise (Security gate finding, see §12); CLI flag takes precedence
  over env var if both exist; the resolver ignores the env var under
  `pytest` unless a test sets it explicitly, so an Environment-level env
  var can't silently change what the CPU-path tests are testing.

## 6. Named Slices and sequencing

| Slice | Scope | Verifiable by |
|---|---|---|
| `device-selection-module` | Device-resolution helper (function-keyword primary, thin CLI/env wrapper); default-vs-explicit-request error semantics (§5); device-string validation and CLI-over-env precedence (§5); registers the `gpu` pytest marker and edits root `pyproject.toml`'s `addopts` and `.github/workflows/ci.yml`'s slow step (§5) — **this Slice's PR modifies `.github/workflows/ci.yml` and root `pyproject.toml`'s test config, so it carries `autonomy:review`, not `autonomy:safe`** (§2, §5) | CI — no hardware needed |
| `training-loop-threading` | Thread the resolved device through **all** tensor-construction call sites, not just `train.py` (1294 lines): also `physics.py`, `dielectric.py`, `embeddings.py`, `model.py` (extending the existing `device=x.device` pattern at `model.py:73`/`:160`), and the `evaluate_relative_l2_error`/`evaluate_field_grid`/`evaluate_field_slice` functions (`train.py:1179`/`:1198`/`:1222`) every test calls. Covers 19 public `train_*` entry points and 9 private `_train_*` inner loops across 18 documented experiments — **corrected from Rev-A's "~10 experiments," which understated this by roughly 2x** (Technical feasibility gate finding, see §12). Handles the `torch.Generator` device-binding at `train.py:357`/`:434` per §5. | CI (CPU-path only, bit-identical-output verification per §5) + code review, per-variant |
| `gpu-selection-verification` | Manual, interactive confirmation that device selection actually places tensors on an accelerator when requested; records wall-clock CPU-vs-accelerator timing for at least one FP32 long-horizon variant and one FP64 variant (per §5's compute-assumption reconciliation — the owner's named GPU, a GTX 1660 Ti, runs FP64 at roughly 1/32 of FP32 throughput, so "GPU is slower for this workload" is a real possible, acceptable conclusion, not just "GPU works"); recorded in an experiment write-up. **Actual blocker is packaging (§5's wheel-index bullet), not hardware access** — corrected from Rev-A, which misdiagnosed this as possibly needing to wait on `cloud-compute-ops` (Cost/compute-budget gate finding, see §12); this Slice is expected to complete locally against the owner's existing GTX 1660 Ti with no `cloud-compute-ops` dependency, preserving this Arc's Tier-1 (no-new-usage-constraint) status. | Manual/interactive only — requires actual hardware; gated by the `gpu` marker from Slice 1 |

## 7. Cross-cutting gaps / risks specific to this Arc

- **`torch.Generator` device-binding** (§5, §6) is the one place Slice 2
  is not purely mechanical, despite Rev-A's "mechanical" framing — flagged
  here as the specific site to watch, not a generic caveat.
- Threading device selection through the now-corrected, wider module set
  (§6) is still mechanical in aggregate but touches every training-loop
  path — real risk of a silent regression in one variant's default
  behavior if verified only in aggregate rather than per-variant against
  the bit-exactness bar in §5.

## 8. Relationship to the Issue/PR loop

Each Slice above becomes its own Intent Issue + PR once this Arc Charter
reaches Rev-0, per the parent Design Charter's §8 / `docs/design/README.md`'s
process. **`device-selection-module`'s PR specifically modifies
`.github/workflows/ci.yml` and root `pyproject.toml`'s pytest
configuration, and therefore carries `autonomy:review`, never
`autonomy:safe`, per §2/§5's inherited carve-out.**

## 9. Gates — Rev-B

- [ ] Security
- [ ] License/compliance
- [ ] Technical feasibility
- [ ] Cost/compute-budget
- [ ] Convention-alignment
- [ ] Goal-delivery

Not yet independently re-reviewed. Rev-B incorporates every finding from
the Rev-A review round (§12) — three of which (§12: TF-1, CA-1, and the
combined GD-1/GD-2 pair) were flagged as individually disqualifying in that
round. A fresh gate pass is warranted before treating any gate as cleared.

## 10. Open questions

- Exact env-var name if a CLI/env resolver is built on top of the
  function-keyword mechanism (§3) — left to Slice 1's own implementation.
- Whether TPU/XLA support (§3, out of scope here) should become its own
  future Arc, or a Slice inside `cloud-compute-ops` once that Arc's
  Charter exists — not decided by this document.
- Whether the `autonomy:review`-for-pytest-config-diffs rule (§5) should be
  proposed back to the parent Design Charter as a Change Order once it
  reaches Rev-0, generalizing the CI-workflow-file carve-out to cover
  test-selection-affecting config more broadly.

## 11. Rollback / abandonment path

Per the parent Design Charter's §12: abandoning this Arc before reaching
its own Rev-0 is a lightweight `status: abandoned` change, not a Change
Order.

## 12. Gate review findings (Rev-A → Rev-B)

Performed by a dedicated review agent covering all six gates in one pass
(proportional to this Arc Charter's smaller scope relative to the parent
Design Charter), back-tracing from each Slice's envisioned finished state
to what Rev-A actually specified, and independently verifying every factual
claim Rev-A made about the codebase against the actual repo state (grep,
direct file reads of `train.py`, `model.py`, `uv.lock`, root
`pyproject.toml`, `CONVENTIONS.md`, `ci.yml`) rather than trusting the
Charter's own description.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| TF-1 | Slice 3 (GPU verification) cannot execute on the owner's Windows environment: `uv.lock` locks `torch` from PyPI, whose CUDA-runtime deps are Linux-only; the Windows wheel has no CUDA runtime at all, a gap `projects/em-piml/CLAUDE.md` already records as unresolved | **Critical** | §5 (wheel-index bullet), §6 (Slice 3 row) |
| CA-1 | The `@pytest.mark.gpu` exclusion mechanism Rev-A asserted twice does not exist and would not work under root `pyproject.toml`'s actual `addopts = "-m 'not slow'"` — a gpu-marked, non-slow test would run in CI's fast step on a GPU-less runner | **Critical** | §5 (marker/addopts/ci.yml bullet), §6 (Slice 1 row) |
| GD-1 | "Must not error when no accelerator is present" (Rev-A) silently converts an explicit GPU request into an unflagged CPU run — a research-provenance risk, sharpened by TF-1 since Windows always reports `is_available()==False` for a packaging reason | High | §5 (default-vs-explicit-request split) |
| GD-2 | No bit-exactness requirement for the CPU default path — the one property CI can actually enforce; existing tests use loose thresholds that wouldn't catch a shifted RNG stream | High | §5 (bit-exactness bullet), §6 (Slice 2 row) |
| CA-4 | No `CONVENTIONS.md` reconciliation section at all, despite touching the testing entry and compute-assumption entry substantively — the same omission the parent Charter's own Convention-alignment gate caught for its optimizer-default entry | High | §4 (new section) |
| CA-2 | The marker/addopts/ci.yml edits (TF-1/CA-1's remedies) contradict Rev-A's "project-scoped, `train.py`-only" framing and never exercise the parent's ci.yml-change carve-out | High | §2, §5 |
| CB-1 | No criterion that the GPU path is actually faster; the named GPU (GTX 1660 Ti, Turing, no tensor cores) runs FP64 at ~1/32 FP32 rate — the Arc could clear "placement works" while making the repo's most expensive test slower | High | §4 (compute-assumption reconciliation), §6 (Slice 3 row) |
| SE-1 | Parent §5's `autonomy:review` carve-out for CI-workflow-touching PRs is not inherited in §2, despite Slice 1 requiring exactly that kind of change | High | §2, §6 (Slice 1 row), §8 |
| TF-2 | `torch.Generator` at `train.py:357`/`:434` is device-bound; threading a non-CPU device without `Generator(device=...)` breaks, and fixing it changes the RNG stream for the same seed | High | §5, §6 (Slice 2 row), §7 |
| TF-3 | Device surface is wider than `train.py` alone — `physics.py`, `dielectric.py`, `embeddings.py`, `model.py`, and the `evaluate_*` functions every test calls are also in scope | Medium | §6 (Slice 2 row) |
| TF-4 | "~10 experiments" understated the actual surface by ~2x (19 public + 9 private training functions across 18 documented experiments) | Medium | §6 (Slice 2 row) |
| LC-1 | The CUDA wheel-index change adds a real license surface (NVIDIA CUDA EULA / cuDNN SLA) that Rev-A's "no new third-party dependency" language read out of existence | Medium | §5 (wheel-index bullet) |
| CB-2 | §6's Apple-Silicon paraphrase of the compute-assumption entry dropped the load-bearing GTX-1660-Ti clause | Medium | §4 (compute-assumption reconciliation) |
| CB-3 | Slice 3's blocker was misdiagnosed as hardware access rather than packaging (TF-1), which risked routing it toward `cloud-compute-ops` and undercutting this Arc's Tier-1 status | Medium | §6 (Slice 3 row) |
| CA-3 | "This repo's existing determinism-verification convention" doesn't exist in `CONVENTIONS.md` — the real citation is `projects/em-piml/CLAUDE.md:63-65` | Medium | §6 (Slice 3 row, citation corrected) |
| CA-6 | No requirement to record device choice in `results.csv`'s `params` column, even though device changes numeric results | Medium | §5 |
| GD-3 | This Arc's stated purpose covers unblocking `cloud-compute-ops`, which names TPU providers, but no Slice addresses TPU/XLA (a separate `torch_xla` package) | Medium | §3 (out-of-scope bullet) |
| GD-4 | Slice 1's CLI-only framing has no argparse surface to attach to anywhere in `projects/em-piml/`, is untestable by Slice 1's own CI claim, and contradicts the function-keyword FP64 precedent it cites | Medium | §3, §6 (Slice 1 row) |
| SE-2 | A diff to root `pyproject.toml`'s pytest config changes what CI runs without touching a literal "workflow file," a gap in the parent's autonomy-labeling carve-out | Medium | §5 |
| SE-3 | An env-var device selector was left unvalidated — could select an unintended device or silently change what CPU-path tests test | Medium | §5 |
| SE-4 | Adding a second package index (for the CUDA wheel) without pinning creates dependency-confusion exposure repo-wide, not just for `torch` | Medium | §5 (wheel-index bullet) |
| CA-5 | Frontmatter used `related-slices:` where `docs/design/README.md`'s schema specifies `related-arcs:`, and `status: draft` without exercising the Arc-level `proposed` value the schema also defines | Low | Frontmatter corrected |

Overall verdict on Rev-A (verbatim from the review): "Not sound enough to
proceed to its first Slices as written — three defects are disqualifying
on their own: [CA-1] is a flat factual error about the exclusion mechanism
the entire verification story rests on ...; [TF-1] means Slice 3 cannot be
executed at all on the owner's Windows environment for a packaging reason
the repo has already documented as unresolved; and [GD-1]/[GD-2] leave both
research-integrity properties ... entirely unspecified. Slices 1 and 2 are
otherwise well-shaped and the project-scoped placement decision (§3) is
correct and well-argued." All findings above are incorporated into this
revision; no gate has been independently re-checked against Rev-B yet (§9).

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-30 | Initial draft | (pending) |
| B | 2026-07-30 | Full six-gate review (22 findings: 2 Critical, 6 High, 12 Medium, 1 Low) incorporated. Corrected the CUDA-on-Windows packaging blocker (§5), the non-functional `@pytest.mark.gpu` exclusion mechanism (§5/§6), split default-vs-explicit-request device error semantics and added a CPU-path bit-exactness requirement (§5), added a `CONVENTIONS.md` reconciliation section (§4, new), widened Slice 2's actual surface and corrected its experiment count (§6), corrected Slice 3's blocker diagnosis from hardware-access to packaging (§6), switched the primary mechanism from an unbuildable CLI flag to a function keyword matching this repo's own FP64 precedent (§3), added the inherited `autonomy:review` CI/pytest-config carve-out (§2/§5/§8), and fixed frontmatter per the process doc's own schema. | (pending re-review) |

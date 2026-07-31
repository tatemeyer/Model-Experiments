---
title: "Arc Charter — field-visualization"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-30-field-visualization
slice: 2026-07-30-charter
revision: C
status: proposed
date: 2026-07-30
related-arcs: []
supersedes: null
superseded-by: null
---

# Arc Charter — field-visualization

See `docs/design/README.md` for what Design/Arc/Slice, revisions, gates, and
Change Orders mean, and the parent Design Charter
(`docs/design/specs/2026-07-28-em-piml-modernization/2026-07-28-foundation/2026-07-28-charter/README.md`,
currently Rev-F) for this Design's overall scope, non-negotiable
constraints, and cross-Arc dependencies. This document assumes both and
doesn't redefine them — it scopes `field-visualization` specifically.

## 1. Purpose & why now

Implements the toolkit decided in the parent Design Charter's Rev-E (§3):
PyVista (primary, 3D/volumetric) + Plotly (embeddable interactives) +
existing matplotlib/`mx-viz` (2D), replacing the earlier Rerun.io proposal
entirely, per the owner's explicit course-correction (PR #55 comment,
quoted in full in the parent Charter's §3): *"When we run an experiment I
want to see the base case (what it is attempting to predict) when possible
and it's output."* **Rev-A did not actually deliver a path to that
outcome** (Goal-delivery gate finding, see §12) — nothing in this repo
persists a predicted field anywhere, so "see the base case and its output"
had no artifact to render. Rev-B adds the Slice that closes this gap (§6).

## 2. Relationship to the Design Charter

This Arc Charter inherits every constraint in the parent Design Charter,
especially:

- §5's no-hosted/live-viewer-dashboard constraint — every output stays a
  file (PNG/GIF/MP4/self-contained HTML), never a running service.
- §5's CI-stays-CPU-only-**and-unchanged**-unless-explicitly-justified
  constraint. **Rev-A dropped the "and unchanged" half** (Technical
  feasibility gate finding, see §12) — this Arc's own Slice 1 requires a
  `ci.yml` change (§5 below) to make PyVista reachable in CI at all; that
  is the "explicitly justified" exception the parent constraint requires,
  named and justified in §5/§6, not a silent violation of it.
- §3's on-the-record license check of PyVista (MIT), Plotly (MIT), and VTK
  (BSD-3-Clause) — already done at Design level. **That check does not
  extend to this Arc's transitive dependencies** (License/compliance gate
  finding, see §12) — see §5.
- **§5's `autonomy:review` carve-out for any PR touching a CI workflow
  file or GitHub Actions secret/Environment** (Security gate finding, see
  §12) — omitted from Rev-A's inheritance list entirely, and directly
  relevant here since Slice 1 modifies `.github/workflows/ci.yml`.
- **This Arc Charter may not be promoted to its own Rev-0 before the
  parent Design Charter reaches Rev-0** (Convention-alignment gate finding,
  see §12): the parent is currently Rev-F/`status: draft`, and its own §18
  states the intended order explicitly — "once this document reaches
  Rev-0, its settled constraints ... are inherited as given by Arc
  Charters." Rev-A asserted inheritance as already-settled fact; it isn't,
  yet. Any parent-Charter change between now and its own Rev-0 requires
  re-checking this section's inheritance list.

This document adds only what's specific to `field-visualization`; it
doesn't re-litigate the parent's constraints.

## 3. Scope

- Extend `tools/viz` (`mx-viz`), not create a new package. **This Arc
  introduces the repo's first optional-dependency-group (extras) pattern**
  (Convention-alignment gate finding, see §12): `tools/viz/pyproject.toml`
  has no `[project.optional-dependencies]` table today — deps are flat
  (`matplotlib>=3.8`, `numpy>=1.26`) and the only existing
  `[dependency-groups]` entry (`dev`) is a PEP 735 dev-group, a different
  mechanism from extras. This Arc adds a `[project.optional-dependencies]`
  group named `3d` (`mx-viz[3d]`) containing PyVista and Plotly together —
  resolving Rev-A's open question about splitting them (no concrete reason
  found to split; keeping both in one group is simpler and both are needed
  for the near-term Slices below). **Answering the parent Charter's own
  delegated sizing question** ("whether this warrants a separate
  `tools/<name>` package or extending `mx-viz` in place," parent §3):
  extending `mx-viz` in place, gated behind the `3d` extra, is the right
  call specifically *because* extras keep the ~150-250MB VTK footprint
  opt-in — a separate package would only be warranted if that isolation
  turned out not to hold in practice.
  - **No workspace member may declare `mx-viz[3d]` among its own default
    (non-optional) dependencies** (re-review finding, see §13 — Rev-B's
    `field-array-persistence` Slice did exactly this and it nullified the
    entire extras mechanism: `uv sync --all-packages` syncs every member's
    *default* deps, so a default `mx-viz[3d]` dependency in
    `projects/em-piml/pyproject.toml` would pull VTK into every PR
    regardless of whether the `ci.yml` sync-step change lands, silently
    defeating both Slice 1's isolation and its stated CI-cost fallback).
    Persisting a field array is pure NumPy work and needs only bare
    `mx-viz` (already a default em-piml dependency) — see §6,
    `field-array-persistence`. Anything that actually needs PyVista at
    render time goes through the `mx-viz` CLI (a separate process/
    environment with the `3d` extra installed), not an in-process import
    from em-piml.
- **Data reality check** (Technical feasibility gate finding, see §12):
  every model in `projects/em-piml/src/em_piml/model.py` is `forward(x, t)
  -> E_z` — a scalar field over 1D space + time. The only *research*
  dataset (`em-piml-1d-cavity-analytical`; `tools/datasets/registry/` also
  has a second, unrelated `example-smoke-test.toml` entry — corrected in
  Rev-C, see §13) is a 200×200 `(x, t)` scalar grid. There is no 3D
  geometry, vector field, or volumetric data anywhere in this repo
  *today*, and the parent Design Charter itself says "there's no reason to
  reach for 3D machinery to render a 2D slice" (parent §3). This Arc's
  scope is therefore split explicitly:
  - **Deliverable against today's data:** a new per-timestep comparison
    figure — three panels (true / predicted / |error|) of `E(x)` at a
    fixed `t`, assembled into a GIF/MP4 over the time axis — **not** an
    animation wrapper around the existing
    `mx_viz.fields.plot_field_heatmap` (corrected in Rev-C, see §13:
    that function already plots the full `(x,t)` heatmap with time on one
    *axis*, not as a frame index, so "animating" it doesn't name a
    coherent operation). The existing `(x,t)` heatmap is retained as-is,
    as the static companion figure it already is — this Arc adds the new
    per-frame function alongside it, in `mx_viz.fields`. Plotly/PyVista
    renderings of this same 2D-over-time data (e.g. a rotatable 3D surface
    plot) are additional, not a replacement.
  - **Forward capacity, not yet exercised:** PyVista's isosurface/
    volume-rendering/streamtube primitives, which are this Arc's stated
    long-term value proposition, have no data of the right shape to
    render *yet*. **Corrected in Rev-C** (§13): this isn't a hypothetical
    future need — issues **#43** (Helmholtz eigenvalue waveguide) and
    **#44** (2D PEC cavity) are both open in this repo's own backlog and
    listed as active leads in `projects/em-piml/CLAUDE.md`'s "Open leads"
    section; either would produce genuinely 2D/3D field data. This Arc
    builds the capability; §10 tracks whether further investment beyond
    the baseline above should wait on #43/#44 landing.
- **New capability, corrected from Rev-A** (Technical feasibility gate
  finding, see §12): `mx_viz.fields.plot_field_heatmap` already renders a
  static 3-panel target/prediction/error comparison — this Arc's actual
  new contribution is the new per-frame comparison function above plus
  3D/interactive rendering backends, not inventing the multi-panel
  comparison itself.
- `io.py` and `cli.py` **are in scope**, corrected from Rev-A's inventory
  omission (Technical feasibility gate finding, see §12) — a field-array
  persistence schema (§6, `field-array-persistence`) and a render CLI verb
  belong exactly there, alongside the existing sweep-JSON schema and
  `mx-viz` CLI surface.
- Existing `mx-viz` matplotlib functions (`fields.py`, `training.py`,
  `sweeps.py`) are otherwise unchanged.

## 4. Reconciliation with existing `CONVENTIONS.md` entries

**Plotting default** (`CONVENTIONS.md:177-203`, 2026-07-27): matplotlib is
the default plotting library, chosen partly because static PNG "reports
findings as static content embedded in PR bodies and experiment-log
markdown, not a hosted/interactive dashboard" and carries "no
GPU/CUDA-adjacent dependency risk." The parent Design Charter's own §4
already reconciles this entry against the PyVista/Plotly addition in
general terms and states a trigger: **once this Arc Charter reaches Rev-0,
record a new dated `CONVENTIONS.md` entry that extends (not supersedes)
the 2026-07-27 entry — matplotlib remains the default for 2D; PyVista/Plotly
are additions scoped specifically to 3D/volumetric field rendering.**
**Rev-A never restated or acted on this trigger** (Convention-alignment
gate finding, see §12), even though this document is the very thing that
trips it. Restating it here: writing that new `CONVENTIONS.md` entry is
part of this Arc's own Rev-0 promotion, not deferred to a Slice, and not
assumed to happen automatically as a side effect of the parent Design
Charter's Rev-0.

## 5. Non-negotiable constraints

- Headless/CI-safe — this Arc doesn't just cite the general
  PyVista-headless-CPU-rendering research finding from the parent Charter,
  it must independently confirm that claim holds in this repo's actual CI
  environment (`pyvista-headless-ci`) before anything downstream depends on
  it.
- No hosted/live viewer — outputs are files only.
- **The packaging mechanism is named precisely, and CI reachability is not
  assumed** (Technical feasibility gate finding, see §12):
  `.github/workflows/ci.yml` installs with `uv sync --all-packages`, which
  syncs default dependencies only — it does **not** enable extras. Under
  `mx-viz[3d]` as scoped in §3, PyVista/Plotly are simply absent from CI
  unless the sync step changes (`--extra 3d` / `--all-extras`) or a
  separate, path-filtered job is added. `pyvista-headless-ci` (§6) must
  propose and justify one of these explicitly — this is the "explicitly
  justified" CI-workflow change §2 already flags as expected, not optional.
  **Any such PR carries `autonomy:review`, never `autonomy:safe`, per the
  inherited carve-out in §2.**
- **Transitive dependencies beyond the three already checked at Design
  level get their own license check** (License/compliance gate finding,
  see §12): `open_gif`/`open_movie` (§6, `field-render-core`) route through
  `imageio`/`imageio-ffmpeg`, which bundles a prebuilt FFmpeg binary whose
  license (LGPL vs. GPL depending on build) is a separate question from
  VTK's BSD-3. `export_html` pulls the `trame`/`trame-vtk`/`trame-vuetify`
  stack. MP4 export is contingent on the FFmpeg-build license check
  clearing under the parent's license-file-not-just-metadata standard
  (parent §5); if it doesn't clear, GIF-only is the fallback, not a
  blocker on the whole Slice.
- **Self-contained HTML export must actually be self-contained** (Security
  gate finding, see §12): Plotly's `write_html` and PyVista's
  `export_html` can both reference a remote CDN instead of inlining their
  JS bundle, depending on options/version. An export that fetches
  third-party JavaScript over the network when opened does not satisfy
  this Arc's file-only constraint. `field-render-core`/`plotly-interactive`
  must verify no remote `http(s)` `src`/`href` appears in the emitted HTML
  (`include_plotlyjs=True` for Plotly; the verified equivalent for
  PyVista), and this is a CI-checkable assertion on the emitted file, not
  just a stated intent.
- **HTML export also triggers the parent's NOTICE-preservation revisit
  condition** (License/compliance gate finding, see §12): parent §5 states
  Apache-2.0 NOTICE obligations aren't currently triggered because "no Arc
  redistributes dependency source... revisit if any Arc's output is ever
  packaged/published standalone." A self-contained HTML file inlining
  plotly.js (MIT) or vtk.js (BSD-3-Clause) source *is* that redistribution.
  `plotly-interactive` must confirm the embedded bundle's license header
  travels with it intact.
- **No dependency added under this Arc may download an executable outside
  `uv.lock`'s resolution at install or runtime** (Security gate finding,
  see §12) — relevant if `imageio-ffmpeg`'s binary-fetch behavior is
  adopted for MP4 export; document and pin it in the Slice PR, or drop MP4
  for GIF-only. **The compiled VTK wheel itself is also part of this
  supply-chain surface** (re-review finding, see §13) — this repo already
  treats supply chain as a first-class concern (`CONVENTIONS.md:69-78`'s
  SHA-pinned-Actions entry, motivated by the `trivy-action` incident); the
  same standard `device-abstraction`'s Arc Charter cites for its own
  wheel-index change applies here.
- **The `mx-viz field <artifact>` CLI verb (§6, `field-array-persistence`)
  must not use an unsafe deserialization format** (re-review finding, see
  §13 — new in Rev-B, not present in Rev-A): loading an on-disk array with
  `pickle`, `torch.load`, or `np.load(..., allow_pickle=True)` would make
  the verb an arbitrary-code-execution sink for any file a user is handed.
  The artifact format is `.npz` with `allow_pickle=False` (or raw arrays
  plus a JSON metadata sidecar) — no exceptions — and "loads with pickle
  disabled" is a CI-checked assertion in `field-array-persistence`'s own
  tests, not just a schema round-trip.
- **Artifact distribution has a named mechanism, since committing is not
  an option** (Convention-alignment / Goal-delivery gate findings, see
  §12): `.gitignore` already excludes generated plots/results
  (`tools/README.md` states the same: write to `.outputs/<project>/`,
  gitignored, "never commit generated plots/results"). PNG/GIF/MP4 are
  distributed via GitHub comment/PR-body upload (GitHub renders these
  inline). **Self-contained HTML has no such channel** — GitHub does not
  render attached HTML in a PR body or comment. `plotly-interactive`'s
  PR-facing deliverable is therefore PNG/GIF (matching the other Slices'
  distribution story); HTML export is a local-inspection/optional artifact
  only, not the Slice's primary claim of value, unless a future revision
  names an actual hosting mechanism for it.
- **VTK's footprint gets a concrete acceptance criterion, not just a
  "worth confirming"** (Cost/compute-budget gate finding, see §12):
  `ci.yml` runs one job for the entire workspace on every PR — if the `3d`
  extra becomes reachable by that job's sync step, ~150-250MB of compiled
  VTK enters install/cache for *every* PR in *every* project. **Acceptance
  criterion, set here rather than deferred to the PR incurring the cost**
  (re-review finding, see §13 — Rev-B's "to be set by `pyvista-headless-ci`'s
  own PR" let the same PR set its own bar, and "measured at warm cache" is
  the measurement that most understates the cost, since `setup-uv`'s cache
  key changes precisely when `uv.lock` gains VTK): PR-gating CI wall-clock
  must not increase by more than **90 seconds**, measured cold-cache (a
  cache-miss run, not a warm-cache one) against the pre-Arc baseline. If
  `pyvista-headless-ci`'s own measurement finds this genuinely unworkable,
  it may propose a revised number with the actual measured numbers
  attached — but it doesn't set the bar unilaterally. Fallback if it
  doesn't clear: keep the `3d` extra out of `uv sync --all-packages`'s
  default scope and run the headless check in a separate, non-blocking,
  path-filtered workflow instead.
- If stock `ubuntu-latest` needs an added system package (e.g.
  `libgl1`/`xvfb`) to make off-screen rendering work, that is itself a
  proposed-and-justified `ci.yml` change under §2's carve-out, carrying
  `autonomy:review` and a recorded justification — not an inline quick
  fix, and **the added package is version-pinned** (re-review finding, see
  §13 — Rev-B fixed the "unjustified" half of this but not the "unpinned"
  half of the original finding).
- Every dependency this Arc actually adds gets its license recorded in the
  Slice PR that adds it (parent §5), even though the underlying check for
  PyVista/Plotly/VTK already happened at Design level (parent §3) — the
  requirement is "recorded in the Slice PR," not just "checked once."

## 6. Named Slices and sequencing

| Slice | Scope | Verifiable by | Issue |
|---|---|---|---|
| `pyvista-headless-ci` | Add `mx-viz[3d]` as a `[project.optional-dependencies]` extras group (PyVista + Plotly); confirm off-screen/headless rendering actually works in this repo's real `ubuntu-latest` CI runner; propose and justify the required `ci.yml` sync-step change (§5) — **this PR modifies `.github/workflows/ci.yml`, so it carries `autonomy:review`, not `autonomy:safe`** (§2, §5) | CI, once the sync-step change lands | [#58](https://github.com/tatemeyer/Model-Experiments/issues/58) |
| `field-array-persistence` (**new in Rev-B**, closes the Goal-delivery gap — see §12, G1/T1) | Define and write a target-field + predicted-field array artifact to `.outputs/<project>/`, wiring into em-piml's existing inference/eval path; add an `mx-viz field <artifact>` CLI verb. **Format fixed explicitly, not deferred** (re-review finding, see §13): `.npz` with keys `x`, `t`, `grid_x`, `grid_t`, `true`, `predicted`, `schema_version` — extending the existing precedent at `tools/datasets/registry/generators/em_piml_1d_cavity_analytical.py:24-36` — loaded with `allow_pickle=False` (§5). **Depends on bare `mx-viz` only, never `mx-viz[3d]`** (§3, §13) — persistence is pure NumPy. Also updates `projects/em-piml/CLAUDE.md`'s experiment-running instructions to mention this capability exists (closes the remaining half of G1 — persistence existing isn't the same as it being a discoverable, documented part of running an experiment) | CI (artifact schema round-trips; CLI verb runs without error on a fixture array; pickle-disabled load asserted) | [#61](https://github.com/tatemeyer/Model-Experiments/issues/61) |
| `field-render-core` | Add a new `mx_viz.fields` function rendering three panels (true / predicted / \|error\|) of `E(x)` at fixed `t`, assembled into `open_gif`/`open_movie`/`export_html` output over the time axis — **not** an animation of the existing `plot_field_heatmap` (corrected in Rev-C, §3, §13); add PyVista-backed rendering of the same 2D-over-time data (rotatable 3D surface); create `tools/viz/CLAUDE.md` (root `CLAUDE.md`'s scaling principle 1 requires one, and `train.py:1208-1209` already references a "tools/viz's CLAUDE.md-equivalent rationale" that doesn't exist yet); **owns the no-remote-CDN assertion on any HTML it emits via `export_html`** (§5 — corrected ownership, see §13) | CI (render doesn't crash; no-CDN assertion on emitted HTML) + manual visual check | [#62](https://github.com/tatemeyer/Model-Experiments/issues/62) |
| `plotly-interactive` | Plotly wrapper (`Isosurface`/`Volume`/`Streamtube`/`Surface`) for lightweight interactives; **PNG/GIF export as the PR-facing deliverable, with its own CI check** (§5 — corrected in Rev-C, see §13: Rev-B's only CI check here was on the secondary HTML artifact, none on the actual PR-facing PNG/GIF path); HTML export verified self-contained (§5) but treated as a secondary, local-inspection artifact | CI (PNG/GIF render doesn't crash; no-CDN assertion on any emitted HTML) + manual visual check | [#63](https://github.com/tatemeyer/Model-Experiments/issues/63) |
| `poc-experiment-rerender` | Re-render one existing em-piml experiment's results (persisted via `field-array-persistence`) end-to-end, as this Arc's own proof the whole path works on real data. Runs locally/manually only — **not because CI can't afford it** (`ci.yml` already has a "Test (slow)" step that trains models, and the ~35s baseline is well under `projects/em-piml/CLAUDE.md`'s actual guidance — corrected over-justification, see §13), **but because a one-off proof-of-concept isn't a standing verification CI should re-run on every PR** | Manual — recorded in an experiment write-up | [#64](https://github.com/tatemeyer/Model-Experiments/issues/64) |

`pyvista-headless-ci` gates everything downstream — nothing should assume
headless rendering works here until it's independently confirmed, not just
cited from prior research. `field-array-persistence` gates
`field-render-core`'s and `poc-experiment-rerender`'s ability to render
anything real.

## 7. Cross-cutting gaps / risks specific to this Arc

- **PyVista's primary stated value (isosurface/volume/streamtube
  rendering) has no data of the right shape in this repo today** (§3) —
  every model is a 1D scalar field over `(x, t)`. This Arc ships the
  capability; exercising it end-to-end depends on issues **#43**
  (Helmholtz eigenvalue waveguide) or **#44** (2D PEC cavity), both open
  in this repo's own backlog — corrected in Rev-C from "a 2D/3D EM problem
  this repo hasn't posed yet," which was factually wrong (§13). Not a
  reason to abandon the toolkit choice (the parent Design Charter's own
  reasoning for picking PyVista over alternatives stands independent of
  today's data shape), but a real gap between what this Arc *can* deliver
  now and what it's *ultimately for*.
- **Data contract**, similar in shape to the one `mx-viz`'s own
  `CONVENTIONS.md` entry already flags between `mx_viz.io`'s JSON schema
  and `results.csv`: target-field and predicted-field arrays need matching
  grid/coordinate metadata to render correctly side-by-side. **Now fully
  specified, not just assigned** (§13): `field-array-persistence` (§6)
  fixes the exact `.npz` schema (keys, `schema_version`,
  `allow_pickle=False`) rather than leaving the format itself open.

## 8. Relationship to the Issue/PR loop

Intent Issues #58 (`pyvista-headless-ci`), #61 (`field-array-persistence`),
#62 (`field-render-core`), #63 (`plotly-interactive`), and #64
(`poc-experiment-rerender`) already exist, opened ahead of this Arc
Charter reaching Rev-0 at the owner's explicit direction — corrected in
Rev-C to match reality (§13; the same correction `device-abstraction`'s
sibling Arc Charter makes in its own §8). **Rev-A's review found real
decision-making compressed into Slice-level Issues with no gate review**
(Convention-alignment gate finding, see §12) — specifically the packaging
mechanism (`pyvista-headless-ci`) and the data-contract shape
(`field-array-persistence`). Rev-B/C resolve both decisions directly in
this Arc Charter (§3, §5, §6) rather than deferring them to a separate
Slice-level document layer. **This is a deliberate deviation from
`docs/design/README.md`'s own hierarchy definition** (Convention-alignment
re-review finding, see §13), which states Slice documents, not Arc
Charters, are "where implementation detail (file paths, function
signatures, test plans) actually belongs." Named explicitly rather than
argued from local precedent alone (Rev-B's framing): the five Intent
Issues above function as this Design's Slice documents in practice, since
they now carry the implementation detail (exact schema keys, exact CLI
flags, exact CI assertions) that a separate Slice-document layer would
otherwise hold — a choice proportionate to this Design's scope, not a
gap.

## 9. Gates — Rev-C

- [ ] Security
- [ ] License/compliance
- [ ] Technical feasibility
- [ ] Cost/compute-budget
- [ ] Convention-alignment
- [ ] Goal-delivery

Not yet independently re-reviewed against Rev-C. An independent re-review
of Rev-B (§13) found 15 of 24 Rev-A findings fully resolved, 5 partially
resolved (later found to be 4 partial + 4 not-yet-verified-clean on
closer read), plus 10 new findings in Rev-B's own text (1 Critical — the
`mx-viz[3d]` default-dependency defect, N1) — all incorporated into this
revision. Per that re-review's own gate-by-gate read: License/compliance
and Convention-alignment were assessed as close to clearable as Rev-B
stood; Cost/compute-budget, Technical feasibility, Goal-delivery, and
Security were held pending exactly the fixes this revision makes. A fresh
pass against Rev-C is still warranted before treating any gate as formally
cleared — this section records the re-review's assessment, not an
independent gate clearance.

## 10. Open questions

- Whether `poc-experiment-rerender` should pick an experiment already
  canonical elsewhere in this repo (e.g. the R3 long-horizon or
  num-bands-gap experiments) or a simpler one chosen specifically for
  being easy to visually sanity-check — unresolved, left to that Slice.
- Whether PyVista's 3D/volumetric capability should wait for issue #43 or
  #44 to land (§7) before further investment beyond `field-render-core`'s
  baseline, or whether building it ahead of that need is worth it as
  forward capacity — not decided by this Charter.

## 11. Rollback / abandonment path

Per the parent Design Charter's §12: abandoning this Arc before reaching
its own Rev-0 is a lightweight `status: abandoned` change, not a Change
Order.

## 12. Gate review findings (Rev-A → Rev-B)

Performed by a dedicated review agent covering all six gates in one pass,
back-tracing from each Slice's envisioned finished state to what Rev-A
actually specified, and independently verifying every factual claim Rev-A
made about the codebase (`tools/viz/`, `.github/workflows/ci.yml`,
`CONVENTIONS.md`, `.gitignore`, `tools/README.md`, `projects/em-piml/`)
rather than trusting the Charter's own description.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| T1 | `poc-experiment-rerender` has nothing to re-render — this repo doesn't persist a predicted field anywhere (`mx_viz/cli.py` states this outright; confirmed by grep) | **Critical** | §6 (new `field-array-persistence` Slice) |
| T2 | `pyvista-headless-ci`'s "Verifiable by: CI" is unreachable under the extras packaging shape Rev-A recommended, since `ci.yml`'s `uv sync --all-packages` doesn't enable extras; the extras-vs-dependency-group mechanism was also never named precisely | **Critical** | §5, §6 (Slice 1 row) |
| G1 | The finished state is a rendering library with no path from "run an experiment" to "see the base case and its output" — the owner's own stated goal — since nothing hooks into the training/eval workflow or persists a field | **Critical** | §6 (new `field-array-persistence` Slice, closes the loop) |
| T3 | The Arc's inheritance of the parent's CI constraint drops the binding "and unchanged" half, understating what Slice 1 actually requires | High | §2 |
| T4 | The Arc's primary renderer's core value (isosurface/volume/streamtube) has no data of that shape anywhere in this repo — every model is a 1D scalar `(x,t)` field | High | §3, §7 (scope split: deliverable-now vs. forward-capacity) |
| L1 | The Design-level license check doesn't cover this Arc's own transitive dependencies (`imageio`/`imageio-ffmpeg`'s bundled FFmpeg, `trame`/vtk.js) | High | §5 |
| C1 | VTK's footprint lands on every PR in the repo (one workspace-wide CI job), with no threshold set for an acceptable slowdown | High | §5 |
| V1 | Every artifact this Arc produces is gitignored (`.outputs/`), while the Arc's own premise assumes PR-embeddable output — never named the conflict | High | §5 (distribution-mechanism bullet) |
| V2 | This Arc Charter could freeze at Rev-0 before its still-draft (Rev-F) parent does, contradicting the parent's own stated intended order | High | §2 |
| G2 | "Drops directly into a PR" has no working distribution channel for self-contained HTML specifically — GitHub doesn't render attached HTML | High | §5, §6 (`plotly-interactive` row: PNG/GIF is the PR-facing deliverable) |
| S1 | "Self-contained HTML" was asserted, never verified — both Plotly and PyVista can emit CDN-referencing (non-self-contained) HTML depending on options | High | §5 |
| T5 | The Arc's "new capability" (target/prediction/error multi-panel) is already half-implemented by `mx_viz.fields.plot_field_heatmap` — the genuinely new part is animation + 3D/interactive backends | Medium | §3 |
| T6 | Rev-A's file inventory of `mx-viz` omitted `io.py`/`cli.py`, which are exactly where the field-array contract and render verb land | Medium | §3, §6 (`field-array-persistence`) |
| L2 | Self-contained HTML export is redistribution of dependency source (plotly.js, vtk.js), which the parent said would trigger a NOTICE-obligation revisit — never mentioned | Medium | §5 |
| C2 | `poc-experiment-rerender`'s retraining cost was unstated and would collide with `projects/em-piml/CLAUDE.md`'s CI-runtime guidance if run in CI | Medium | §6 (Slice row: manual/local only) |
| V3 | Never recorded the `CONVENTIONS.md`-entry-trigger obligation from the parent's §4, even though this document is the trigger | Medium | §4 (new section) |
| V4 | `mx-viz[3d]` as a first-in-repo extras pattern has an unstated consumer-side change needed in `projects/em-piml/pyproject.toml` | Medium | §3 |
| V5 | `tools/viz` has no scoped `CLAUDE.md` (root `CLAUDE.md` scaling principle 1 requires one); `train.py` already references one that doesn't exist | Medium | §6 (`field-render-core` row) |
| V6 | Slice-level Issues would carry real, ungated decisions (data contract, packaging split, experiment choice) with no gate review | Medium | §3/§5/§6 (decisions resolved directly in this Arc Charter, not deferred), §8 (rationale for not adding a separate Slice-document layer) |
| G3 | §9's packaging-split open question blocked the very Slice (`pyvista-headless-ci`) that needed to implement it | Medium | §3 (resolved: combined `3d` extras group) |
| S2 | New binary/transitive supply-chain surface (VTK compiled wheels, `imageio-ffmpeg`'s bundled FFmpeg) unacknowledged in a repo with an explicit SHA-pinning supply-chain posture | Medium | §5 |
| S3 | Parent's `autonomy:review` CI-diff carve-out not restated, despite Slice 1 being the most likely Slice in this Design so far to touch `ci.yml` | Medium | §2, §6 (Slice 1 row) |
| V7 | Frontmatter used `related-slices:` where the process doc's schema specifies `related-arcs:`, and `status: draft` without exercising the `proposed` Arc-level value | Low | Frontmatter corrected |
| S4 | The headless-rendering fallback path (system packages, `xvfb`) is an unpinned, unjustified CI change if the primary claim fails | Low | §5 |

Overall verdict on Rev-A (verbatim from the review): "Not sound enough to
proceed to its first Slices as written — three Critical findings sit at the
Arc's foundations: `poc-experiment-rerender` has no persisted experiment
results to re-render ..., `pyvista-headless-ci`'s 'verifiable by CI' is
unreachable under the optional-extras shape §3 recommends without a
`ci.yml` change the Charter never proposed, and no Slice closes the loop
from 'run an experiment' to 'see the base case vs. the output,' which is
the owner-stated goal the entire Arc exists to serve... A Rev-B that fixes
[these], adds the workflow-closing Slice ..., and records the
`.outputs/`-distribution and `CONVENTIONS.md`-trigger obligations would be
in good shape — none of this requires rethinking the PyVista/Plotly toolkit
choice itself." All findings above are incorporated into this revision; no
gate has been independently re-checked against Rev-B yet (§9).

## 13. Re-review findings (Rev-B → Rev-C)

Performed by a second, independent review agent (not the one that produced
§12), whose job was specifically to verify — not assume — that Rev-B's
claimed remedies actually hold: for each of §12's 24 findings, it
independently re-derived the underlying facts (grep/read the real
codebase) rather than trusting Rev-B's own citations, then separately
searched Rev-B's new text — including the brand-new
`field-array-persistence` Slice, since it's new content rather than a fix
to existing text — for problems Rev-A didn't have.

**Tally against the 24 Rev-A findings: 18 RESOLVED (including T2, resolved
for Slice 1 specifically but with the same defect reintroduced elsewhere —
see N2), 6 PARTIALLY RESOLVED (G1, T4, C1, V6, S2, S4), 0 NOT RESOLVED.**

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| N1 | `field-array-persistence`, Rev-B's own new Slice, made `mx-viz[3d]` a *default* em-piml dependency — since `uv sync --all-packages` syncs default deps for every workspace member, this pulled VTK into every PR regardless of `pyvista-headless-ci`'s workflow change, nullifying the extras isolation and voiding §5's own stated CI-cost fallback | **Critical** | §3 (constraint added: no default `mx-viz[3d]` dependency), §6 (Slice depends on bare `mx-viz` only) |
| N2 | The "Verifiable by: CI" claim T2 fixed for Slice 1 recurred, unfixed, in Slices 2 and 4's own CI columns, since both assume the `3d` extra is reachable in the gating job without stating which job or under which of §5's two branches (default-sync vs. separate-workflow) | High | §6 (`field-render-core`/`plotly-interactive` rows now own their specific CI assertions) |
| N3 | The flagship near-term deliverable ("animate the existing `plot_field_heatmap`") named an operation that isn't coherent — that function already puts time on an axis, not a frame index, so "animating" it doesn't specify whether the result is a row-reveal of a static image or an entirely different figure | High | §3, §6 (`field-render-core`: named as a new per-frame `E(x)` comparison function, not an animation wrapper) |
| N4 | §5's no-CDN-in-HTML requirement and §6's per-Slice verification columns disagreed: `field-render-core` (which owns `export_html`) had no CI check for it, while `plotly-interactive`'s only CI check was on the HTML artifact §5 demoted to secondary, with none on its actual PR-facing PNG/GIF deliverable | Medium | §6 (both rows corrected) |
| N5 | "No 2D/3D EM problem exists in this repo" was factually wrong — issues #43 and #44 are both open and listed as active leads in `projects/em-piml/CLAUDE.md` | Medium | §3, §7 (reframed with the actual issue numbers) |
| N6 | §8 claimed the data-contract decision was "resolved" while §6/§7 both only assigned *where* it would be defined, not the format itself — no keys, no versioning | Medium | §6 (`field-array-persistence`: exact `.npz` schema now specified) |
| N7 | The new `mx-viz field <artifact>` CLI verb (entirely new in Rev-B) had no format constraint — `pickle`/`torch.load`/`allow_pickle=True` would each satisfy "schema round-trips" while making the verb an arbitrary-deserialization sink | Medium (Security) | §5, §6 (`.npz`, `allow_pickle=False`, CI-checked) |
| N8 | Ownership mismatches: `tools/viz/CLAUDE.md` assigned to `field-render-core` in the Charter but also listed as a Slice-1 (`pyvista-headless-ci`) criterion in Issue #58; `related-arcs: [jax-migration]` in frontmatter never justified anywhere in the body | Low | §6 (single owner: `field-render-core`), frontmatter (`jax-migration` dropped) |
| N9 | Two small factual errors: "the only registered dataset" ignored a second, unrelated smoke-test dataset entry; `poc-experiment-rerender`'s stated reason for staying out of CI (would breach the CI-runtime guidance) was over-stated, since CI already has a slow step that trains models within that same guidance | Low | §3 (dataset citation corrected), §6 (`poc-experiment-rerender` re-justified on the real ground: a one-off proof, not a recurring CI concern) |
| N10 | §8's rationale for skipping a separate Slice-document layer cited only local precedent ("no Slice anywhere has one") while `docs/design/README.md` itself defines Slice documents as where implementation detail belongs — the opposite of what Rev-B actually did | Medium (Convention-alignment) | §8 (named as a deliberate, stated deviation; the Intent Issues function as this Design's Slice documents in practice) |

Two items are folded into N-numbered findings above rather than listed
separately: the Cost/compute-budget gate's C1 finding was reassessed as
**still not resolved** on independent re-check (the threshold was deferred
to the PR incurring the cost, and "warm cache" was identified as the wrong
measurement) — fixed in §5 with a concrete, cold-cache-measured 90-second
threshold set here rather than deferred; and the Security gate's S4
finding was found only half-fixed (the "unjustified" half closed, the
"unpinned" half still open) — fixed in §5 with an explicit
version-pinning requirement.

Overall verdict from the re-review (verbatim, including a "15 of 24"
figure in the review's own closing summary that doesn't match the 18
tallied from its own per-finding verdicts above — preserved as actually
written rather than silently corrected): "Rev-B is a large and genuinely
well-grounded improvement... The improvement over Rev-A is real and
substantial: all three Rev-A Criticals are structurally addressed, 15
of 24 findings are fully resolved on independent re-verification... What
blocks it: Cost/compute-budget — N1 is disqualifying on its own... The
needed changes are contained: strike the `mx-viz[3d]` dependency from
Slice 2, specify the animation's per-frame quantity, fix or admit the
data-contract deferral, pin the artifact format with pickle disabled, and
reconcile §5/§6's verification columns. None of this requires rethinking
the PyVista/Plotly toolkit choice or the new Slice's existence." All
findings above are incorporated into this revision.

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-30 | Initial draft | (pending) |
| B | 2026-07-30 | Full six-gate review (24 findings: 3 Critical, 8 High, 11 Medium, 2 Low) incorporated. Added a new `field-array-persistence` Slice that closes both the Goal-delivery gap (no path from "run an experiment" to "see the result") and the Technical-feasibility gap (nothing to re-render) in one move. Corrected the CI-packaging mechanism for `pyvista-headless-ci` and named the required `ci.yml` change explicitly. Split scope into what's deliverable against today's 1D-cavity data vs. PyVista's forward-capacity 3D value proposition. Added a `CONVENTIONS.md` reconciliation section (§4, new) recording the trigger the parent Charter already named. Named the artifact-distribution mechanism given `.outputs/` is gitignored. Added the inherited `autonomy:review` CI-diff carve-out and a Rev-0-ordering constraint relative to the still-draft parent. | (pending re-review) |
| C | 2026-07-30 | Independent re-review (§13) verified 18/24 Rev-B remedies fully hold, 6 partially, 0 failed, plus 10 new findings (1 Critical). Fixed the Critical: `field-array-persistence` no longer makes `mx-viz[3d]` a default em-piml dependency, which had silently defeated Slice 1's entire extras-isolation mechanism (§3/§6). Named the flagship near-term deliverable's exact per-frame quantity instead of an incoherent "animate the heatmap" (§3/§6). Fixed the `.npz` field-array schema explicitly rather than leaving it to a Slice (§6). Pinned the new CLI verb's format against unsafe deserialization (§5/§6). Set a concrete, cold-cache-measured CI-cost threshold instead of deferring it to the PR incurring the cost (§5). Corrected a factually wrong "no 2D/3D problem exists" claim against this repo's own open issues #43/#44 (§3/§7). Reconciled §5/§6's CI-verification-column mismatches, fixed frontmatter and two small factual errors, and named the Slice-document-layer deviation explicitly rather than only by precedent (§8). | (pending re-review) |

---
title: "Arc Charter — field-visualization"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-30-field-visualization
slice: 2026-07-30-charter
revision: B
status: proposed
date: 2026-07-30
related-arcs: [jax-migration]
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
  for the near-term Slices below). **This requires a consumer-side change**
  in `projects/em-piml/pyproject.toml`, whose `[tool.uv.sources] mx-viz =
  { workspace = true }` entry currently resolves bare `mx-viz` — any
  em-piml code that wants 3D/interactive rendering must depend on
  `mx-viz[3d]` instead.
- **Data reality check** (Technical feasibility gate finding, see §12):
  every model in `projects/em-piml/src/em_piml/model.py` is `forward(x, t)
  -> E_z` — a scalar field over 1D space + time. The only registered
  dataset (`em-piml-1d-cavity-analytical`) is a 200×200 `(x, t)` scalar
  grid. There is no 3D geometry, vector field, or volumetric data anywhere
  in this repo today, and the parent Design Charter itself says "there's
  no reason to reach for 3D machinery to render a 2D slice" (parent §3).
  This Arc's scope is therefore split explicitly:
  - **Deliverable against today's data:** an animated version of the
    existing 2D target/prediction/error comparison
    (`mx_viz.fields.plot_field_heatmap`, which already renders exactly
    this 3-panel layout statically) over the field's time axis, plus
    Plotly/PyVista renderings of that same 2D data where they add real
    value (e.g. a rotatable 3D surface plot of a 2D scalar field over
    time, which PyVista/Plotly both support natively).
  - **Forward capacity, not yet exercised:** PyVista's isosurface/
    volume-rendering/streamtube primitives, which are this Arc's stated
    long-term value proposition, have no data of the right shape to
    render until a genuinely 2D/3D EM problem exists in this repo. This
    Arc builds the capability; it does not currently have a use case that
    exercises it end-to-end.
- **New capability, corrected from Rev-A** (Technical feasibility gate
  finding, see §12): `mx_viz.fields.plot_field_heatmap` already renders a
  static 3-panel target/prediction/error comparison — this Arc's actual
  new contribution is *animating* that comparison over time and *adding*
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
  for GIF-only.
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
  VTK enters install/cache for *every* PR in *every* project. Acceptance
  criterion: PR-gating CI wall-clock must not increase by more than a
  stated threshold (to be set by `pyvista-headless-ci`'s own PR, measured
  at warm cache) beyond the pre-Arc baseline. Fallback if it doesn't clear:
  keep the `3d` extra out of `uv sync --all-packages`'s default scope and
  run the headless check in a separate, non-blocking, path-filtered
  workflow instead.
- If stock `ubuntu-latest` needs an added system package (e.g.
  `libgl1`/`xvfb`) to make off-screen rendering work, that is itself a
  proposed-and-justified `ci.yml` change under §2's carve-out, carrying
  `autonomy:review` and a recorded justification — not an inline quick fix.
- Every dependency this Arc actually adds gets its license recorded in the
  Slice PR that adds it (parent §5), even though the underlying check for
  PyVista/Plotly/VTK already happened at Design level (parent §3) — the
  requirement is "recorded in the Slice PR," not just "checked once."

## 6. Named Slices and sequencing

| Slice | Scope | Verifiable by |
|---|---|---|
| `pyvista-headless-ci` | Add `mx-viz[3d]` as a `[project.optional-dependencies]` extras group (PyVista + Plotly); confirm off-screen/headless rendering actually works in this repo's real `ubuntu-latest` CI runner; propose and justify the required `ci.yml` sync-step change (§5) — **this PR modifies `.github/workflows/ci.yml`, so it carries `autonomy:review`, not `autonomy:safe`** (§2, §5) | CI, once the sync-step change lands |
| `field-array-persistence` (**new in Rev-B**, closes the Goal-delivery gap — see §12, G1/T1) | Define and write a target-field + predicted-field array artifact (with matching grid/coordinate metadata) to `.outputs/<project>/`, wiring into em-piml's existing inference/eval path; add an `mx-viz field <artifact>` CLI verb; `projects/em-piml/pyproject.toml` depends on `mx-viz[3d]` (§3) | CI (artifact schema round-trips; CLI verb runs without error on a fixture array) |
| `field-render-core` | Animate the existing `mx_viz.fields.plot_field_heatmap` 3-panel comparison over the time axis; add PyVista-backed rendering of the same 2D data (rotatable 3D surface, `open_gif`/`open_movie`/`export_html`); create `tools/viz/CLAUDE.md` (root `CLAUDE.md`'s scaling principle 1 requires one, and `train.py:1208-1209` already references a "tools/viz's CLAUDE.md-equivalent rationale" that doesn't exist yet) | CI (render doesn't crash) + manual visual check |
| `plotly-interactive` | Plotly wrapper (`Isosurface`/`Volume`/`Streamtube`/`Surface`) for lightweight interactives; PNG/GIF export as the PR-facing deliverable (§5); HTML export verified self-contained (§5) but treated as a secondary, local-inspection artifact | CI (no-CDN assertion on emitted HTML) + manual visual check |
| `poc-experiment-rerender` | Re-render one existing em-piml experiment's results (persisted via `field-array-persistence`) end-to-end, as this Arc's own proof the whole path works on real data. **Runs locally/manually only, never as a CI step** — it involves retraining (35-500s CPU per `projects/em-piml/CLAUDE.md`'s documented runtimes) and would breach that file's CI-runtime guidance if run in CI | Manual — recorded in an experiment write-up |

`pyvista-headless-ci` gates everything downstream — nothing should assume
headless rendering works here until it's independently confirmed, not just
cited from prior research. `field-array-persistence` gates
`field-render-core`'s and `poc-experiment-rerender`'s ability to render
anything real.

## 7. Cross-cutting gaps / risks specific to this Arc

- **PyVista's primary stated value (isosurface/volume/streamtube
  rendering) has no data of the right shape in this repo today** (§3) —
  every model is a 1D scalar field over `(x, t)`. This Arc ships the
  capability; exercising it end-to-end depends on a 2D/3D EM problem this
  repo hasn't posed yet. Not a reason to abandon the toolkit choice (the
  parent Design Charter's own reasoning for picking PyVista over
  alternatives stands independent of today's data shape), but a real gap
  between what this Arc *can* deliver now and what it's *ultimately for*.
- **Data contract gap**, similar in shape to the one `mx-viz`'s own
  `CONVENTIONS.md` entry already flags between `mx_viz.io`'s JSON schema
  and `results.csv`: target-field and predicted-field arrays need matching
  grid/coordinate metadata to render correctly side-by-side.
  `field-array-persistence` (§6) is where this contract gets defined —
  Rev-A left it as an unassigned gap; Rev-B assigns it explicitly.

## 8. Relationship to the Issue/PR loop

Each Slice above becomes its own Intent Issue + PR once this Arc Charter
reaches Rev-0, per the parent Design Charter's §8 / `docs/design/README.md`'s
process. **Rev-A's review found real decision-making compressed into
Slice-level Issues with no gate review** (Convention-alignment gate
finding, see §12) — specifically the packaging mechanism
(`pyvista-headless-ci`) and the data-contract shape
(`field-array-persistence`). Rev-B resolves both decisions directly in this
Arc Charter (§3, §5, §6) rather than deferring them to a separate
Slice-level document layer this repo's process doesn't otherwise use — no
Slice anywhere in this Design so far has gotten its own document distinct
from its Arc Charter's own Named-Slices table, and introducing that layer
here for just two Slices would be more process than this gap actually
requires. Their Intent Issues carry the resolved decisions directly instead.

## 9. Gates — Rev-B

- [ ] Security
- [ ] License/compliance
- [ ] Technical feasibility
- [ ] Cost/compute-budget
- [ ] Convention-alignment
- [ ] Goal-delivery

Not yet independently re-reviewed. Rev-B incorporates every finding from
the Rev-A review round (§12), including three flagged as individually
disqualifying in that round (T1/G1's combined field-persistence gap, T2's
unreachable CI-verification claim). A fresh gate pass is warranted before
treating any gate as cleared.

## 10. Open questions

- Whether `poc-experiment-rerender` should pick an experiment already
  canonical elsewhere in this repo (e.g. the R3 long-horizon or
  num-bands-gap experiments) or a simpler one chosen specifically for
  being easy to visually sanity-check — unresolved, left to that Slice.
- Whether PyVista's 3D/volumetric capability should wait for a genuinely
  2D/3D EM problem to exist (§7) before further investment beyond
  `field-render-core`'s baseline, or whether building it ahead of that need
  is worth it as forward capacity — not decided by this Charter.
- Whether the acceptance threshold for CI wall-clock increase (§5) should
  be set here or left entirely to `pyvista-headless-ci`'s own measurement —
  leaning toward the latter since no baseline measurement exists yet.

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

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-30 | Initial draft | (pending) |
| B | 2026-07-30 | Full six-gate review (24 findings: 3 Critical, 8 High, 11 Medium, 2 Low) incorporated. Added a new `field-array-persistence` Slice that closes both the Goal-delivery gap (no path from "run an experiment" to "see the result") and the Technical-feasibility gap (nothing to re-render) in one move. Corrected the CI-packaging mechanism for `pyvista-headless-ci` and named the required `ci.yml` change explicitly. Split scope into what's deliverable against today's 1D-cavity data vs. PyVista's forward-capacity 3D value proposition. Added a `CONVENTIONS.md` reconciliation section (§4, new) recording the trigger the parent Charter already named. Named the artifact-distribution mechanism given `.outputs/` is gitignored. Added the inherited `autonomy:review` CI-diff carve-out and a Rev-0-ordering constraint relative to the still-draft parent. | (pending re-review) |

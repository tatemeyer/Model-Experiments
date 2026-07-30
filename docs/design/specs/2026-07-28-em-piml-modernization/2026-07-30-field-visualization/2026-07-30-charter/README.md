---
title: "Arc Charter — field-visualization"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-30-field-visualization
slice: 2026-07-30-charter
revision: A
status: draft
date: 2026-07-30
related-slices: [pyvista-headless-ci, field-render-core, plotly-interactive, poc-experiment-rerender]
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
quoted in full in the parent Charter's §3). This Arc closes `mx-viz`'s
current static-PNG-only gap specifically for base-case-vs-predicted EM
field rendering, physics-sim-style — not a live training dashboard in any
form, live or recorded.

## 2. Relationship to the Design Charter

This Arc Charter inherits every constraint in the parent Design Charter,
especially:

- §5's no-hosted/live-viewer-dashboard constraint — every output stays a
  file (PNG/GIF/MP4/self-contained HTML), never a running service.
- §5's CI-stays-CPU-only-unless-explicitly-justified constraint.
- §3's on-the-record license check of PyVista (MIT), Plotly (MIT), and VTK
  (BSD-3-Clause) — already done at Design level; this document doesn't
  redo that check, only records where each dependency actually lands
  (Slice-level `pyproject.toml` change, per §5's "recorded in the Slice PR
  that adds it" requirement).

This document adds only what's specific to `field-visualization`; it
doesn't re-litigate the parent's constraints.

## 3. Scope

- Extend `tools/viz` (`mx-viz`), not create a new package.
  **Recommendation resolving the parent Charter's own install-footprint
  deferral (§3):** add PyVista as an optional dependency group (e.g.
  `mx-viz[3d]`) rather than a hard dependency of `mx-viz` itself — VTK's
  ~150-250MB footprint shouldn't become mandatory for callers who only need
  2D matplotlib output. Plotly is lightweight enough to consider a hard
  dependency, but default to the same optional-group treatment for
  consistency unless a Slice finds a concrete reason to split them (see
  Open questions, §9).
- New capability: linked multi-panel rendering of target field | prediction
  | error, animated over the field's own time/frequency axis.
- Existing `mx-viz` matplotlib functions (`fields.py`, `training.py`) are
  unchanged — this Arc adds capability, it doesn't touch or replace what's
  already there.

## 4. Named Slices and sequencing

| Slice | Scope | Verifiable by |
|---|---|---|
| `pyvista-headless-ci` | Add PyVista as an optional `mx-viz` dependency group; confirm off-screen/headless rendering actually works in this repo's real `ubuntu-latest` CI runner — not just the general PyVista research claim | CI |
| `field-render-core` | Linked target/prediction/error `Plotter`, `open_gif`/`open_movie` export, `export_html` | CI (render doesn't crash) + manual visual check |
| `plotly-interactive` | Plotly wrapper (`Isosurface`/`Volume`/`Streamtube`) for lightweight embeddable figures | CI + manual visual check |
| `poc-experiment-rerender` | Re-render one existing em-piml experiment's results (e.g. a cavity experiment) through the new toolkit end-to-end, as this Arc's own proof the whole path works on real data | Manual — recorded in an experiment write-up |

`pyvista-headless-ci` gates the other three — nothing downstream should
assume the headless-rendering claim holds in this repo's actual CI until
it's independently confirmed here, not just cited from prior research.

## 5. Non-negotiable constraints

- Headless/CI-safe — this Arc doesn't just cite the general
  PyVista-headless-CPU-rendering research finding from the parent Charter,
  it must independently confirm that claim holds in this repo's actual CI
  environment (`pyvista-headless-ci`) before anything downstream depends on
  it.
- No hosted/live viewer — outputs are files only.
- Every dependency this Arc actually adds gets its license recorded in the
  Slice PR that adds it (parent §5), even though the underlying license
  check already happened at Design level (parent §3) — the requirement is
  "recorded in the Slice PR," not just "checked once."

## 6. Cross-cutting gaps / risks specific to this Arc

- **Data contract gap**, similar in shape to the one `mx-viz`'s own
  `CONVENTIONS.md` entry already flags between `mx_viz.io`'s JSON schema
  and `results.csv` (`CONVENTIONS.md:198-203`): target-field and
  predicted-field arrays need matching grid/coordinate metadata to render
  correctly side-by-side, and em-piml's current inference/eval pipeline
  doesn't currently package that metadata alongside the raw array in a
  standard form. `field-render-core` needs to either define this contract
  or confirm one already exists before assuming it away.
- VTK's install footprint (~150-250MB) is real dev-environment weight —
  worth confirming this doesn't meaningfully slow down CI before treating
  it as pulled in by default rather than only for the Slices that actually
  render (consistent with the optional-dependency-group recommendation in
  §3).

## 7. Relationship to the Issue/PR loop

Each Slice above becomes its own Intent Issue + PR once this Arc Charter
reaches Rev-0, per the parent Design Charter's §8 / `docs/design/README.md`'s
process.

## 8. Gates — Rev-A

- [ ] Security
- [ ] License/compliance
- [ ] Technical feasibility
- [ ] Cost/compute-budget
- [ ] Convention-alignment
- [ ] Goal-delivery

Not yet reviewed — this is the first draft.

## 9. Open questions

- Optional-dependency-group split (PyVista only, or Plotly too) —
  recommended in §3, not yet confirmed.
- Whether `poc-experiment-rerender` should pick an experiment already
  canonical elsewhere in this repo (e.g. the R3 long-horizon or
  num-bands-gap experiments) or a simpler one chosen specifically for being
  easy to visually sanity-check.

## 10. Rollback / abandonment path

Per the parent Design Charter's §12: abandoning this Arc before reaching
its own Rev-0 is a lightweight `status: abandoned` change, not a Change
Order.

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-30 | Initial draft | (pending) |

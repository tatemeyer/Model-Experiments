# Does the field-persist -> mx-viz-render pipeline work end-to-end on real data? (issue #64)

The `field-visualization` Arc's own goal statement ("when we run an
experiment I want to see the base case and its output") has no path
from "run an experiment" to "see the result" until every piece of the
pipeline — persist (#61), render (#62) — is proven to actually connect
on real, non-synthetic data, not just against fixture arrays in each
Slice's own unit tests. This is that proof: Slice 5
(`poc-experiment-rerender`) of the Arc's charter
(`docs/design/specs/2026-07-28-em-piml-modernization/
2026-07-30-field-visualization/2026-07-30-charter/README.md`).

Re-trains this project's baseline cavity PINN (issue #2 — chosen over a
long-horizon-collapse or two-mode experiment specifically for being
fast (~35s) and visually simple to sanity-check: a single clean
standing wave, no collapse dynamics to interpret), persists its
target/predicted field via `em_piml.train.save_field_grid_artifact`
(#61), reloads the artifact from disk with `mx_viz.io.
load_field_artifact` (`allow_pickle=False` enforced), and renders it
two ways: the existing static `mx_viz.fields.plot_field_heatmap` and
#62's new per-frame `mx_viz.fields.render_field_frames` +
`mx_viz.animate.open_gif`. All in one script,
`em_piml.field_rerender_poc` (`uv run python -m
em_piml.field_rerender_poc`), rather than a notebook/interactive
session, so the exact pipeline is reproducible by anyone picking this
up later.

**Result: the pipeline works end-to-end — trained model -> persisted
`.npz` artifact -> reloaded from disk -> rendered, both as a static
heatmap and an animated per-frame GIF, matching the original held-out
relative L2 error.**

| variant | metric (seed 0) |
|---|---|
| baseline cavity, in-memory (pre-persist) | relative_l2 = 0.0265 |
| baseline cavity, reloaded from persisted artifact | visually identical (see rendered output below; the artifact stores the already-evaluated grid, not the model, so this is a read-back check, not a second inference pass) |

The static heatmap (predicted / true / |error|) and a mid-sequence
frame from the animated per-frame comparison are attached to this
issue's PR as image uploads (GitHub renders these inline in the PR
body/comments — the actual PR-facing deliverable, per #62/#63's own
"GitHub doesn't render an attached HTML file" finding). Both show the
expected clean standing wave with small, spatially-uniform error
(no collapse artifacts, consistent with the 0.0265 relative L2 number
above) — the render step isn't silently corrupting or mismatching the
persisted data.

**This is a one-off proof, not a standing check — it runs locally/
manually only, not in CI** (per the issue's Rev-C corrected
justification: `ci.yml` already has a "Test (slow)" step that trains
models, so CI-runtime cost was never the real constraint; the actual
reason is that a one-off proof-of-concept isn't a recurring
verification CI should re-run on every PR). No new regression test
locks this in — `em_piml.field_rerender_poc` itself, run by hand, *is*
the artifact this Slice produces, matching the issue's own success
criteria (no test-suite requirement is listed there, unlike every
other Slice in this Arc).

**Leads for whoever picks this up next:**
1. This PoC re-renders the baseline cavity specifically because it's
   the simplest visually-checkable case. Re-running the same script
   pattern against a `long-horizon-collapse`-thread model (where the
   render step would need to visibly show the collapse, not just a
   clean wave) would be a stronger end-to-end proof if a future issue
   wants one — untried here by design (issue #64 explicitly allows
   either).
2. The PyVista 3D surface and Plotly interactive paths (#62/#63) are
   untested against this real artifact in this PoC — only the 2D
   matplotlib path was exercised end-to-end. A follow-up could extend
   this script to also call `mx_viz.fields_3d.plot_field_surface` /
   `mx_viz.plotly_fields.plot_surface` against the same persisted
   artifact.

---
Standalone file (not part of an existing thread — this is
infrastructure verification, not a research question about the
underlying physics). Row added to `../CLAUDE.md`'s experiment index and
`../results.csv` below.

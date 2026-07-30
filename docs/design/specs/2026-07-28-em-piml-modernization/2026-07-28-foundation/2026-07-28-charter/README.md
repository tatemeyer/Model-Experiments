---
title: "Design Charter — em-piml Modernization"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-28-foundation
slice: 2026-07-28-charter
revision: E
status: draft
date: 2026-07-30
related-arcs: [field-visualization, jax-migration, cloud-compute-ops, device-abstraction]
supersedes: null
superseded-by: null
---

# Design Charter — em-piml Modernization

See `docs/design/README.md` for what Design/Arc/Slice, revisions, gates, and
Change Orders mean — this document assumes that vocabulary and doesn't
redefine it.

## 1. Purpose & why now

`projects/em-piml/` — a physics-informed neural network (PINN) research
testbed — has grown into a real body of work: eleven merged experiments as
of this writing (`projects/em-piml/CLAUDE.md`'s experiment index), a
7-issue open backlog, and a plotting toolkit (`mx-viz`) that has already
outgrown static PNGs as the way to actually *see* what these experiments are
doing. Three changes were directed as a combined initiative rather than
three independent Issues. **Correcting an inaccuracy from earlier
revisions** (Technical-feasibility gate finding, see §15): GPU/TPU compute
has never been used not because of the PyTorch/JAX framework choice — PyTorch
has had first-class CUDA/GPU device placement its entire history — but
because `projects/em-piml` has zero device-awareness anywhere (§3,
`device-abstraction`), a gap orthogonal to which ML framework sits on top of
it. `device-abstraction` and `cloud-compute-ops` are technically independent
of `jax-migration` and would deliver real value even if `jax-migration` is
later abandoned (§12) — they are bundled into one Design because they were
directed together and because `jax-migration`'s own case benefits from
GPU/TPU access once it exists, not because any one Arc is technically
blocked without the others. What genuinely doesn't fit this repo's
single-Issue loop is scale, not technical entanglement: three
multi-week changes to a project's core framework, compute, and
visualization stack, each large enough to warrant its own Arc Charter and
Slice sequence. This Charter exists to give that combined initiative a
single, coherent scope and a set of named, sequenceable Arcs before any
implementation starts.

## 2. Relationship to existing repo philosophy

Root `CLAUDE.md`'s "Working rules" section (`CLAUDE.md:89-91`) states: "Do
not add scaffolding, abstractions, or process beyond what the current
Issue's intent requires — this repo is deliberately minimal until real
experiments give it code to run." The `docs/design/specs/` process this
Charter is the first document in is a direct, visible departure from that
rule — a multi-revision, gated, dated-hierarchy spec process is exactly the
kind of scaffolding that rule warns against adding casually.

This is a **deliberate, bounded, owner-directed exception**, not a change to
that rule. It applies only inside this Design's own tree
(`docs/design/specs/2026-07-28-em-piml-modernization/`). Every other project
in this repo, and every future single-Issue-shaped piece of work inside
em-piml itself, keeps using the existing loop unchanged: Intent Issue → PR →
CI → autonomy-label-gated merge (`CLAUDE.md:14-35`). This process exists
because this specific initiative — three multi-week changes to a project's
core framework, compute, and visualization stack, each large enough for its
own Arc Charter — doesn't fit that loop's single-Issue shape, not because
the loop itself is inadequate. **This Charter's inline `(Added in Rev-X...)`
provenance tags, in addition to the full finding tables (§13-onward) and the
Revision History table, are a Charter-specific choice, not something
`docs/design/README.md` requires** — that policy only calls for git history
plus a Revision History table. Future Arc/Slice documents inheriting this
process can drop the inline tags and keep just the finding tables and
Revision History if length becomes a concern at that scope. *(Added in
Rev-D, Convention-alignment gate finding, see §17.)*

## 3. Scope

### field-visualization

**Renamed from `rerun-visualization` in Rev-E, following an explicit owner
course-correction — not a document-completeness gate finding.** The owner's
own words (PR #55 comment, verbatim): "it seems like a good/useful
visualization tool but I don't particularly care about watching graphs on a
dashboard while it trains. The output/inference/predictions are what I care
about. When we run an experiment I want to see the base case (what it is
attempting to predict) when possible and it's output. Overall, less focus
here on a dashboard and closer to a physics-em-sim renderer." This overrides
Rev-D's Rerun.io proposal at the root, not at the margin: the Goal-delivery
gate's Rev-D Critical finding (§18, finding 1 — recording-only mode may
under-deliver "experience the experiment with my eyes") is resolved by this
course-correction directly, not by choosing between the two options §11
offered. The owner declined both and redirected the arc's entire toolkit
choice instead.

Dedicated research (this Charter's own research process, not a formal gate
review) surveyed EM-field-capable rendering toolkits against the
base-case-vs-predicted-field framing above, and found the strongest signal
in what actual EM/FDTD simulators do for this exact data shape: Meep
(`github.com/NanoComp/meep`) renders field slices with matplotlib
(`imshow`, diverging colormap, `Animate2D` → GIF/MP4); gprMax and openEMS
write VTK files and open them in ParaView. Neither builds a live metrics
dashboard — that finding is the strongest available evidence for what
actually fits this data shape.

This arc commits to:
- **PyVista** (`github.com/pyvista/pyvista`, MIT) as the primary
  3D/volumetric renderer — the same VTK stack gprMax/openEMS already
  target. Confirmed headless-CPU-capable (off-screen rendering works with
  the stock PyPI `vtk` wheel as of VTK 9.5, no Xvfb or custom OSMesa build
  required) and NumPy/JAX-array-native (`np.asarray()` is the entire
  bridge). Emits PNG, GIF, MP4, and self-contained interactive HTML
  (`export_html`) — every format this repo's PR-review workflow needs, with
  no viewer app required to inspect a result. The natural
  base-case-vs-predicted pattern is a linked multi-panel `Plotter` (target
  field | prediction | error), animated over the field's own time/frequency
  axis via `open_gif`.
- **Plotly** (MIT) for lightweight, rotatable interactives that need to drop
  directly into a PR with zero viewer install (`write_html`, self-contained;
  `Isosurface`/`Volume`/`Streamtube` trace types cover the EM-field
  vocabulary) — used selectively for hero figures, not as the default.
- **matplotlib**, via the existing `mx-viz` (`tools/viz/src/mx_viz/`), kept
  as-is for 2D field slices and magnitude/phase maps — this is literally
  Meep's own approach, and there's no reason to reach for 3D machinery to
  render a 2D slice.

**Rerun.io is dropped from this Design entirely**, not merely descoped to a
narrower use case. Re-evaluated specifically against the
base-case-vs-predicted-field framing (not the metrics-dashboard framing
that originally motivated ruling out its live mode, §5/§13), it still
doesn't clear the bar, for two independent reasons: it has no isosurface,
volume-rendering, or streamline primitives (its `Tensor`/`Image`/`Mesh3D`/
`Points3D`/`Arrows3D` archetypes cover slices and point clouds, not the
physics-sim vocabulary the owner asked for), and there is no supported path
from a recording to a static image for a PR — a public feature request
asking for exactly that ("documenting experimental outcomes,"
"publication-ready figures") was closed upstream as not planned. §5's
Rerun-specific recording-only-mode enforcement bullet and §11's escalated
owner question are both retired by this rename, not carried forward.

**This also resolves Rev-D's Technical-feasibility finding** (§15, finding
2) about `train.py` instrumentation scope, rather than just restating it
under a new name: rendering target-vs-predicted fields is a post-hoc/batch
operation over already-materialized arrays — the same shape as `mx-viz`'s
existing `fields.py`/`training.py` functions — not a per-step live-logging
concern requiring changes inside the training loop itself. If time-scrubbing
a field's evolution *during* training becomes a wanted feature later, that
is a new, explicitly-scoped capability for a future revision, not something
this rename silently assumes.

**License check of this arc's own primary new dependencies (§5's
dependency-vetting bar, exercised on the record):** PyVista is MIT
(`github.com/pyvista/pyvista` `LICENSE`); Plotly is MIT (upstream
`LICENSE.txt`); VTK itself (PyVista's compiled dependency) is
BSD-3-Clause. No `jaxpi`-style setup.py/LICENSE mismatch found in any of the
three. All strictly more permissive than Rerun's already-clean MIT/Apache-2.0
dual license — this rename removes an entire dependency's worth of license
surface, it doesn't add any.

**Install-footprint note for this arc's own Arc Charter to size:**
`pyvista[all]` pulls in compiled VTK, roughly 150-250MB — larger than
matplotlib's existing footprint in `mx-viz`. Whether this warrants a
separate `tools/<name>` package or extending `mx-viz` in place is left to
`field-visualization`'s own Arc Charter, consistent with §5's `tools/<name>`
placement requirement already established for `cloud-compute-ops`/
`device-abstraction` tooling.

### jax-migration

Commit to JAX as em-piml's ML framework for new development, replacing
PyTorch where the migration proves tractable — softened from "replacing
PyTorch" unconditionally (Technical feasibility gate finding, see §15),
since §7's SOAP/L-BFGS gap means full unconditional replacement may not be
achievable; see §7 for the candidate resolution paths this qualifier is
standing in for. The most
relevant existing reference implementation is `jaxpi`
(`github.com/PredictiveIntelligenceLab/jaxpi`) — the Perdikaris lab's own
JAX PINN library, and directly relevant because this repo has already
hand-implemented several of that same lab's papers in PyTorch (causal
loss-reweighting, NTK-based adaptive reweighting — both in
`projects/em-piml/LITERATURE.md`, both found not to transfer to em-piml's
long-horizon-collapse target) and has two more queued unimplemented (issue
#39, Random Weight Factorization; issue #41, PirateNets).
**`jaxpi` is reference-only, not a dependency**: its `setup.py` claims
Apache 2.0, but its actual `LICENSE` file is a non-standard Penn
non-commercial-research license that forbids redistribution without Penn's
written approval. This repo will study `jaxpi`'s implementations of causal
weighting, RWF, PirateNets, and NTK reweighting for correctness, then
reimplement the underlying papers' techniques directly in this repo's own
JAX code — the same pattern this repo already uses for every other paper it
tests, and it sidesteps the license question entirely. **"Reimplement" means
deriving from the cited papers' equations/algorithms — `jaxpi`'s source may
be read for correctness verification but must not be copied, adapted
line-by-line, or have its comments/variable/function names carried into
this repo's code; a Slice PR that reimplements RWF, PirateNets, or NTK
reweighting must cite the paper, not `jaxpi`'s file/function, as its source
of truth.** *(Added in Rev-C, License/compliance gate finding — see §14.)*
This is a narrower outcome than the original directive's "potentially even
building on top of it" phrasing — an explicit, owner-approved narrowing
(via a clarifying question, prior to this Charter), recorded here for the
historical record, not an unreviewed drift. *(Added in Rev-D, Goal-delivery
gate finding, see §18.)* `train.py` (1294 lines, every training-loop
variant across ~10 experiments) is the dominant migration surface.

**License check of this arc's own primary new dependencies (§5's
dependency-vetting bar, exercised on the record here rather than deferred
entirely):** `jax`/`jaxlib` (`google/jax`), `optax`
(`google-deepmind/optax`), `equinox` (`patrick-kidger/equinox`), and
`diffrax` (`patrick-kidger/diffrax`) were each checked against their actual
upstream `LICENSE` file, not just package metadata — all four are genuine
Apache License 2.0, no `jaxpi`-style setup.py/LICENSE mismatch found. *(Added
in Rev-C, License/compliance gate finding — see §14.)*

### cloud-compute-ops

Set up and maintain remote/cloud GPU or TPU compute from verifiable,
legitimate sources, ranked (highest priority first) by prior research in
this session: Google Colab via its new official `google-colab-cli`
(scriptable, T4 GPU, no eligibility gate); Kaggle Notebooks (~30 GPU-hrs and
~20 TPU-hrs/week, API-scriptable); Google Cloud TPU Research Cloud (real
GCE-based TPU quota, open to independent researchers, application-gated);
NSF ACCESS "Explore" tier (fast approval, independent-researcher eligibility
not fully confirmed). AWS/Azure research-credit programs were ruled out —
they require institutional or startup affiliation this repo's context
doesn't establish.

This arc's near-term value depends on workload size: em-piml's own
documented training runs are typically 35-500s on CPU
(`projects/em-piml/CLAUDE.md`, various experiment write-ups), and cloud
session setup/teardown overhead may be comparable to or larger than the
compute time saved at that scale. `cloud-compute-ops`'s Arc Charter must
state, on the record, either a phased rollout (start with the
single highest-ranked provider, add the others only once `device-abstraction`/
`jax-migration` produce workloads whose runtime materially exceeds this
baseline) or an explicit reason all four providers are needed immediately.
*(Added in Rev-D, Technical feasibility and Cost/compute-budget gate
findings, see §15 and §16.)*

The original directive said "**setup and maintain**" — this Charter's
treatment so far (§5, §7) covers setup thoroughly (provider selection,
credential hygiene, Terms-of-Service-vs-license distinction, rotation/
revocation) but not ongoing maintenance. `cloud-compute-ops`'s Arc Charter
must additionally define: a recurring cadence for checking quota
consumption against each provider's stated free-tier limits; a recurring
cadence for re-confirming provider Terms of Service/pricing haven't changed
since adoption; and an explicit contingency if a provider's free program is
discontinued or materially changes terms mid-Design. *(Added in Rev-D,
Goal-delivery gate finding, see §18.)*

Any credential-check or provisioning script this arc produces that isn't
specific to a single training run must be scoped as a `tools/<name>`
workspace package following the `tools/datasets`/`tools/viz` pattern (see
`tools/README.md`), not an ad hoc script inside `projects/em-piml/` —
`cloud-compute-ops`'s Arc Charter must decide the package name and check
`tools/README.md` before creating one. *(Added in Rev-D,
Convention-alignment gate finding, see §17.)*

### device-abstraction (prerequisite arc — research-surfaced, not owner-named)

**This arc was not named in the original directive; it is a prerequisite
this Charter is recommending be tracked explicitly, pending confirmation
(see "Open questions," §11).** A repo-wide grep found zero device-awareness
anywhere in `projects/em-piml/src/em_piml/` today — no `.cuda()` call, no
`torch.device` construction, no CLI/env flag for device selection. Every
tensor is implicitly CPU. Both `jax-migration` and `cloud-compute-ops` need
a real device-abstraction layer to mean anything in practice; without it,
"commit to JAX" and "set up cloud GPU access" are each individually
incomplete. Whether this layer belongs in `projects/em-piml/src/em_piml/`
(project-scoped, since it's core training-loop logic) or `tools/` is
currently unstated either way and should be decided by this arc's own Arc
Charter rather than left ambiguous. *(Added in Rev-D, Convention-alignment
gate finding, see §17.)* All data trained on under any Arc in this Design is synthetic
or analytically-generated in-repo (via `mx-data`, e.g.
`em-piml-1d-cavity-analytical`) — no sensitive, proprietary, or personally
identifiable data is shipped to third-party rented compute under
`cloud-compute-ops`. If a future experiment ever needs non-synthetic data,
that changes this constraint and must be re-flagged as a Security-gate item
before compute is provisioned. *(Added in Rev-B, Security gate finding —
see §13.)*

## 4. Reconciliation with existing `CONVENTIONS.md` entries

**Compute assumption** (`CONVENTIONS.md:44-51`, 2026-07-14): "CPU primarily,
with an optional single consumer GPU ... and free-tier cloud only
(Colab/Kaggle-class, no paid rented compute). Don't default to multi-GPU,
large-batch, or paid-cloud-only designs; note explicitly in a project's
`CLAUDE.md` if it needs more than this." The `cloud-compute-ops` ranking
above (Colab, Kaggle, TRC, NSF ACCESS) is consistent with this entry as
written — no convention change is needed, only the explicit
`projects/em-piml/CLAUDE.md` note this entry itself already calls for, once
`cloud-compute-ops` actually starts. **Qualifying this conclusion**
(Cost/compute-budget gate finding, see §16): this entry's "free-tier cloud
only" language was written before any scripted/automated cloud-compute
usage existed in this repo and does not itself distinguish occasional
manual use from `cloud-compute-ops`'s planned scripted/repeated automation
(`google-colab-cli`). Staying within the letter of "free-tier cloud only"
does not by itself establish that regular automated use carries the same
negligible cost-risk profile as occasional manual use — the monitoring/
budget-alert controls named in §7 are a required companion to, not a
substitute for, this convention-alignment conclusion, and `cloud-compute-ops`'s
Arc Charter should independently confirm whether the convention itself
should later gain a new dated entry addressing automated/scripted usage
specifically, once real usage patterns are known.

**Optimizer default** (`CONVENTIONS.md:87-102`, 2026-07-15): "Default is
still `torch.optim` only... ["]when a specific research result names an
optimizer not in `torch.optim`... and a small, actively-maintained,
narrowly-scoped PyPI package implements it faithfully... adopting it is
preferable[."]" This entry's own anchor point (`torch.optim` as the
baseline) becomes literally inaccurate once `jax-migration` replaces
PyTorch, and this Design's own §7 already engages its subject matter at
length (SOAP, "this repo's single most load-bearing optimizer result") —
but this entry was omitted from reconciliation in earlier revisions
(Convention-alignment gate finding, see §17). Same trigger mechanism as the
ML-framework entry below: once `jax-migration`'s Arc Charter resolves the
SOAP/L-BFGS gap named in §7 (via a JAX port, a reimplementation, or an
accepted, documented capability loss), record the outcome as a new dated
`CONVENTIONS.md` entry superseding the 2026-07-15 one, not as a side effect
of this Design Charter reaching Rev-0.

**Plotting default** (`CONVENTIONS.md:177-203`, 2026-07-27): "matplotlib is
the default plotting library for this repo... Why matplotlib over
plotly/bokeh/altair: this repo reports findings as static content embedded
in PR bodies and experiment-log markdown, not a hosted/interactive
dashboard... it carries no GPU/CUDA-adjacent dependency risk." `field-visualization`'s
Rev-E commitment to PyVista and Plotly (§3) touches this entry directly and
must be reconciled here, not left implicit — this is the same class of gap
the Convention-alignment gate caught for the optimizer-default entry in
Rev-D (§17), and this Charter is applying that lesson to itself proactively
rather than waiting for a future gate to catch it again. Two distinctions
keep this addition consistent with the entry's original reasoning rather
than silently contradicting it: (1) Plotly's role here is `write_html`-exported,
self-contained files attached to a PR or experiment write-up — the same
static-artifact posture the entry already commits to, not the
hosted/interactive dashboard the entry explicitly ruled out; (2) PyVista's
dependency is VTK's CPU-based off-screen rendering (confirmed headless-capable
with the stock PyPI wheel, no Xvfb or custom build), not a CUDA/GPU runtime
dependency — the entry's "no GPU/CUDA-adjacent dependency risk" concern was
about ML-framework-class GPU deps bleeding into a plotting tool, which
doesn't apply to VTK's software-rasterization path. What the entry's
reasoning does not anticipate: matplotlib has no 3D isosurface/volume-rendering/
streamline capability at all, which is the actual gap `field-visualization`
exists to close — this isn't picking a fancier tool for the same job, it's a
capability matplotlib structurally lacks. Same trigger mechanism as the
other two entries here: once `field-visualization`'s Arc Charter reaches
Rev-0, record a new dated `CONVENTIONS.md` entry that *extends* (not
supersedes) the 2026-07-27 entry — matplotlib remains the default for 2D;
PyVista/Plotly are additions scoped specifically to 3D/volumetric field
rendering — not as a side effect of this Design Charter reaching Rev-0.

**ML framework default** (`CONVENTIONS.md:53-60`, 2026-07-14): "Unless a
project's issue says otherwise, default to PyTorch... Revisit per-project if
a project's research question specifically benefits from JAX (e.g. needing
to differentiate through a JAX-backed simulator)." This entry already
anticipates and permits exactly this kind of per-project override — it does
not need to be superseded, only exercised. The trigger for actually
recording that exercise as a new dated `CONVENTIONS.md` entry (per this
file's own "add a new entry rather than silently editing history" rule,
`CONVENTIONS.md:3-7`) is: **once the `jax-migration` arc's own Arc Charter
reaches Rev-0**, not as a side effect of this Design Charter reaching Rev-0.
Stating that trigger here now, so it isn't ambiguous later.

## 5. Non-negotiable constraints

Every Arc must either honor these or explicitly justify an exception in its
own Arc Charter:

- Compute stays CPU-primarily / free-tier-cloud-only (§4) unless an Arc
  explicitly justifies paid compute. **This bullet was pure stated intent
  with no enforcement mechanism through Rev-C** (Cost/compute-budget gate
  finding, see §16) — the same gap the hosted/live-viewer-dashboard bullet
  below was already upgraded past in Rev-B, when it was still Rerun-specific.
  `cloud-compute-ops`'s Arc Charter must specify,
  before any provider integration lands: for GCP/TPU Research Cloud, a
  Cloud Billing budget cap with a near-zero-threshold alert, since TRC
  requires a real GCP project with a linked billing account and an
  IAM-scoped service account (§7) does not by itself prevent provisioning a
  billable, non-TRC-covered resource (persistent disk, a non-preemptible
  instance, egress beyond the free allowance); for Kaggle/Colab, a stated
  understanding (confirmed from current provider documentation, not
  assumed) of what happens on quota exceedance, a practice of logging
  cumulative GPU/TPU-hours consumed per run against the stated weekly
  quota, and an explicit rule that no payment method is ever attached to
  any account used under this Arc, so quota exceedance cannot silently
  convert into billing.
- No hosted/live-viewer dashboard as a required artifact for any
  visualization tooling. `field-visualization`'s renderer choices
  (PyVista/Plotly/matplotlib, §3) satisfy this natively: every output is a
  file (PNG/GIF/MP4/self-contained HTML) produced by a local, headless
  process, not a running service — there is no live-mode equivalent to
  enforce against, unlike Rerun's. *(Original bullet added in Rev-B,
  Security gate finding, targeted at Rerun's live-streaming mode
  specifically — see §13; retargeted in Rev-E following the
  `rerun-visualization` → `field-visualization` rename, §3, which removed
  the live-mode risk this bullet was enforcing against rather than just
  restating it.)*
- Docs stay agent-first everywhere else in the repo (`CLAUDE.md:77-81`) —
  this spec tree is an explicit, bounded exception to that terseness bias,
  confined to `docs/design/specs/`.
- The existing PR-gating CI (`.github/workflows/ci.yml`, single `ubuntu-latest`
  job, no GPU runner, no matrix) stays CPU-only and unchanged unless an Arc
  explicitly proposes and justifies a change to it.
- **Any PR that adds, modifies, or references a GitHub Actions secret,
  Environment, or CI workflow file must carry `autonomy:review`, never
  `autonomy:safe`, regardless of how small the diff looks** — this is a
  standing exception to normal autonomy-labeling and applies for the life
  of this Design, not just this document's current revision, because
  `auto-merge.yml` merges on
  green CI with zero human review, and a mislabeled small-looking diff to
  CI/secrets territory would otherwise land on `main` unreviewed. *(Added
  in Rev-B, Security gate finding — see §13.)*
- **Every new third-party dependency added under this Design** (any
  ecosystem — PyPI, cloud provider CLIs) gets the same
  license-file-not-just-metadata check applied to `jaxpi` in §3, recorded
  in the Slice PR that adds it. Unofficial/community forks or ports of a
  library (e.g. a JAX port of `pytorch_optimizer.SOAP`, see §7) require an
  explicit maintenance/provenance note **and** must independently pass the
  same license-file check as any other dependency — the provenance note is
  additive to that check, not a substitute for it. Apache-2.0's
  NOTICE-preservation clause (relevant to most of
  the JAX ecosystem) is not currently triggered — no Arc redistributes
  dependency source; revisit if any Arc's output is ever packaged/published
  standalone. *(Added in Rev-B, Security gate finding, see §13; fork-bullet
  wording tightened and NOTICE note added in Rev-C, License/compliance gate
  finding, see §14.)*
- **A software license check is not the same thing as a service's Terms of
  Service.** `cloud-compute-ops`'s Arc Charter must separately record each
  provider's Terms of Service / Acceptable Use Policy constraints relevant
  to scripted/automated use — in particular, Colab's free-tier terms have
  historically restricted bypassing the UI for automated use, which is in
  direct tension with `google-colab-cli`'s core value proposition (scripted,
  headless GPU provisioning) named in §3. Checking that `google-colab-cli`
  itself is permissively licensed does not substitute for checking whether
  *using* it the way this Design intends is consistent with Colab's actual
  service terms. *(Added in Rev-C, License/compliance gate finding — see
  §14.)*
- **`cloud-compute-ops`'s Arc Charter must specify a local-credential
  convention before any provider integration lands**: extend `.gitignore`'s
  Secrets section to cover each provider's actual credential filename
  pattern (e.g. `kaggle.json`, `*-service-account*.json`,
  `client_secrets.json` — none of which the current `.env`/`*.pem`/`*.key`
  patterns catch), and state explicitly that credentials are never pasted
  into a notebook cell that gets committed or exported into
  `docs/design/` or `projects/em-piml/experiments/`. *(Added in Rev-B,
  Security gate finding — see §13.)*
- Everything outside this Design's own tree is untouched by this process
  (§2).

## 6. Named arcs and sequencing

| Arc | Scope | Depends on |
|---|---|---|
| `device-abstraction` | Real device selection/abstraction in em-piml's training code (currently zero) | none — prerequisite for the other three |
| `jax-migration` | Replace PyTorch with JAX; reimplement (not depend on) `jaxpi`'s techniques | `device-abstraction` for GPU/TPU to matter in practice |
| `cloud-compute-ops` | Set up and maintain legitimate free/low-cost GPU/TPU access | `device-abstraction` to actually use the compute once accessed |
| `field-visualization` | PyVista (primary, 3D/volumetric) + Plotly (embeddable interactives) + existing matplotlib/`mx-viz` (2D) for base-case-vs-predicted EM field rendering | largely independent of the other three |

`field-visualization` can proceed in parallel with the others — it doesn't
depend on JAX or cloud compute, and vice versa. `device-abstraction` is the
one clear blocking prerequisite: both `jax-migration` and `cloud-compute-ops`
are individually incomplete without it.

**Sequencing caveat** (Technical feasibility gate finding, see §15):
`device-abstraction`'s natural first implementation is PyTorch-idiom
(`.to(device)` calls, `device=` threaded through tensor-construction call
sites — the same pattern this repo already used for `dtype` in the FP64
precision experiment). JAX's device model is structurally different
(functional placement via `jax.device_put`, largely automatic backend
selection) — there is no `.to(device)` equivalent to migrate. If
`device-abstraction` starts before `jax-migration` lands, its PyTorch-idiom
layer should be treated as an interim measure for the current codebase, not
code `jax-migration` inherits wholesale; `jax-migration`'s own Arc Charter
must specify device placement in JAX's own idiom rather than assuming the
earlier layer transfers.

## 7. Cross-cutting gaps surfaced by research (surfaced, not yet assigned)

- **No mature JAX equivalent of `pytorch_optimizer.SOAP`** (only an
  unofficial third-party port exists — subject to §5's provenance-note
  requirement, since it's exactly the "unofficial fork" case that bullet
  names) or of `torch.optim.LBFGS` with `strong_wolfe` line search
  (`jaxopt` is sunset; `optax.lbfgs` and `optimistix.BFGS` use different,
  unverified-equivalent line-search strategies). SOAP is this repo's single
  most load-bearing optimizer result — it fully closed the `num_bands=4`
  instability
  (`projects/em-piml/experiments/num-bands-gap/011-soap-optimizer.md`).
  `jax-migration`'s Arc Charter needs to address this directly, not assume
  it away. Candidate resolutions to evaluate, not treated as exhaustive
  (Technical feasibility gate finding, see §15): (a) a hybrid outcome where
  SOAP-optimized training stages remain on PyTorch while the rest of the
  stack moves to JAX, (b) an interop layer calling PyTorch SOAP from JAX
  code, (c) accepting a different JAX-native optimizer and re-validating
  against `011-soap-optimizer.md`'s bar. Full abandonment (§12) remains the
  fallback if none prove tractable.
- **No correctness/parity-verification strategy exists for the
  PyTorch-to-JAX reimplementation.** Distinct from the license-driven
  "reimplement, don't copy" requirement (§3): reimplementing correctly and
  reimplementing *verifiably* correctly are different problems, and only
  the license angle was addressed through Rev-C. A silent regression in a
  reimplemented technique (e.g. SOAP no longer closing the `num_bands=4`
  instability) would be a research-integrity failure invisible until much
  later. `jax-migration`'s Arc Charter must state how each reimplemented
  technique (causal loss-reweighting, NTK reweighting, RWF, PirateNets, the
  SOAP-equivalent from the bullet above) will be checked against its
  existing PyTorch-validated result before that result is treated as still
  holding — at minimum, re-running the `num_bands=4` SOAP experiment under
  the JAX port and confirming the instability stays closed. *(Added in
  Rev-D, Goal-delivery gate finding, see §18.)*
- **Credentials/secrets for cloud compute are unwired.** The pattern exists
  (`.github/SETUP.md:208-224`: GitHub Environments + narrowly-scoped
  Secrets, "nothing needed yet") but `cloud-compute-ops` will be the first
  thing in this repo to actually need it (`.github/SETUP.md:153-158`
  similarly flags Environments as "not needed yet... revisit if/when a
  project publishes something" — cloud-compute credential scoping is a
  reasonable trigger for that revisit too). `cloud-compute-ops`'s Arc
  Charter must state, per provider, the minimum viable credential scope
  (e.g. a GCP service account restricted to the specific TRC project, not
  a broad project-editor role) and a revocation step (where in each
  provider's console a leaked credential gets revoked) before any
  credential is actually issued — see also §5's local-credential-convention
  and autonomy-labeling bullets. *(Rotation/revocation requirement added in
  Rev-B, Security gate finding — see §13.)*
- **CI implications are undecided.** Does GPU/cloud-compute work stay
  entirely outside the existing PR-gating CI (interactive/manually
  triggered only), or does a new, separate, non-blocking workflow get
  added? Not decided by this Charter — assigned to `cloud-compute-ops`'s
  Arc Charter, which inherits §5's `autonomy:review`-for-CI/secrets-diffs
  carve-out regardless of which way this is decided, and which is exactly
  where §5's provider-Terms-of-Service bullet bites hardest — scripted CI
  usage of a free-tier provider is the case most likely to run against a
  ToS automation restriction. `device-abstraction`'s own Arc Charter must
  separately state its verification story, independent of whatever
  `cloud-compute-ops` decides: CPU-path correctness (defaults to CPU,
  doesn't error when no accelerator is present) is CI-verifiable and should
  be; actual GPU/TPU-selection correctness requires hardware CI doesn't
  have, and should be verified manually/interactively with the result
  recorded in that Arc's own experiment write-up, per this repo's existing
  determinism-verification convention. Any new tests either Arc adds that
  require actual GPU/TPU hardware or live cloud credentials must still live
  colocated per `CONVENTIONS.md`'s testing entry (`projects/em-piml/tests/`,
  no top-level `tests/` tree) and be excluded from the default fast/slow CI
  split via a new marker (e.g. `@pytest.mark.gpu`) rather than reusing
  `slow` for a fundamentally different exclusion reason (hardware
  unavailability, not runtime). *(Device-abstraction CI story and
  GPU-marker requirement added in Rev-D, Technical feasibility and
  Convention-alignment gate findings, see §15 and §17.)*

## 8. Relationship to the Issue/PR loop

This spec tree is upstream design authority, not a parallel tracking
system. When a Slice document under any of these Arcs is ready to be built,
the work still goes through this repo's normal loop unchanged: an Intent
Issue linking back to that Slice document, a PR implementing it, CI as the
source of truth for done, autonomy-label-gated merge (`CLAUDE.md:14-35`).
No epics are introduced into the Issue tracker. A PR delivering a Design or
Arc Charter revision (as opposed to a Slice implementation, like this PR)
is exempt from `.github/pull_request_template.md`'s `## Intent`/`##
Autonomy` sections, since no Issue or autonomy label exists yet at this
stage — state this explicitly in the PR body rather than leaving those
sections silently absent. *(Added in Rev-D, Convention-alignment gate
finding, see §17.)*

## 9. Process recap

See `docs/design/README.md` for the full definition of the Design/Arc/Slice
hierarchy, the Rev-A → Rev-0 lifecycle, Change Orders, the six review
lenses, and a glossary.

## 10. Gates — Rev-E

- [x] Security — cleared in Rev-B against Rev-A (§13). Re-affirmed for
      Rev-E's scope change: `field-visualization`'s new dependencies
      (PyVista, Plotly) introduce no new secrets/credential surface and no
      live/hosted process — if anything, a smaller security surface than
      the Rerun proposal they replace, since there is no live-mode call to
      guard against in the first place (§5).
- [x] License/compliance — cleared in Rev-C against Rev-B (§14). Re-affirmed
      for Rev-E: PyVista (MIT), Plotly (MIT), and VTK (BSD-3-Clause) each
      independently checked against their actual upstream `LICENSE` file,
      not just metadata, consistent with the standard this Charter set with
      the `jaxpi` catch. No new risk introduced — see §3's license-check
      paragraph.
- [x] Technical feasibility — cleared in Rev-D against Rev-C (§15). Rev-E's
      rename resolves finding 2 (the `train.py`-instrumentation scope
      question) outright rather than restating it: base-case-vs-predicted
      rendering is a post-hoc/batch operation, the same shape as `mx-viz`'s
      existing functions, not a live-logging concern. The gprMax/openEMS/Meep
      precedent found during research (§3) is additional positive evidence
      the approach is proven in this exact domain.
- [x] Cost/compute-budget — cleared in Rev-D against Rev-C (§16). Re-affirmed
      for Rev-E: PyVista's off-screen rendering is confirmed CPU-only/headless
      (no paid rendering service, no GPU dependency); Plotly and matplotlib
      render client-side/CPU respectively. No change to this Design's
      compute posture.
- [x] Convention-alignment — cleared in Rev-D against Rev-C (§17). Re-affirmed
      for Rev-E: `field-visualization` extends the existing `tools/viz`
      (`mx-viz`) package rather than inventing a new one, and §4 now
      proactively reconciles the 2026-07-27 plotting-default `CONVENTIONS.md`
      entry against this rename — the same class of gap this gate caught for
      the optimizer-default entry in Rev-D, addressed here before being
      found rather than after.
- [x] Goal-delivery — the Rev-D Critical finding (§18, finding 1) is
      resolved: not by choosing between §11's two offered options, but by an
      explicit owner course-correction (PR #55 comment, quoted in §3) that
      redirects the arc's entire toolkit choice toward what was actually
      asked for — base-case-vs-predicted field rendering, physics-sim-style,
      not a training dashboard in any form, live or recorded.

Reviewer notes: all six gates now show cleared for Rev-E. **This is a
lighter-weight self-check by the document's own author against this
revision's specific scope change (a tool substitution backed by the
research cited in §3), not a fresh independent dedicated-agent review of the
kind Rev-B through Rev-D each received.** Per `docs/design/README.md`'s
lifecycle rule, a revision clearing every gate is eligible to become Rev-0 —
but given that five of these six clearances rest on a self-check rather than
an independent pass, promoting straight to Rev-0 off this revision is a
judgment call for the owner, not something this revision claims
unilaterally. Recommend either (a) accepting this self-check and promoting
to Rev-0 directly, or (b) running a fresh independent gate pass (at minimum
License/compliance and Convention-alignment, since they touch concrete new
dependencies) before promotion — at the owner's discretion.

## 11. Open questions — none decided by this Charter, deferred to a future revision or Arc Charter

- ~~Owner decision required, escalated by the Goal-delivery gate (§18,
  Critical finding): does `.rrd` record-then-replay/scrub satisfy "I want to
  be able to experience the experiment with my eyes," or does it
  under-deliver?~~ **Resolved in Rev-E** — see §3 (`field-visualization`).
  The owner declined both options this bullet originally offered (keep
  Rerun recording-only vs. a scoped Rerun live-viewer exception) and
  redirected the arc's entire toolkit choice instead, toward
  PyVista/Plotly/matplotlib. Full original text preserved via git history
  (Rev-D) for the audit trail.
- Confirm `device-abstraction` as its own Arc versus folding it into
  `foundation` — this Charter recommends a standalone Arc since both
  `jax-migration` and `cloud-compute-ops` depend on it directly, but it was
  research-surfaced, not owner-named, and should be confirmed rather than
  assumed.
- CI-scope decision (§7): fully outside existing CI, or a new
  non-blocking workflow.
- NSF ACCESS "Explore" tier eligibility for a fully independent (non
  institutionally-affiliated) researcher — not independently confirmed by
  prior research. Independent of eligibility: also confirm whether
  ACCESS Explore-tier allocations carry any cost-recovery, overage-billing,
  or institutional-billing-arrangement dimension before this provider is
  used — not assumed cost-free solely because it's described as an
  "Explore" tier. *(Cost dimension added in Rev-D, Cost/compute-budget gate
  finding, see §16.)*
- Sequencing: should `device-abstraction` fully complete before
  `jax-migration`/`cloud-compute-ops` start, or can they proceed with a
  minimal device-abstraction slice first and iterate?
- **Repo-wide gap, out of this Charter's scope to fix, surfaced by the
  License/compliance gate:** this repo has no `LICENSE` file at all
  (`licenseInfo: null` on a public GitHub repo) and no `license` field in
  any workspace `pyproject.toml`. This predates this Design and applies to
  every existing dependency (`torch`, `numpy`, `matplotlib`,
  `pytorch-optimizer`) already, not something this initiative created — but
  this Design's own dependency-vetting bar (§5) implicitly assumes "does a
  dependency's license permit this repo's use" is a well-posed question,
  which is harder to reason about precisely without a stated repo license.
  Recommend tracking "state this repo's license terms" as a separate,
  repo-wide Intent Issue, not something `docs/design/` resolves. *(Added in
  Rev-C, License/compliance gate finding — see §14.)*

## 12. Rollback / abandonment path

Any Arc under this Design may be abandoned before reaching its own Rev-0 —
this is a legitimate, expected outcome (e.g. if the SOAP/L-BFGS optimizer
gap in §7 turns out unbridgeable for `jax-migration`), recorded as a
lightweight `status: abandoned` change at the Arc level, not a Change Order.
Change Orders exist only to amend content that already reached Rev-0.

## 13. Security review findings (Rev-A → Rev-B)

Performed by a dedicated Security-gate review agent, back-tracing from each
Arc's envisioned finished state to what Rev-A actually specified. Full
justification for each finding lived in the review itself; this table is
the permanent audit trail of what was found and where it's now addressed.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | No carve-out from `autonomy:safe` auto-merge for CI/secrets-touching diffs — a mislabeled small diff could merge to `main` with zero human review | Critical | §5 (autonomy:review carve-out bullet) |
| 2 | No local-credential-hygiene convention for `cloud-compute-ops` providers (`kaggle.json`, GCP service-account JSON not covered by current `.gitignore` patterns) | High | §5 (local-credential-convention bullet) |
| 3 | Dependency-vetting rigor applied to `jaxpi` (§3) doesn't generalize to other new packages this Design adds | High | §5 (dependency-vetting bullet), §7 (SOAP bullet cross-reference) |
| 4 | No rotation/revocation story if a cloud credential leaks | Medium | §7 (credentials bullet) |
| 5 | Rerun "recording-only" constraint is stated intent with no enforcement mechanism | Medium | §5 (Rerun bullet, enforcement mechanism) |
| 6 | Training-data sensitivity for rented compute never explicitly confirmed in writing | Low | §3 (`device-abstraction` paragraph) |

Overall Security-gate verdict on Rev-A (verbatim from the review): none of
the findings required rethinking `cloud-compute-ops`'s scope itself — the
provider choices are sound and consistent with `CONVENTIONS.md`'s compute
assumption — they required the handful of additional constraints now
incorporated above.

## 14. License/compliance review findings (Rev-B → Rev-C)

Performed by a dedicated License/compliance-gate review agent, which
independently fetched and read the actual `LICENSE` files of Rerun and the
core JAX-ecosystem packages (not just trusting stated metadata — the same
standard §3's `jaxpi` catch already set) and researched the relevant cloud
providers' Terms of Service.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | `jaxpi` "reference-only" boundary never prohibited copying its actual code/comments/naming — only stated "reimplement" | High | §3 (`jax-migration`, new sentence) |
| 2 | Cloud provider Terms of Service (distinct from software licenses) never addressed — Colab's free-tier terms have historically restricted the scripted automation `google-colab-cli` exists to enable | High | §5 (new ToS bullet), §7 (CI-implications cross-reference) |
| 3 | Repo has no `LICENSE` file at all — pre-existing, repo-wide, out of this Charter's scope to fix, but was a silent assumption underpinning §5's dependency-vetting bar | High | §11 (tracked as a deferred, separately-owned open item) |
| 4 | §5's dependency-vetting bar had never been exercised on the record against `jax-migration`'s own primary dependencies (only ever applied to `jaxpi`, which got rejected) | Medium | §3 (`jax-migration`, on-the-record license check of `jax`/`jaxlib`/`optax`/`equinox`/`diffrax`) |
| 5 | Unofficial-fork provenance-note requirement didn't also require the fork itself to pass a license check | Medium | §5 (fork bullet tightened) |
| 6 | Rerun's license stated as "Apache-2.0" only; actually dual-licensed MIT/Apache-2.0 | Low | §3 (`rerun-visualization`) |
| 7 | Apache-2.0 NOTICE-preservation obligations never mentioned (correctly non-blocking at spec level, but worth closing on the record) | Low | §5 (dependency-vetting bullet) |

Overall License/compliance-gate verdict on Rev-B (verbatim from the
review): nothing found rises to a currently-accurate active violation that
would require rethinking the initiative's scope — every provider and
package choice independently verified (Rerun, `jax`, `jaxlib`, `optax`,
`equinox`, `diffrax`, `google-colab-cli`) turned out to be exactly what the
Charter assumed, and the `jaxpi` catch already on record in §3 remains
sound. What was missing was process completeness (a copy-paste boundary, a
ToS-vs-license distinction, an on-the-record check of the initiative's own
primary dependencies), now closed above.

## 15. Technical feasibility review findings (Rev-C → Rev-D)

Performed by a dedicated Technical-feasibility-gate review agent, grounded
against `train.py`/`model.py`'s actual current structure, `ci.yml`'s
CPU-only shape, and this project's own documented per-experiment runtimes.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | `device-abstraction`'s natural PyTorch-idiom implementation doesn't structurally carry into JAX's functional device-placement model | High | §6 (sequencing caveat) |
| 2 | `rerun-visualization`'s "240 lines, smallest surface area" claim only covers the `mx-viz` plot-function swap, not the `train.py` instrumentation the arc's actual live-timeline value proposition requires | High | §3 (`rerun-visualization`) |
| 3 | §1's "entanglement" rationale was technically inaccurate — GPU compute was never blocked by the PyTorch/JAX choice, only by zero device-awareness | High | §1 |
| 4 | §3's unconditional "Commit to JAX... replacing PyTorch" language oversold the SOAP/L-BFGS gap already admitted in §7, with no mitigation options named | High | §3 (`jax-migration`), §7 (SOAP bullet) |
| 5 | CI's GPU-verification story was only assigned to `cloud-compute-ops`, leaving `device-abstraction`'s own correctness-verification unaddressed | Medium | §7 (CI bullet) |
| 6 | `cloud-compute-ops`'s fit against em-piml's actual (small, 35-500s CPU) workload was never weighed as a technical question | Medium | §3 (`cloud-compute-ops`) |

Overall Technical-feasibility-gate verdict on Rev-C (verbatim from the
review): not blocked outright — no named prerequisite lacks a credible
technical path (even the SOAP gap has real mitigation options; they were
just unrecorded). Three findings point to the same underlying pattern: the
Charter's dependency graph (§6) and comparative-scope claims (§3) were
written from an intuitive read of each Arc rather than a back-traced one.
None require rethinking an Arc's scope wholesale.

## 16. Cost/compute-budget review findings (Rev-C → Rev-D)

Performed by a dedicated Cost/compute-budget-gate review agent, checked
against `CONVENTIONS.md`'s compute-assumption entry in full.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | GCP TPU Research Cloud requires a billing-account-linked project; IAM role-scoping alone doesn't prevent provisioning a billable, non-TRC-covered resource | Critical | §5 (budget-cap/billing-alert requirement) |
| 2 | No usage-monitoring or budget-alert control for Kaggle/Colab free-tier quotas; overage behavior not independently confirmed | High | §5 (quota-monitoring requirement) |
| 3 | §4's "no convention change needed" conclusion didn't account for scripted/repeated automated use, only occasional manual use | High | §4 (compute-assumption entry, qualifying paragraph) |
| 4 | No stated justification for four parallel provider integrations against em-piml's current small (35-500s CPU) workload | Medium | §3 (`cloud-compute-ops`, phased-rollout requirement) |
| 5 | NSF ACCESS Explore-tier cost dimension (beyond eligibility) not independently confirmed | Low | §11 (extended existing bullet) |

Overall Cost/compute-budget-gate verdict on Rev-C (verbatim from the
review): not ready to clear as written. The Critical finding is a genuine,
currently-plausible path to real-world cost that existed precisely because
§5's "free-tier-cloud-only" constraint was asserted as intent with no
enforcement mechanism — the same gap pattern the Security gate had already
treated as disqualifying for Rerun's recording-only mode. None of this
requires rethinking `cloud-compute-ops`'s provider choices, only committing
to concrete budget-cap/alert/monitoring controls before Rev-0.

## 17. Convention-alignment review findings (Rev-C → Rev-D)

Performed by a dedicated Convention-alignment-gate review agent, which read
all of `CLAUDE.md` and `CONVENTIONS.md` in full (not just the entries this
Charter already cited) and spot-checked the document's own internal
consistency.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | Five Rev-C provenance tags cited `see §13` (Security findings) when they meant `see §14` (License/compliance findings) — a copy-paste artifact from Rev-B's tags never updated | High | Six citations corrected throughout the document |
| 2 | `CONVENTIONS.md`'s optimizer-default entry (2026-07-15, `torch.optim`-anchored) is substantively touched by `jax-migration` but was omitted from §4's reconciliation | High | §4 (new "Optimizer default" paragraph) |
| 3 | `cloud-compute-ops`/`device-abstraction` operational tooling had no stated home in the `tools/<name>` workspace-package pattern this repo already established for comparable needs | Medium | §3 (`cloud-compute-ops`, `device-abstraction`) |
| 4 | The `slow`-marker testing convention doesn't cover a fundamentally new exclusion reason (hardware unavailability) that GPU-gated tests would need | Medium | §7 (CI bullet, new marker requirement) |
| 5 | PR #55 was stale relative to the Charter's actual revision and didn't address whether Design/Arc Charter PRs are exempt from the Slice-level PR template sections | Medium | §8 (exemption stated); PR #55 itself updated separately, outside document text |
| 6 | Inline provenance tags + finding tables + Revision History prose triple-record the same findings — not required by `docs/design/README.md`, a Charter-specific choice worth flagging as optional | Medium (no text change required) | §2 (self-consistency note) |

Overall Convention-alignment-gate verdict on Rev-C (verbatim from the
review): not ready to advance as-is. The five broken cross-references are a
small, mechanical, but real defect — exactly what this gate exists to
catch. The optimizer-convention omission is the more substantive finding,
now given the same explicit-trigger treatment as the ML-framework-default
entry. The tooling-placement and testing-convention gaps are real but only
block the Arc Charters that come after this one, not this Charter's own
advancement. None of this rises to a scope-rethinking problem.

## 18. Goal-delivery review findings (Rev-C → Rev-D)

Performed by a dedicated Goal-delivery-gate review agent, checked directly
against the owner's own original directive language (quoted verbatim in
the review brief), not a paraphrase.

| # | Finding | Severity | Addressed in |
|---|---|---|---|
| 1 | Rerun's recording-only constraint (set by the Security gate, reusing a static-reporting-era rationale) may under-deliver the initiative's primary named goal ("experience the experiment with my eyes") — never offered back to the owner as the tradeoff it is | **Critical** | §11 — escalated as an explicit owner decision, **not resolved by this revision** |
| 2 | `cloud-compute-ops` scopes "setup" thoroughly but never operationalizes "maintain," the second half of the original directive | High | §3 (`cloud-compute-ops`, new paragraph) |
| 3 | No correctness/parity-verification strategy for the PyTorch-to-JAX reimplementation — a research-integrity gap distinct from the license-driven copy-paste boundary | Medium | §7 (new bullet) |
| 4 | The six-gate process risks compounding hedges faster than any Arc Charter gets written — an early warning sign, not yet a failure | Medium | §9 (process-scope note, see below) |
| 5 | The `jaxpi` narrowing relative to "potentially even building on top of it" is functionally clear but wasn't connected back to the original directive language for the historical record | Low | §3 (`jax-migration`, new sentence) |

Overall Goal-delivery-gate verdict on Rev-C (verbatim from the review):
largely on track — neither the `jaxpi` narrowing nor §7's general
scope-expansion shows accidental drift. But one genuine, unflagged
goalpost-move exists at the center of the initiative (finding 1), and it
should be resolved by the owner, explicitly, before an Arc Charter for
`rerun-visualization` is written against a constraint that may quietly
under-deliver the goal that started the whole Design. Setup-shaped,
defensible, easily-specified work (findings 2-5) is getting done
thoroughly; the harder-to-operationalize, ongoing/experiential parts of the
original ask needed this gate to surface them.

A short process-scope note responding to finding 4: this Design Charter's
job is to fix the shape of the problem and its non-negotiable boundaries —
not to pre-resolve every risk an Arc Charter will encounter. Once this
document reaches Rev-0, its settled constraints (§5, §7, §13-§18) are
inherited as given by Arc Charters, not re-litigated from scratch in each
Arc's own gate cycle. If an Arc Charter's own review process exceeds Rev-D
without converging, that is itself a signal worth raising to the owner
rather than continuing to iterate. *(Added in Rev-D, Goal-delivery gate
finding, see above.)*

**Rev-E editorial note:** the `rerun-visualization` arc named throughout
§13-§18's findings tables above was renamed `field-visualization` and its
underlying toolkit changed from Rerun.io to PyVista/Plotly/matplotlib, per
an explicit owner course-correction (§3). The findings above remain
accurate as the historical record of what Rev-A through Rev-D actually
contained and how each gate reasoned about it at the time — they are not
restated or retracted, only superseded in current effect by §3's Rev-E text
and the Gates checklist in §10. In particular, finding 1 immediately above
(this section's own Critical finding) is resolved as described there, not
by either of the two options its own "Addressed in" column named.

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-28 | Initial draft | (pending) |
| B | 2026-07-29 | Security gate review (6 findings) incorporated: autonomy-label carve-out for CI/secrets diffs, local-credential-hygiene convention, generalized dependency-vetting bar, credential rotation/revocation requirement, Rerun recording-only enforcement mechanism, training-data-sensitivity confirmation. Fixed a stale "five lenses" cross-reference in §9. | Security |
| C | 2026-07-29 | License/compliance gate review (7 findings) incorporated: `jaxpi` copy-paste prohibition, provider Terms-of-Service-vs-license distinction (Colab automation-restriction tension named explicitly), on-the-record license check of `jax`/`jaxlib`/`optax`/`equinox`/`diffrax`, tightened unofficial-fork bullet, corrected Rerun's license citation (dual MIT/Apache-2.0), NOTICE-obligation closing note, repo-wide no-LICENSE-file gap tracked as a deferred open item. | Security, License/compliance |
| D | 2026-07-29 | Four gates reviewed in parallel against Rev-C: Technical feasibility (6 findings), Cost/compute-budget (5 findings, 1 Critical — GCP TRC billing enforcement), Convention-alignment (6 findings, including 5 broken §13/§14 cross-references from Rev-C now fixed), Goal-delivery (5 findings, 1 **Critical**, escalated to §11 as an owner decision, not resolved here). All non-escalated findings incorporated: corrected §1's "entanglement" rationale, corrected `rerun-visualization`'s surface-area claim, softened `jax-migration`'s unconditional replacement language and named SOAP mitigation options, added a PyTorch→JAX parity-verification requirement, added `cloud-compute-ops` workload-fit/phased-rollout and "maintain" (not just "setup") requirements, added GCP billing-alert and Kaggle/Colab quota-monitoring requirements, added the `CONVENTIONS.md` optimizer-default entry to §4's reconciliation, added `tools/<name>` placement requirements and a GPU test-marker requirement, added a Design/Arc Charter PR-template-exemption note, and added a process-scope note on gate-cascade risk. | Security, License/compliance, Technical feasibility, Cost/compute-budget, Convention-alignment (5 of 6 — Goal-delivery open pending owner decision) |
| E | 2026-07-30 | Owner course-correction (PR #55 comment) redirected the visualization arc away from Rerun.io entirely, following dedicated research into EM-field-capable rendering toolkits (surveyed PyVista, Plotly, matplotlib, K3D, ParaView, Mayavi, VisPy, napari, yt, and the visualization approaches of Meep/gprMax/openEMS). Renamed `rerun-visualization` → `field-visualization`; committed to PyVista (primary, 3D/volumetric) + Plotly (lightweight embeddable interactives) + existing matplotlib/`mx-viz` (2D), replacing Rerun.io entirely. Resolves Rev-D's escalated Goal-delivery Critical finding (§18) directly rather than choosing between the two options §11 offered. Added a third `CONVENTIONS.md` reconciliation paragraph (plotting default, 2026-07-27 entry, §4) proactively rather than waiting for a future gate to catch the omission. Self-checked all six gates against this revision's scope change (§10) — License/Cost/Convention/Technical-feasibility re-affirmed using the same research backing §3, Goal-delivery cleared via the owner's explicit decision — explicitly flagged as a lighter-weight self-check rather than a fresh independent per-gate agent review, with promotion to Rev-0 left as an explicit owner choice. | Security, License/compliance, Technical feasibility, Cost/compute-budget, Convention-alignment, Goal-delivery (6 of 6, self-checked — see §10 reviewer notes) |

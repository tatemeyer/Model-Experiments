---
title: "Design Charter — em-piml Modernization"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-28-foundation
slice: 2026-07-28-charter
revision: B
status: draft
date: 2026-07-29
related-arcs: [rerun-visualization, jax-migration, cloud-compute-ops, device-abstraction]
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
three independent Issues, because they're entangled: adopting JAX plausibly
unlocks GPU/TPU compute the current PyTorch/CPU-only setup has never used;
GPU/TPU compute plausibly unlocks the kind of live, high-frequency
visualization Rerun.io is built for; and none of the three can be scoped
responsibly as a single-Issue PR the way this repo's existing experiments
are. This Charter exists to give that combined initiative a single, coherent
scope and a set of named, sequenceable Arcs before any implementation
starts.

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
because this specific initiative — three entangled, multi-week changes to a
project's core framework, compute, and visualization stack — doesn't fit
that loop's single-Issue shape, not because the loop itself is inadequate.

## 3. Scope

### rerun-visualization

Replace/extend `mx-viz`'s current visualization capability with Rerun.io
(`github.com/rerun-io/rerun`, Apache-2.0) as the long-term solution. `mx-viz`
today (`tools/viz/src/mx_viz/`, 240 lines across `fields.py`, `training.py`,
`sweeps.py`, `io.py`, `cli.py`) produces static matplotlib PNGs only — no
time-axis motion, no live view into a training run, no way to watch e.g.
issue #37's R3 adaptive-sampling collocation points migrate during
retain/resample/release. Rerun is purpose-built for time-varying multimodal
data and can unify field state, collocation-point distribution, and loss
into one synchronized, scrubbable timeline. Its live/streaming viewer mode
needs a running native process, which conflicts with this repo's existing
"static PR/markdown artifact, no hosted dashboard" posture (the rationale
`CONVENTIONS.md` already gives for choosing matplotlib in the first place);
the mitigation is Rerun's recording-file (`.rrd`) mode — log during
training, no persistent viewer required, inspect or export later. This is
the smallest-surface-area of the three initiatives (240 lines to touch,
versus 1294 for `train.py`, see `jax-migration` below).

### jax-migration

Commit to JAX as em-piml's ML framework, replacing PyTorch. The most
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
tests, and it sidesteps the license question entirely. `train.py` (1294
lines, every training-loop variant across ~10 experiments) is the dominant
migration surface.

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

### device-abstraction (prerequisite arc — research-surfaced, not owner-named)

**This arc was not named in the original directive; it is a prerequisite
this Charter is recommending be tracked explicitly, pending confirmation
(see "Open questions," §11).** A repo-wide grep found zero device-awareness
anywhere in `projects/em-piml/src/em_piml/` today — no `.cuda()` call, no
`torch.device` construction, no CLI/env flag for device selection. Every
tensor is implicitly CPU. Both `jax-migration` and `cloud-compute-ops` need
a real device-abstraction layer to mean anything in practice; without it,
"commit to JAX" and "set up cloud GPU access" are each individually
incomplete. All data trained on under any Arc in this Design is synthetic
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
`cloud-compute-ops` actually starts.

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
  explicitly justifies paid compute.
- No hosted/live-viewer dashboard as a required artifact — Rerun's
  recording-file mode, not its live-streaming mode, unless a later revision
  explicitly revisits this. `rerun-visualization`'s Arc Charter must specify
  how recording-only mode is *enforced*, not just stated — e.g. a single
  shared logging helper in `mx-viz` that wraps `rr.save(...)`/file-mode
  logging as the only sanctioned entry point, with `rr.connect`/`rr.serve`-style
  live-mode calls flagged in code review as an explicit deviation requiring
  this section's own exception clause. *(Added in Rev-B, Security gate
  finding — see §13.)*
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
  ecosystem — PyPI, the Rerun SDK, cloud provider CLIs) gets the same
  license-file-not-just-metadata check applied to `jaxpi` in §3, recorded
  in the Slice PR that adds it. Unofficial/community forks or ports of a
  library (e.g. a JAX port of `pytorch_optimizer.SOAP`, see §7) require an
  explicit maintenance/provenance note, not just a license check, before
  being added to `uv.lock`. *(Added in Rev-B, Security gate finding — see
  §13.)*
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
| `rerun-visualization` | Rerun.io as the long-term visualization solution | largely independent of the other three |

`rerun-visualization` can proceed in parallel with the others — it doesn't
depend on JAX or cloud compute, and vice versa. `device-abstraction` is the
one clear blocking prerequisite: both `jax-migration` and `cloud-compute-ops`
are individually incomplete without it.

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
  it away.
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
  carve-out regardless of which way this is decided.

## 8. Relationship to the Issue/PR loop

This spec tree is upstream design authority, not a parallel tracking
system. When a Slice document under any of these Arcs is ready to be built,
the work still goes through this repo's normal loop unchanged: an Intent
Issue linking back to that Slice document, a PR implementing it, CI as the
source of truth for done, autonomy-label-gated merge (`CLAUDE.md:14-35`).
No epics are introduced into the Issue tracker.

## 9. Process recap

See `docs/design/README.md` for the full definition of the Design/Arc/Slice
hierarchy, the Rev-A → Rev-0 lifecycle, Change Orders, the six review
lenses, and a glossary.

## 10. Gates — Rev-B

- [ ] Technical feasibility
- [ ] License/compliance
- [ ] Cost/compute-budget
- [ ] Convention-alignment
- [ ] Goal-delivery
- [x] Security — reviewed against Rev-A; 6 findings (1 Critical, 2 High, 2
      Medium, 1 Low), all incorporated into this revision. See §13 for the
      full findings log. Re-review recommended once an Arc Charter is
      written against the new constraints, to confirm the remedies as
      *worded* actually hold up once there's real content to check them
      against — this sign-off covers the Charter text, not any future
      implementation.

Reviewer notes: Security gate cleared for Rev-B content. Remaining five
gates not yet reviewed.

## 11. Open questions deferred to Rev-B

- Confirm `device-abstraction` as its own Arc versus folding it into
  `foundation` — this Charter recommends a standalone Arc since both
  `jax-migration` and `cloud-compute-ops` depend on it directly, but it was
  research-surfaced, not owner-named, and should be confirmed rather than
  assumed.
- CI-scope decision (§7): fully outside existing CI, or a new
  non-blocking workflow.
- NSF ACCESS "Explore" tier eligibility for a fully independent (non
  institutionally-affiliated) researcher — not independently confirmed by
  prior research.
- Sequencing: should `device-abstraction` fully complete before
  `jax-migration`/`cloud-compute-ops` start, or can they proceed with a
  minimal device-abstraction slice first and iterate?

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

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-28 | Initial draft | (pending) |
| B | 2026-07-29 | Security gate review (6 findings) incorporated: autonomy-label carve-out for CI/secrets diffs, local-credential-hygiene convention, generalized dependency-vetting bar, credential rotation/revocation requirement, Rerun recording-only enforcement mechanism, training-data-sensitivity confirmation. Fixed a stale "five lenses" cross-reference in §9. | Security |

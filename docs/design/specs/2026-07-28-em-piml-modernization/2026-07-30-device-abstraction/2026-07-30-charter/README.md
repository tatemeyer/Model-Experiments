---
title: "Arc Charter — device-abstraction"
design: 2026-07-28-em-piml-modernization
arc: 2026-07-30-device-abstraction
slice: 2026-07-30-charter
revision: A
status: draft
date: 2026-07-30
related-slices: [device-selection-module, training-loop-threading, gpu-selection-verification]
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

A repo-wide grep found zero device-awareness anywhere in
`projects/em-piml/src/em_piml/` (no `.cuda()` call, no `torch.device`
construction, no CLI/env flag for device selection) — every tensor is
implicitly CPU today. This is the one clear blocking prerequisite named in
the parent Design Charter's §6 dependency table: both `jax-migration` and
`cloud-compute-ops` are individually incomplete without it — "commit to
JAX" and "set up cloud GPU access" each mean nothing in practice while
nothing in the codebase can select a device. Confirmed as its own
standalone Arc (not folded into `foundation`) as part of the usage-constraint/
dev-time sequencing breakdown adopted 2026-07-30 (parent Charter §11).

## 2. Relationship to the Design Charter

This Arc Charter inherits every constraint in the parent Design Charter,
especially:

- §5's compute-stays-CPU-primarily/free-tier-cloud-only constraint.
- §5's CI-stays-CPU-only-unless-explicitly-justified constraint.
- §7's CI-implications split: CPU-path correctness (defaults to CPU,
  doesn't error absent an accelerator) is CI-verifiable and should be;
  actual GPU/TPU-selection correctness requires hardware CI doesn't have,
  and should be verified manually/interactively with the result recorded in
  an experiment write-up, gated by a new `@pytest.mark.gpu` marker rather
  than reusing `slow` (parent §7, Convention-alignment gate finding, §17).
- §6's sequencing caveat: this Arc's natural PyTorch-idiom implementation
  (`.to(device)` calls, `device=` threaded through tensor-construction call
  sites) does not structurally carry into JAX's functional device-placement
  model (`jax.device_put`, largely automatic backend selection) — treat it
  as an interim measure for the current codebase, not code `jax-migration`
  inherits wholesale.

This document adds only what's specific to `device-abstraction`; it doesn't
re-litigate the parent's constraints.

## 3. Scope

- Add real device selection to `projects/em-piml/src/em_piml/`'s training
  code: a CLI flag / environment variable (exact naming left to Slice 1)
  that selects an accelerator when available and requested, defaulting to
  CPU otherwise.
- **Package home: project-scoped** (`projects/em-piml/src/em_piml/`), not a
  `tools/<name>` package — this resolves the parent Design Charter's own
  deferral on this point (§3, `device-abstraction` paragraph). Device
  selection here is core training-loop logic specific to em-piml's own
  experiments, not a reusable utility other projects would import, which is
  what distinguishes it from `tools/viz`/`tools/datasets`-class shared
  tooling.
- Implementation idiom: PyTorch's (`.to(device)`, `device=` threaded
  through tensor-construction call sites) — the same pattern this repo
  already used for `dtype` in the FP64 precision experiment (parent §6).
- **Out of scope for this Arc:** any JAX-idiom device placement
  (`jax.device_put`) — that belongs to `jax-migration`'s own Arc Charter
  once it exists, per §2 above.

## 4. Named Slices and sequencing

| Slice | Scope | Verifiable by |
|---|---|---|
| `device-selection-module` | A single device-selection helper + CLI/env flag; defaults to CPU; does not error when no accelerator is present | CI — no hardware needed |
| `training-loop-threading` | Thread the selected device through `train.py`'s (1294 lines) existing training-loop variants across ~10 experiments | CI (CPU-path only) + code review, per-variant |
| `gpu-selection-verification` | Manual, interactive confirmation that device selection actually places tensors on an accelerator when one is present and requested; recorded in an experiment write-up per this repo's existing determinism-verification convention | Manual/interactive only — requires actual hardware; gated by a new `@pytest.mark.gpu` marker so it's excluded from the default CI run |

`gpu-selection-verification` may need to wait on either the owner having
local GPU access or `cloud-compute-ops` reaching a working provider
integration — stating this dependency explicitly rather than assuming
hardware is available on this Arc's own timeline. `device-selection-module`
and `training-loop-threading` have no such dependency and can proceed
immediately.

## 5. Non-negotiable constraints

- Must default to CPU and must not error/degrade when no accelerator is
  present (parent §7).
- Any test requiring actual GPU hardware carries `@pytest.mark.gpu`, lives
  colocated in `projects/em-piml/tests/` per `CONVENTIONS.md`'s testing
  entry (no top-level `tests/` tree), and is excluded from the default
  fast/slow CI split (parent §7/§17).
- No dependency changes beyond what device selection itself requires —
  PyTorch already provides `torch.cuda.is_available()`/`torch.device`; no
  new third-party dependency is expected for this Arc's scope. If a Slice
  turns out to need one, parent §5's dependency-vetting bar applies exactly
  as it did for `field-visualization`'s PyVista/Plotly.

## 6. Cross-cutting gaps / risks specific to this Arc

- **Apple Silicon (`mps` backend) is not addressed by `CONVENTIONS.md`'s
  compute assumption** (`CONVENTIONS.md:44-51` names CPU + optional single
  consumer GPU + free-tier cloud; doesn't mention Apple Silicon
  specifically). Given the owner's observed working environment this
  session is Windows, `mps` support is likely out of scope for this Arc's
  first Slice — flagged here rather than silently assumed, since a future
  contributor on Apple hardware might expect it.
- Threading device selection through ~10 experiment variants in `train.py`
  (Slice 2) is mechanical but touches every training-loop path — real risk
  of a silent regression in one variant's default behavior if verified only
  once rather than per-variant.

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

- Exact flag/env-var naming (e.g. `--device` vs. `EM_PIML_DEVICE`) — left to
  Slice 1's own implementation, doesn't need Arc-level resolution.
- Whether `gpu-selection-verification` (Slice 3) blocks this Arc's own
  Rev-0, or can be deferred to run opportunistically whenever hardware
  becomes available without holding up Slices 1-2's completion.

## 10. Rollback / abandonment path

Per the parent Design Charter's §12: abandoning this Arc before reaching its
own Rev-0 is a lightweight `status: abandoned` change, not a Change Order.

## Revision History

| Rev | Date | Summary of changes | Gates cleared |
|---|---|---|---|
| A | 2026-07-30 | Initial draft | (pending) |

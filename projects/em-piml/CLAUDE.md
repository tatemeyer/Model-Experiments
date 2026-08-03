# em-piml

Physics-informed ML for electromagnetics. Motivating research interest:
tokenization/embedding schemes for PIML on EM problems (see issue #2 for
origin). This baseline exists to have a verified walking skeleton before
that research work branches off it — it is intentionally a plain
coordinate-input MLP, not a tokenization/embedding experiment.

## Problem being solved (baseline)

The fundamental mode of a 1D perfect-electric-conductor (PEC) cavity of
length `L` — a direct 1D reduction of Maxwell's equations to the wave
equation:

```
d^2E/dt^2 = c^2 * d^2E/dx^2,   E(0,t) = E(L,t) = 0
```

with closed-form solution `E(x,t) = A * sin(n*pi*x/L) * cos(omega*t)`,
`omega = n*pi*c/L` (see `src/em_piml/physics.py`; `L = c = n = A = 1`).
Closed-form means verification is exact, not eyeballed.

## Model and training

`CavityPINN` (`src/em_piml/model.py`) is a 3-layer, 32-wide tanh MLP
taking `(x, t)` and predicting `E_z`. `train_cavity_baseline`
(`src/em_piml/train.py`) minimizes four loss terms via Adam: the PDE
residual (via double autograd), the two boundary conditions, the
initial field `E(x,0)`, and `dE/dt(x,0) = 0` (true for this
standing-wave mode). Defaults (4000 steps, lr=3e-3, small
collocation/boundary/initial batches) were tuned empirically — checked
convergent and stable across training seeds 0/1/2/7 (relative L2 error
0.026-0.046 against the analytical solution), ~35s on CPU. Don't
increase `steps` or network size without re-checking CI runtime stays
well under a minute.

Reproduce: `uv run python3 -m em_piml.train` (prints relative L2 error
over 500 random held-out points).

## Verification

`tests/test_baseline_cavity.py` trains with a fixed seed (0) and
asserts relative L2 error against the analytical solution, evaluated at
500 random `(x, t)` points sampled with a *different* seed (123) than
training — genuinely held-out, not memorized collocation points.
Tolerance is `0.1`, chosen with ~2-4x margin above the empirically
observed 0.026-0.046 range so the test doesn't flake on ordinary
run-to-run variance.

Pitfall already hit once: don't evaluate relative L2 error at a single
`(x, t)` snapshot — if it lands on a zero-crossing of the analytical
solution (e.g. `t = PERIOD/4`, where `cos(omega*t) = 0`), the near-zero
denominator inflates the metric to nonsense. Always evaluate over many
points spanning the domain.

Second pitfall already hit once (issue #19): **seed the RNG before
constructing the model, not after.** A refactor while adding a second
training variant moved `torch.manual_seed(seed)` to after `model =
SomeModel(...)` — the model's weight initialization then drew from
whatever the ambient RNG state happened to be, not the intended seed,
silently breaking reproducibility (only training-time sampling stayed
seeded). Caught by re-verifying that a fixed seed reproduces the exact
same result before trusting a seed-to-seed comparison. If you add a new
`train_*` function here, verify determinism (same seed in, bit-identical
result out) before trusting any numbers from it — a standing rule that
has caught real bugs more than once (see
`experiments/long-horizon-collapse/032-curriculum-long-horizon.md`'s
erratum for a case where determinism held but the number was still
wrong for an unrelated reason).

## Data

Ground-truth grid is registered via `mx-data`, not computed ad hoc:
`uv run mx-data fetch em-piml-1d-cavity-analytical`. The generator
(`tools/datasets/registry/generators/em_piml_1d_cavity_analytical.py`)
imports `em_piml.physics.analytical_field` rather than duplicating the
formula — it works because `uv sync --all-packages` installs every
workspace member (including this one) into the one shared venv.

## Where to find things

This file stays a short, stable router — it does not grow a new section
per issue. Per-experiment write-ups, structured results, and literature
notes live elsewhere:

- **`experiments/<thread>/NNN-slug.md`** (or a top-level `experiments/
  NNN-slug.md` for a standalone question) — the full write-up for each
  experiment: mechanism, results, mechanistic/pointwise diagnosis,
  what the corresponding test locks in, and leads. `experiments/
  TEMPLATE.md` is the skeleton for the next one, including where a new
  file should go (existing thread folder vs. new standalone file).
- **`results.csv`** — every experiment's numeric results in tidy
  long format: one row per `(issue, variant, seed, metric)` datapoint,
  columns `issue,experiment_slug,variant,seed,metric,value,params,date`
  (`params` is a JSON blob of whatever hyperparameter(s) that row's
  experiment swept). This is the machine-readable source for plotting
  results across experiments — don't hand-transcribe a results table
  into prose only; add rows here too.
- **`LITERATURE.md`** — every paper this project has cited or tested,
  one row per paper, with a verdict (tried/worked, tried/didn't,
  tried/actively-worse, ruled-out, or theory-only) and why. Check here
  before proposing a paper as a "new" lead.

## Experiment index

Verdict key: ✅ helped · ⚠️ partial/modest · ❌ no effect · 🔻 actively worse.

**Baseline**

| issue | question | verdict | where |
|---|---|---|---|
| #2 | PINN baseline for the 1D cavity fundamental mode | ✅ | this file, "Problem being solved" / "Model and training" above |

**Thread: `num-bands-gap/`** — why does `num_bands=4` Fourier embedding
destabilize training, and how is it fixed?

| issue | question | verdict | where |
|---|---|---|---|
| #4 | Fourier feature embedding vs. raw-coordinate baseline | ⚠️ neutral at num_bands=2, unstable at 4+ | `experiments/num-bands-gap/004-fourier-embedding.md` |
| #6 | Does L-BFGS fix the num_bands=4 instability? | ⚠️ partial | `experiments/num-bands-gap/006-lbfgs-optimizer.md` |
| #8 | Does denser collocation fix the rest? | ⚠️ most of it, noisily | `experiments/num-bands-gap/008-denser-collocation.md` |
| #12 | Is the density non-monotonicity about count or which points? | — it's the points | `experiments/num-bands-gap/012-point-draw-variance.md` |
| #10 | Does more network capacity close the residual gap? | ✅ | `experiments/num-bands-gap/010-network-capacity.md` |
| #11 | Does SOAP close the rest of the gap? | ✅ (fully, independently of #10) | `experiments/num-bands-gap/011-soap-optimizer.md` |
| #38 | Is the residual instability an FP32 precision artifact ("FP64 is All You Need")? | ❌ not a precision artifact | `experiments/num-bands-gap/038-fp64-precision.md` |
| #40 | Does Sobol quasi-random sampling reduce the issue #12 point-draw variance? | ⚠️ real but modest on top of issue #10's capacity fix, which already did most of the work | `experiments/num-bands-gap/040-sobol-sampling.md` |

**Thread: `two-mode-spectral-bias/`** — does a two-mode target reproduce
spectral bias, and what fixes it?

| issue | question | verdict | where |
|---|---|---|---|
| #22 | Does a two-mode superposition break the baseline, and does Fourier embedding fix it? | ⚠️ breaks it, Fourier partially helps | `experiments/two-mode-spectral-bias/022-two-mode-superposition.md` |
| #25 | Does raising num_bands close the gap? | ❌ helps slightly at 4, destabilizes past it | `experiments/two-mode-spectral-bias/025-num-bands-sweep.md` |
| #39 | Does Random Weight Factorization close the gap, alone and combined with Fourier embeddings? | ⚠️ small mixed effect (helps alone and with num_bands=4, slightly hurts with num_bands=2), doesn't close it | `experiments/two-mode-spectral-bias/039-random-weight-factorization.md` |

**Thread: `long-horizon-collapse/`** — training/evaluating over multiple
periods collapses to a degenerate near-constant output; what fixes it?

| issue | question | verdict | where |
|---|---|---|---|
| #23 | Does a longer horizon break causality, and does causal reweighting fix it? | ❌ breaks badly, causal reweighting doesn't help | `experiments/long-horizon-collapse/023-long-horizon-causal.md` |
| #30 | Does pseudo-sequence tokenization fix it? | ❌ same collapse, no better | `experiments/long-horizon-collapse/030-pseudo-sequence-long-horizon.md` |
| #32 | Does a short-to-long curriculum fix it? | ⚠️ real but modest improvement | `experiments/long-horizon-collapse/032-curriculum-long-horizon.md` |
| #34 | Does NTK-based adaptive loss reweighting fix it? | 🔻 actively worse than doing nothing | `experiments/long-horizon-collapse/034-ntk-reweighted-long-horizon.md` |
| #35 | Does an explicit anti-trivial-solution regularizer fix it? | 🔻 mildly worse than doing nothing | `experiments/long-horizon-collapse/035-antitrivial-regularizer.md` |
| #36 | Does a Neuro-Spectral Architecture (NeuSA) fix it? | ✅ essentially solves it | `experiments/long-horizon-collapse/036-neusa-long-horizon.md` |
| #37 | Does R3 (Retain-Resample-Release) adaptive sampling fix it? | 🔻 mildly worse than doing nothing | `experiments/long-horizon-collapse/037-r3-long-horizon.md` |

**Standalone**

| issue | question | verdict | where |
|---|---|---|---|
| #20 | Does pseudo-sequence tokenization (PINNsFormer) beat the raw-coordinate baseline? | 🔻 markedly worse | `experiments/020-pseudo-sequence-tokenization.md` |
| #46 | Does capacity help resolve a local dielectric-interface derivative kink, in contrast to issue #25's negative global-spectral-content finding? | ⚠️ real but modest, reduced-scope result | `experiments/046-dielectric-interface-capacity.md` |

## Open leads

Actively maintained — when one of these is picked up by a new issue,
its entry is struck or replaced with a pointer to the result, not left
stale.

**num-bands-gap:**
- SOAP/SS-Broyden hyperparameters (`lr`, `betas`, `precondition_frequency`)
  are untuned library defaults; there may be room to reduce `steps` below
  2000 without losing accuracy (`011-soap-optimizer.md`).
- ~~Why density-vs-accuracy was non-monotonic in the 1000-4000 range was
  investigated (`012-point-draw-variance.md`: it's the point draw); does
  quasi-random sampling (Sobol) fix it?~~ Tried in issue #40, which also
  finally re-verified issue #12's own flagged 32-hidden-vs-64-hidden gap
  along the way: at the current 64-hidden default, uniform sampling's
  point-draw stdev had already shrunk ~4-7x from capacity alone (0.035/
  0.047 → 0.0082/0.0066) — issue #10's capacity fix, not sampling
  method, turned out to be the dominant lever. Against that matched
  64-hidden baseline, Sobol gives a real further stdev reduction
  (~1.4-2.3x) but only marginal mean-accuracy improvement (~5%) — a
  modest refinement on top of the capacity fix, not a standalone
  dramatic fix. Latin hypercube (issue #12's other named candidate) is
  still untried. See `040-sobol-sampling.md`.
- ~~Is the residual `num_bands=4` L-BFGS instability actually an FP32
  precision artifact rather than an optimization one?~~ Tested in
  **issue #38**: no — FP64 at a matched iteration budget leaves both the
  original 32-hidden plateau (0.889-0.922, vs. FP32's 0.822-0.851) and the
  shipped 64-hidden config (0.028-0.058, vs. FP32's 0.018-0.041)
  essentially unchanged. It does, however, cost ~10-60x more wall time —
  see `038-fp64-precision.md` for why (L-BFGS's convergence test stops
  exiting early under FP64, confirmed via a reduced-budget ablation), and
  for the still-open question of whether a much larger budget would
  eventually help.

**two-mode-spectral-bias:**
- Network capacity was never widened specifically for this target the
  way `010-network-capacity.md` did for the single-mode case — untried.
  Queued as **issue #41** (PirateNets-style adaptive-residual
  architecture).
- ~~Does Random Weight Factorization (Wang et al., arXiv:2210.01274)
  close the gap, alone and combined with the existing Fourier
  embeddings?~~ Tried in issue #39: small, mixed effect (real but modest
  improvement alone and combined with `num_bands=4`; slightly *worse*
  combined with `num_bands=2`), doesn't close it. Pointwise-diagnosed as
  the same underlying mechanism as every prior fix in this thread: RWF
  reparameterizes the weights, not the input representation, so it
  never touches the missing-`8*pi`-basis-frequency gap `022-...md`
  identified — see `039-random-weight-factorization.md`.
- NTK-based adaptive loss reweighting (Wang/Teng/Perdikaris) was flagged
  here but never actually tried against *this* target — only against
  the long-horizon collapse (`034-ntk-reweighted-long-horizon.md`,
  where it hurt). Untried here specifically.
- Why `num_bands=8` SOAP diverges far harder/more seed-dependently than
  L-BFGS at the same `num_bands` is unexplained.

**long-horizon-collapse:**
- ~~The degenerate near-constant "escape hatch" itself (trivially
  satisfies the wave-equation residual) still isn't directly attacked by
  any fix tried so far — an explicit anti-trivial-solution penalty
  (variance/curvature floor) is the most literature-direct remaining
  lead.~~ Tried in issue #35 (Leiteritz & Pflueger's residual-gradient
  penalty, arXiv:2112.05620): mildly worse than doing nothing
  (0.968-0.972 vs. uniform's 0.923-0.926), not a fix. Diagnosed via
  per-chunk instrumentation: the penalty's own value is *highest* where
  the model still tracks real oscillation and decreases monotonically to
  near-zero deep in the collapsed plateau — the paper's mechanism
  assumes a trivial solution creates a localized residual-gradient spike
  (from an abrupt truth-to-trivial switch in a collocation-starved
  domain); this project's collapse is a smooth, domain-wide settling
  where the trivial region is the *cheapest* place to satisfy the
  penalty, not a place it flags. See `035-antitrivial-regularizer.md` for
  the full diagnosis and its own leads (a collocation-starved variant
  might transfer better; untried).
- ~~Neuro-Spectral Architecture (NeuSA) is untried against this
  collapse.~~ Done, see `036-neusa-long-horizon.md` — essentially solves
  it (architectural BC/IC + a linear ODE in spectral-coefficient space
  with a small learned correction), the first fix in this thread to do
  more than partially help.
- ~~R3 (Retain-Resample-Release) adaptive sampling is still untried
  against this collapse.~~ Tried in issue #37: mildly worse than doing
  nothing (0.930-0.934 vs. uniform's 0.923-0.926), not a fix. Diagnosed
  mechanistically: R3's retain criterion correctly finds the genuinely
  highest-residual region (the first ~40% of the domain), but that isn't
  where the collapse lives — the collapsed region has near-zero residual
  by construction, so R3 never once retains a point from it across a
  full training run, and concentrating budget on the already-best-fit
  early region implicitly starves the collapsed region of the sampling
  pressure uniform resampling gave it. Third residual-based adaptive
  mechanism (after NTK reweighting #34 and the anti-trivial regularizer
  #35) to fail for the same underlying reason: this project's collapse
  presents as *low*-residual, not high-residual, to any residual-derived
  signal. See `037-r3-long-horizon.md` for the full diagnosis and leads
  (a directly output-smoothness-based signal, rather than a
  residual-derived one, is the one class of fix in this thread not yet
  tried).
- Network capacity has never been varied across any of the five
  long-horizon experiments (#23/#25/#30/#32/#34) — still open. Now being
  tested via new problem variants that isolate capacity from the
  collapse mechanism: does a lossy/driven cavity (breaking the
  conservative-Hamiltonian structure the trivial solution exploits)
  remove the collapse and let capacity matter (**issue #45**)? Does a 2D
  PEC cavity reproduce the collapse with capacity-vs-mode-count as a
  controllable axis (**issue #44**)? Does a time-independent Helmholtz
  eigenvalue problem show capacity effects that the time-domain problem
  doesn't (**issue #43**)?
- NTK reweighting's formula could be inverted/symmetrized for this
  project's inverted regime (PDE term easy, not hard) instead of
  discarded outright (`034-ntk-reweighted-long-horizon.md` lead #1) —
  untried.

**dielectric-interface-capacity:**
- Does capacity help resolve a *local* dielectric-interface kink, in
  contrast to its established irrelevance to *global* spectral content
  (issue #25)? ✅/⚠️ resolved with a caveat: capacity gives a real,
  monotonic-on-average improvement here (unlike #25), but this issue's
  numbers use a reduced 3-capacity/2-seed/600-step budget because of
  transient sandbox CPU contention -- see
  `046-dielectric-interface-capacity.md`. Rerunning the full
  `{16,32,64,128,256}` x 4-seed x 4000-step convention whenever more
  time/compute is available is still open.
- The pointwise diagnosis found capacity reduces error roughly
  uniformly across distance from the interface rather than
  disproportionately at the interface itself -- an explicit
  interface-localized loss term (denser collocation right at the
  interface, or a soft penalty directly on the curvature-jump
  condition) is untried and might close the interface-specific gap
  faster than capacity alone.

## Known deferred items

- `torch` is installed from plain PyPI (bundles CUDA deps, larger than
  necessary for a CPU-only baseline). The dedicated CPU-only wheel index
  (`https://download.pytorch.org/whl/cpu`, via `[tool.uv.sources]` +
  `[[tool.uv.index]]`) is the right fix but couldn't be verified from
  this session's sandboxed network — untested config wasn't worth
  shipping blind. Revisit if CI install time/size becomes a problem.
- Equation tokenization (PITT) and patch-based multi-scale tokenization
  (MeshTok) were evaluated in issue #20 and ruled out of scope for this
  project's current shape (single fixed equation, no gridded field) —
  see `LITERATURE.md` and `experiments/020-pseudo-sequence-tokenization.md`
  for the reasoning.
- The `num_bands=4/6` instability is now mostly explained and resolved
  (see the `num-bands-gap` thread above); an FP32-precision-artifact
  explanation was tested and ruled out (issue #38), and quasi-random
  (Sobol) sampling was tested and found to help modestly (issue #40).


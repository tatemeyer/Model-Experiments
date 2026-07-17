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

Second pitfall already hit once: **seed the RNG before constructing the
model, not after.** A refactor while adding a second training variant
moved `torch.manual_seed(seed)` to after `model = SomeModel(...)` — the
model's weight initialization then drew from whatever the ambient RNG
state happened to be, not the intended seed, silently breaking
reproducibility (only training-time sampling stayed seeded). Caught by
re-verifying that a fixed seed reproduces the exact same result before
trusting a seed-to-seed comparison. If you add a new `train_*` function
here, verify determinism (same seed in, bit-identical result out)
before trusting any numbers from it.

## Data

Ground-truth grid is registered via `mx-data`, not computed ad hoc:
`uv run mx-data fetch em-piml-1d-cavity-analytical`. The generator
(`tools/datasets/registry/generators/em_piml_1d_cavity_analytical.py`)
imports `em_piml.physics.analytical_field` rather than duplicating the
formula — it works because `uv sync --all-packages` installs every
workspace member (including this one) into the one shared venv.

## Embedding experiment #1: Fourier features (issue #4)

`FourierCavityPINN` (`src/em_piml/model.py`) embeds `(x, t)` — each
normalized to `[0, 1]` — via a NeRF/Tancik-style positional encoding
(`src/em_piml/embeddings.py`) before the *same* MLP body shape as
`CavityPINN`: `[u, sin(2^0 pi u), cos(2^0 pi u), ..., sin(2^(k-1) pi u),
cos(2^(k-1) pi u)]` per scalar, for `x` and `t` independently.
`train_fourier_cavity_baseline` reuses the exact same training loop,
loss construction, steps, and optimizer as the baseline — the only
variable is the input representation.

**Finding:** at `num_bands=2` (the shipped default), performance is
statistically indistinguishable from the raw-coordinate baseline on
this problem at this scale — relative L2 error 0.033-0.043 across
seeds 0/1/2/7, versus the baseline's 0.026-0.046. Fourier features
neither clearly help nor hurt here.

**More interesting finding:** `num_bands=4` and `num_bands=6` — more
expressive embeddings — *destabilize* training at the same fixed
learning rate/step budget as the baseline (relative L2 error ~0.95-1.06,
i.e. it doesn't learn the solution at all). This wasn't chased further
because doing so (e.g. lowering the learning rate for higher band
counts) would break the controlled-comparison premise of this issue —
but it's a concrete, actionable lead for whoever picks up the next
embedding iteration: naive higher-frequency Fourier features need their
own optimization treatment (lower LR, warmup, or SIREN-style init), they
don't drop in for free at a fixed budget tuned for raw coordinates.

## Does the optimizer explain the num_bands=4 instability? (issue #6)

["Spectral bias in physics-informed and operator learning: Analysis and
mitigation guidelines"](https://www.alphaxiv.org/abs/2602.19265)
(Khodakarami et al., Brown/Karniadakis group, Feb 2026) argues via NTK
theory that this kind of instability under higher-frequency inputs is
primarily *dynamical* (an Adam/first-order-optimizer limitation — each
frequency mode's effective learning rate scales with its NTK
eigenvalue, which decays sharply with frequency), not a representational
failure of the embedding, and reports quasi-second-order optimizers
(SOAP, L-BFGS, SS-Broyden) largely resolving it.

`train_fourier_cavity_lbfgs` (`src/em_piml/train.py`) tests this
directly: same `num_bands=4` `FourierCavityPINN`, same loss
construction, `torch.optim.LBFGS` (built into PyTorch, no new
dependency) instead of Adam. Since L-BFGS assumes a fixed/deterministic
objective across its internal line-search evaluations, collocation
points are sampled once per run rather than resampled every step (see
`_train_pinn_lbfgs`), unlike the Adam path.

**Result: partial support, not full resolution.**

- Adam at `num_bands=4`: relative L2 error ~1.0-1.04 — doesn't learn the
  solution at all, with or without more training steps.
- L-BFGS at `num_bands=4` (from a fresh random init): converges to
  ~0.79-0.88 relative L2 across seeds 0/1 and a sweep of
  `outer_steps`/`max_iter` budgets (10-100 outer steps, 20-100 inner
  iterations each) — a real, substantial improvement over Adam's total
  failure, but the error **plateaus** there; more iterations stop
  helping past `outer_steps=50, max_iter=50` (the shipped default, ~40s).
- Tried Adam-warmup-then-L-BFGS too (the paper mentions L-BFGS is "often
  used after Adam warm-up"): 1000 Adam steps first (still ~1.03-1.04,
  consistent with Adam's failure above) then L-BFGS refinement converges
  to the *same* ~0.86 plateau as starting L-BFGS from scratch. Warmup
  doesn't change the outcome — this looks like a genuine local optimum
  of this loss landscape for this architecture/point-budget, not an
  initialization sensitivity.

So at the time: optimizer choice mattered a lot (L-BFGS clearly did
something Adam couldn't) but didn't fully explain the instability by
itself at 200 points. See below — lead #1 turned out to be most of the
answer.

## Does collocation-set density explain the rest? (issue #8)

Swept the fixed collocation/boundary/initial point-set size for
`train_fourier_cavity_lbfgs` on the same `num_bands=4` configuration —
architecture, optimizer, iteration budget all held fixed, density is
the only variable:

| n_collocation | seed 0 | seed 1 | seed 2 | seed 7 |
|---|---|---|---|---|
| 200 (original) | 0.822 | 0.851 | - | - |
| 1000 | 0.090 | 0.143 | - | - |
| 2000 | 0.098 | 0.104 | 0.096 | 0.065 |
| 3000 | 0.142 | 0.163 | - | - |
| 4000 | 0.055 | 0.129 | - | - |

**Density was most of the answer** — an order-of-magnitude improvement
over the 200-point plateau at every density tried from 1000 up. But the
relationship is **noisy, not monotonic**: 3000 points did *worse* than
2000, and 4000 was a mix of the best (0.055) and a middling (0.129)
result. This is a single fixed point sample per (seed, density) — which
specific points get drawn matters as much as the nominal count, at
least in this 1000-4000 range.

**Shipped default: `n_collocation=2000, n_boundary=400, n_initial=400`**
— chosen because it's the most thoroughly tested (4 seeds) and most
consistent (0.065-0.104, all within roughly 2x of each other, unlike
the wider spread at other densities). `tests/test_fourier_lbfgs.py`
asserts `< 0.15` (comfortable margin above the observed 0.104 worst
case) — tighter than issue #6's `0.95` partial-improvement bound by
over 6x, but still not the standard `0.1` baseline bar, because seed 1
landed at 0.104 — just over it. Don't tighten this further without
re-sweeping; the noise between seeds/densities above means a single
lucky seed isn't good evidence of a real margin.

**Leads at the time**, in rough order of how cheap they are to test:
1. ~~Collocation-set density~~ — largely resolved by this issue.
2. ~~Network capacity~~ — resolved, see issue #10 below.
3. SOAP/SS-Broyden (the paper's actual best performers) weren't tried —
   out of scope here since they require adopting a new
   package/implementation, not a `torch.optim` drop-in like L-BFGS.
4. ~~Understand *why* density vs. accuracy is non-monotonic~~ — see
   issue #12 below: it's the point draw, not the count.

## Is the density non-monotonicity about count or which points? (issue #12)

Issue #8's sweep drew *one* fixed collocation/boundary/initial point set
per (seed, density) — so "3000 did worse than 2000" could mean either
"3000 points is a worse count" or "that particular draw of 3000 points
was unlucky." This issue disentangles the two: hold `n_collocation`
*and* the model-init seed fixed, and vary only which points get drawn.

That required decoupling point-sampling randomness from model-init
randomness — previously both came from one `torch.manual_seed(seed)`
call before model construction, so there was no way to redraw points
without also reinitializing the model. `_sample_points` now takes an
optional `generator: torch.Generator | None` (default `None` preserves
the exact old behavior — draws from whatever the global RNG state is,
bit-for-bit unaffected), and `train_fourier_cavity_lbfgs` gained a
`points_seed: int | None` argument: when set, points are drawn from an
independent `torch.Generator().manual_seed(points_seed)`, decoupled
from `seed` (model init). See `src/em_piml/train.py`.

`src/em_piml/point_draw_sweep.py` (`uv run python3 -m
em_piml.point_draw_sweep`) sweeps 5 independent point-set draws
(`points_seed` 100-104) at each of `n_collocation=2000` and `4000`,
model-init `seed=0` and `n_boundary=n_initial=400` held fixed at PR #9's
shipped values throughout — architecture, optimizer, and iteration
budget (`outer_steps=50, max_iter=50`) unchanged from PR #9, exactly as
issue #12 required.

Note: the results below were measured against the 32-hidden architecture
(issue #8's shipped config, the current default at the time this issue
ran). Issue #10 landed in parallel and bumped the default to 64-hidden —
re-running `point_draw_sweep.py` now trains at 64-hidden instead, so
absolute numbers would likely shift down (per issue #10's findings).
The qualitative conclusion below (within-density variance rivaling
between-density variance) isn't expected to flip, but hasn't been
re-verified at 64-hidden.

**Results (relative L2 error, one model-init seed, 5 point draws each, 32-hidden):**

| n_collocation | draws | mean | stdev | range |
|---|---|---|---|---|
| 2000 | 0.0708, 0.0413, 0.0591, 0.1446, 0.0715 | 0.078 | 0.035 | 0.041-0.145 |
| 4000 | 0.1657, 0.0258, 0.0562, 0.1029, 0.0805 | 0.086 | 0.047 | 0.026-0.166 |

For comparison, issue #8's between-density sweep (1000-4000, mixing
different model seeds *and* one point draw each) ranged 0.055-0.163
(spread 0.108, pooled stdev ~0.033 across those 10 seed/density pairs).

**Conclusion: it's the points, not the count.** The within-density
spread from resampling alone (0.103 at 2000, 0.140 at 4000) is as large
as or larger than the entire between-density spread issue #8 found
across the whole 1000-4000 range, and the within-density stdev (0.035,
0.047) exceeds issue #8's pooled between-density stdev (0.033). Holding
`n_collocation` and the model-init seed completely fixed and only
redrawing points reproduces essentially the *same magnitude* of
variance that issue #8 saw from varying density itself. Density isn't
doing nothing (1000+ points is still an order of magnitude better than
the 200-point plateau, per issue #8), but past ~1000-2000 points, "which
points you happen to draw" is at least as good a predictor of the final
error as "how many points" — the 3000-worse-than-2000,
4000-mixed-results pattern in issue #8 is better explained as point-draw
noise than as a real non-monotonic effect of count. Consistent with
this: mean error at 4000 (0.086) was not better than at 2000 (0.078) in
this resampled experiment either.

This was investigated, not shipped as a new default or tightened test
bound — `tests/test_fourier_lbfgs.py`'s `< 0.15` bound and the
`n_collocation=2000` default from issue #8 stand as-is; this issue's
finding argues *against* trusting a tighter bound derived from any
single seed/draw, not for changing the current one.
`tests/test_point_draw_seed.py` covers the `points_seed` plumbing itself
(determinism, independence from global RNG) as a fast regression check;
the full point-draw sweep is deliberately not part of the pytest suite
(10 full L-BFGS training runs — reran here in the ~1.5-2.5 min/run
range, but re-running it on every CI invocation isn't worth the ~7x
slowdown to feed information CLAUDE.md now documents directly). Rerun
`point_draw_sweep.py` directly if this needs revisiting.

**Leads for whoever picks this up next:**
1. ~~Collocation-set density~~ (issue #8) and ~~count vs. which-points~~
   (this issue) — both resolved. The dominant lever left in the
   1000-4000 range is which points get drawn, not how many.
2. Network capacity, still held fixed since issue #4 — unexplored.
3. SOAP/SS-Broyden — still out of scope (new dependency).
4. Given points matter this much, a stratified/quasi-random sampling
   scheme (e.g. Latin hypercube or Sobol sequence instead of uniform
   `torch.rand`) might reduce the draw-to-draw variance found here
   without needing more points — untried.

## Does more network capacity close the residual gap? (issue #10)

PR #9 got `num_bands=4` L-BFGS down to 0.065-0.104 relative L2 — close
to, but not reliably under, the standard `0.1` bar (seed 1 landed at
0.104). Architecture (32-hidden, 3-layer) had been held fixed since
issue #4 specifically to keep every comparison in this thread
controlled against the `num_bands=2` baseline. This issue tests lead #2
above: was 32-hidden ever the right size for the higher-dimensional
Fourier-embedded input at `num_bands=4` (18-dim: `1 + 2*4` per scalar,
times 2 scalars), now that collocation density is no longer the
confound?

Swept `hidden`/`num_layers` in `FourierCavityPINN`, holding
`num_bands=4`, point-set density (`n_collocation=2000/n_boundary=400/
n_initial=400`), and L-BFGS settings (`outer_steps=50, max_iter=50`)
exactly at PR #9's shipped values — capacity is the only variable:

| hidden x layers | seed 0 | seed 1 | seed 2 | seed 7 |
|---|---|---|---|---|
| 32x3 (previous default) | 0.078 | 0.144 | 0.175 | 0.053 |
| 64x3 | 0.027 | 0.041 | 0.026 | 0.018 |
| 64x4 | 0.022 | 0.018 | 0.015 | 0.023 |

**Finding: capacity was most of the remaining gap.** Widening 32→64
(same depth) takes the worst-seed error from 0.175 down to 0.041 — every
seed now comfortably clears the standard `0.1` bar, and the four seeds
land in a tight 0.018-0.041 band instead of the noisy 0.053-0.175 spread
at 32-hidden. Going deeper as well (64x4) gives a further small
improvement (0.015-0.023) but doesn't change the qualitative picture —
diminishing returns from depth once width fixes the bottleneck.

**Shipped: `hidden=64, num_layers=3`** for `train_fourier_cavity_lbfgs`
(`num_bands=2` untouched, per the issue's constraint —
`train_fourier_cavity_baseline` still uses 32-hidden).
`tests/test_fourier_lbfgs.py` now asserts the standard `< 0.1` bar
(replacing the `0.15` bound from issue #8) — the residual gap from
issues #6/#8 is resolved, not just narrowed. Went with 64x3 over 64x4:
smallest capacity bump that already clears the bar, and the further
64x4 improvement wasn't worth the extra depth/runtime given this repo's
minimalism default (see root `CLAUDE.md`).

Runtime: L-BFGS closures scale with parameter count, so 64-hidden is
slower per run than 32-hidden — noticeably so on this sandbox's shared
CPU, though the exact multiplier depends heavily on how much other
concurrent load the box has (see below). Still comfortably fast enough
for a single test run in CI.

Note on how these numbers were produced: this sandbox intermittently
runs multiple concurrent agent sessions on the same 4-core box (other
`em-piml` issues being worked in parallel worktrees), which caused
severe CPU oversubscription (load average 12-16) during this sweep.
`torch`'s default intra-op threading compounds badly under that
contention — a single 32x3 run inflated from the documented ~40s to
30+ minutes of wall time under load. Pinning `torch.set_num_threads(1)`
for the sweep script avoided the thread-storm and restored normal
per-run cost (~100-220s depending on capacity); the table above was
produced that way. This doesn't change the qualitative finding (same
seeds, same algorithm, just a different intra-op reduction order), but
if these exact numbers don't reproduce bit-for-bit outside this sandbox,
that's why — rerun the sweep rather than assume something regressed.

**Leads for whoever picks this up next:**
1. SOAP/SS-Broyden (the paper's actual best performers, see issue #6)
   still weren't tried — out of scope here since they require adopting
   a new package/implementation, not a `torch.optim` drop-in like
   L-BFGS.
2. The density-vs-accuracy non-monotonicity from issue #8 (lead #4
   there) is still open and now somewhat moot at this capacity — 64x3
   already clears the bar at the shipped 2000-point density, so
   re-sweeping density at 64-hidden isn't urgent, but the underlying
   "why is it non-monotonic" question wasn't answered by this issue.

## Does SOAP close the rest of the num_bands=4 gap? (issue #11)

Note: developed in parallel with issue #10 above — at the time this was
written, the `num_bands=4` gap was still open (issues #6/#8's
0.065-0.104). Issue #10 independently closed that same gap via network
capacity instead. Both are documented; SOAP is a genuinely different
mechanism (optimizer, not architecture) and remains a valid, separately
useful result even though it's no longer the only fix.

Issues #6/#8 deliberately stayed dependency-free (`torch.optim.LBFGS` only)
and got `num_bands=4` from a ~0.8 plateau down to 0.065-0.104 — a real
improvement, but short of the standard `0.1` bar (worst seed landed at
0.104). Khodakarami et al. report SOAP and SS-Broyden as substantially
stronger than L-BFGS for exactly this failure mode. This issue tests SOAP,
with a new dependency justified for the first time in this thread.

**Dependency adopted: `pytorch-optimizer` (PyPI, `kozistr/pytorch_optimizer`).**

- **What:** a 100+-optimizer PyTorch collection; we use exactly one class,
  `pytorch_optimizer.SOAP` (Vyas et al., "Improving and Stabilizing Shampoo
  using Adam", arXiv:2409.11321) — the official SOAP paper's own
  formulation, not a from-scratch reimplementation risk.
- **Why trusted:** actively maintained (v3.10.1, released within the last
  two months of this issue; 92 releases total, not a one-off drop), 421
  GitHub stars, CI + Codecov badges, Apache-2.0, single maintainer but a
  multi-year (since 2021) consistent release cadence — considered against
  the alternative of vendoring the paper authors' reference
  implementation (`nikhilvyas/SOAP`, which isn't published to PyPI and
  explicitly tells users to copy the file in, i.e. take on the
  maintenance burden ourselves) or `heavyball` (also credible, 334 stars,
  but younger project, more actively-changing API surface per its own
  migration-guide history). `pytorch_optimizer` was the better-established
  choice of the two PyPI-published options.
- **What it costs:** one new transitive-dependency-free package (only
  depends on `torch`/`numpy`, both already present); adds ~300KB wheel.
  We only use `SOAP`, none of the other 99+ optimizers it ships, so most
  of its surface area is dead weight we don't exercise or test — accepted
  because the alternative (reimplementing Shampoo-preconditioned SOAP
  in-repo) is far more maintenance risk than importing one class from an
  established package.

`train_fourier_cavity_soap` (`src/em_piml/train.py`) mirrors
`train_fourier_cavity_lbfgs`'s controlled-comparison setup exactly: same
`num_bands=4` `FourierCavityPINN`, same fixed (not resampled)
`n_collocation=2000/n_boundary=400/n_initial=400` point set, only the
optimizer swapped — `pytorch_optimizer.SOAP` (library defaults: `lr=3e-3`,
`betas=(0.95, 0.95)`, `precondition_frequency=10`) run for `steps=2000`
plain (no closure/line-search — unlike L-BFGS, SOAP doesn't need one).

**Result: fully closes the gap, doesn't just clear the bar.**

| seed | L-BFGS (issue #8) | SOAP (this issue) |
|---|---|---|
| 0 | 0.098 | 0.0357 |
| 1 | 0.104 | 0.0304 |
| 2 | 0.096 | 0.0233 |
| 7 | 0.065 | 0.0232 |

SOAP lands at 0.0232-0.0357 across seeds 0/1/2/7 — not merely under the
standard `0.1` bar, but in the *same range as the plain-coordinate Adam
baseline* (0.026-0.046, see top of this file) and tighter/more consistent
across seeds than L-BFGS's spread. This matches Khodakarami et al.'s
claim that SOAP is a stronger fix than L-BFGS for this exact spectral-bias
failure mode. `tests/test_fourier_soap.py` asserts `< 0.08` (~2.2-3.4x
margin above the observed 0.0357 worst case, matching the margin style of
`test_baseline_cavity.py`).

**Performance note (sandbox-specific, not a general claim):** SOAP
recomputes a Shampoo-style eigenbasis preconditioner every
`precondition_frequency` steps (extra `eigh` calls on top of an Adam-like
update), which made it dramatically more sensitive to CPU oversubscription
than Adam/L-BFGS when this issue was developed — on a contended sandbox
shared with unrelated concurrent processes, per-step wall time dropped
~200x (6.6s to 0.03s) after pinning `torch.set_num_threads(1)` for the
duration of the SOAP training call (see `_train_pinn_soap`), restored
afterward so it doesn't leak into other tests. 2000 steps takes ~60-80s
single-threaded, comparable to the existing L-BFGS test's ~40s. Revisit if
a quieter CI runner makes this pinning unnecessary or counterproductive —
it was not re-validated on an uncontended machine.

**Leads still open**, in case this is picked up further:
1. `SS-Broyden` (the paper's other top performer) wasn't tried — SOAP
   already met the bar, so there was no need to justify a second new
   dependency in the same issue.
2. SOAP hyperparameters (`lr`, `betas`, `precondition_frequency`) were
   left at library defaults — untuned. Given the result already beats
   the baseline range, tuning wasn't pursued, but there may be room to
   reduce `steps` below 2000 without losing accuracy.
3. Network capacity, embedding `num_bands` values other than 4, and the
   density non-monotonicity from issue #8 remain unexplored with SOAP —
   out of scope for this issue's specific question (does SOAP close the
   `num_bands=4` gap at the existing shipped point-set default).

## Does pseudo-sequence tokenization beat the raw-coordinate baseline? (issue #20)

Every embedding experiment up to this point (issues #4/#6/#8/#10/#11) varied
the Fourier-feature band count, the optimizer, or the network capacity — never
the *architecture* consuming the input. This issue tests the project's actual
motivating research question (tokenization/embedding for PIML, see issue #2's
origin) via a literature pass and a genuinely different axis: does turning each
pointwise `(x, t)` input into a short *token sequence*, processed by a small
Transformer, do better than a plain MLP?

**Literature pass (alphaXiv):** three tokenization schemes were evaluated for
fit against this project's 1D, single-fixed-equation, CPU-only scale:

1. **Equation tokenization** (Lorsung et al., "Physics Informed Token
   Transformer" / PITT, arXiv:2305.08757) tokenizes the *governing equation
   itself* as a symbolic sequence, to condition an operator-learning model
   across a *family* of parametric PDE instances (varying viscosity, forcing
   amplitude, etc.). Ruled out: em-piml's baseline is one fixed equation
   (fixed `L`, `c`, `n`) — there's no varying equation instance to tokenize or
   condition on. Would only become relevant if the project generalizes to a
   family of cavity problems.
2. **Patch-based multi-scale tokenization** (e.g. MeshTok, arXiv:2606.04366)
   adaptively patchifies large gridded 2D/3D PDE fields for Transformer
   foundation models (AMR-inspired, refining high-activity regions). Ruled
   out: em-piml's input is scalar `(x, t)` point samples, not a discretized
   field with spatial structure to patchify.
3. **Pseudo-sequence tokenization** (Zhao et al., "PINNsFormer", ICLR 2024,
   arXiv:2307.11833) expands each `(x, t)` into a short sequence of `k` nearby
   timesteps `{[x,t], [x,t+dt], ..., [x,t+(k-1)dt]}`, processed by a small
   encoder-decoder Transformer with a sequential PINN loss and a `Wavelet`
   activation (`omega1*sin(x) + omega2*cos(x)`, learnable `omega1`/`omega2` —
   the paper's own ablation found this necessary; ReLU/Sigmoid fail outright,
   plain `Sin` is inconsistent). This one fit: no equation family or spatial
   field needed, and their own closest benchmark (a 1D wave equation,
   `d^2u/dt^2 - beta^2 d^2u/dx^2 = 0`, sinusoidal IC, Dirichlet BCs) is
   essentially the same equation family as our cavity mode. Implemented as
   `PseudoSequenceCavityPINN` (`src/em_piml/model.py`) — no LayerNorm, per the
   same ablation (it didn't help and sometimes caused NaN paired with
   Wavelet); decoder has no self-attention, reusing the encoder's own
   embeddings as its query, per the paper's design.

**The sequential loss needed a non-obvious derivative trick.** The encoder's
self-attention mixes information across the `k` sequence positions, so a
naive `torch.autograd.grad(u, t_seq, grad_outputs=ones)` call sums
cross-position contributions instead of isolating each position's own
derivative (`d(u_j)/d(t_seq_j)`, per PINNsFormer eq. 5) — `_sequence_derivative`
(`src/em_piml/train.py`) extracts this per-position Jacobian diagonal via one
backward pass per sequence position. **Verified directly against finite
differences before trusting any result from it** (perturbed a single sequence
position's input, held the rest fixed, compared to the analytical gradient —
matched to 4 significant figures). This makes the sequential PDE residual (2nd
order in both `x` and `t`) cost `O(4k)` backward-style passes per collocation
batch, on top of the forward pass.

**Result: does not beat the baseline — performs markedly worse, and not for
lack of trying.** Shipped config (`train_pseudo_sequence_cavity`: `d_model=16,
heads=2, ff_dim=32, num_layers=1, k=3, dt=1e-3`, Adam `lr=3e-3`, `steps=600`,
`n_collocation=30/n_boundary=16/n_initial=16`) reaches relative L2 error
**0.958-1.383 across seeds 0/1/2/7** — worse than the raw-coordinate baseline's
0.026-0.046 by more than an order of magnitude, and comparable to or worse
than the total-failure `num_bands=4` Adam case from issue #4 (~1.0-1.04).

This was **not** a case of insufficient tuning — the following were all tried
and none closed the gap, each isolating a different candidate explanation:
- **A bug in the derivative math** — ruled out by the finite-difference check
  above.
- **Overfitting a small fixed L-BFGS collocation set** (this repo's own
  issue #6/#8 precedent for why L-BFGS can fail this way) — ruled out by
  switching to Adam with fresh point resampling every step (removes the
  "memorize a fixed set" failure mode entirely) and running up to 1500 steps;
  same ~1.0-1.4 plateau, reached within the first few hundred steps and flat
  thereafter despite the model's own training loss (PDE residual + BC + IC)
  converging to ~1e-4 — i.e. the model satisfies its own loss almost exactly
  while still not matching the analytical field.
- **`dt` mismatched to this problem's timescale** (paper's own benchmarks are
  mostly on `t` in `[0, 1]`; our `PERIOD` is ~6.28) — ruled out by rerunning
  with `dt` scaled up 6x; same outcome.
- **Insufficient training** — ruled out by both the loss-convergence evidence
  above and by denser collocation (tried up to 400 points) making no
  qualitative difference.

**Working interpretation:** the sequential loss's PDE-residual/BC/IC terms are
inherently local/pointwise constraints (as they are for the plain baseline
too) — satisfying them at every sampled point does not, by itself, guarantee
recovering the *unique* global solution, and this architecture apparently
finds an alternate function that satisfies them without matching the true
field elsewhere. The plain baseline avoids this in practice, but this
architecture's much higher per-step cost (`~15-30x` the baseline, from the
`O(k)` derivative extraction above) forces a far smaller model and point
budget within a CI-suitable wall-clock time, which may be part of why. It's
also worth noting PINNsFormer's own reported gain on their closest analog
(1D-wave) was modest without additional machinery: 0.335 (plain PINN) to 0.283
(PINNsFormer alone) — their headline 0.058 result required also adding NTK
adaptive loss-term reweighting on top, which neither this implementation nor
anything else in this repo currently has. A faithful implementation might
credibly have landed closer to "no clear win" even in the best case; landing
substantially *worse* than the baseline is the more surprising part of this
finding.

`tests/test_pseudo_sequence.py` asserts relative L2 error stays under `2.0` —
not an accuracy bar (there isn't one to clear here) but a regression check on
this documented negative result, with margin above the observed 1.383 worst
case. Determinism (same seed -> same result) was re-verified per this
project's standing lesson before trusting any of the numbers above.

**Leads for whoever picks this up next:**
1. NTK-style adaptive loss-term reweighting (Wang et al., cited by both
   PINNsFormer and issue #6's Khodakarami et al.) is the one lever the
   paper's own ablation shows matters most (0.283 -> 0.058) and is entirely
   unexplored in this repo — a more promising next step than further tuning
   the architecture in isolation.
2. The `O(k)` per-position Jacobian-diagonal extraction is the compute
   bottleneck (see above) — a vectorized/batched implementation (e.g. via
   `torch.func.jacrev`/`vmap`, untried here to keep this issue's scope to the
   tokenization question itself) could afford a much larger model/collocation
   budget within the same wall-clock time, which this issue couldn't rule out
   as the actual fix.
3. This toy problem is a single low-frequency mode with a known closed form —
   PINNsFormer's own headline results are on *harder* PDEs (high-frequency
   convection, PINN failure-mode benchmarks) where a plain MLP already
   struggles. It's possible pseudo-sequence tokenization's real benefit only
   shows up once the baseline itself is failing for a reason this
   architecture specifically addresses (spectral bias / temporal
   error-propagation), which isn't obviously true of our current baseline.

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
  see that section for the reasoning. Pseudo-sequence tokenization
  (PINNsFormer) was implemented and didn't beat the baseline; see leads
  #1-#3 in that same section for what's still open.
- The `num_bands=4/6` instability is now partially explained (see
  above) but not resolved — see the three leads above.

## Does a two-mode superposition break the baseline, and does Fourier
## embedding fix it? (issue #22)

Every embedding experiment up to issue #20 shared a blind spot: the
baseline problem (a single low-frequency fundamental mode, `n=1`) is
already solved well by a plain coordinate MLP (0.026-0.046 relative L2),
so there was no real failure mode for a fancier architecture to fix —
issue #4's Fourier features were "neutral," issue #20's pseudo-sequence
tokenization was actively worse, but neither result distinguished
"tokenization doesn't help" from "there was nothing here to help with."
This issue tests a harder target designed specifically to induce a
well-documented PINN failure mode: **spectral bias** (Rahaman et al.
2019; Xu et al.'s F-Principle, arXiv:1901.06523) — coordinate MLPs learn
low-frequency content fast and high-frequency content far more slowly,
or not at all.

`analytical_field_two_mode` (`src/em_piml/physics.py`) adds a second,
much higher spatial mode (`N_MODE_2=8`) on top of the existing
fundamental (`n=1`), equal amplitude (`0.5` each). This is still exact —
the wave equation is linear and both terms individually satisfy the
same Dirichlet BCs and `dE/dt(x,0)=0`, so their sum does too — no
numerical solver, no new dependency. `PERIOD`/`OMEGA` stay derived from
`N_MODE=1` only, so the existing training/eval domain (one period of the
fundamental) already spans 8 full spatial half-cycles and 8 full
temporal cycles of the added mode with no domain-size change.
`train_cavity_two_mode` and `train_fourier_cavity_two_mode`
(`src/em_piml/train.py`) reuse `train_cavity_baseline`/
`train_fourier_cavity_baseline`'s exact shipped defaults (steps,
`n_collocation`, `lr`, architecture, `num_bands=2`) — the *only*
variable is the target field, via a new `field_fn` parameter threaded
through `_pinn_loss`/`_train_pinn_adam`/`evaluate_relative_l2_error`
(defaults to the original `analytical_field`, so every existing call
site and test is bit-for-bit unaffected).

**Result: the baseline does fail here, exactly as predicted — and
Fourier features only partially help, also exactly as predicted.**

| model | relative L2 (seeds 0/1/2/7) |
|---|---|
| plain `CavityPINN` | 0.7699, 0.7944, 0.7947, 0.7876 |
| `FourierCavityPINN` (`num_bands=2`) | 0.7029, 0.6995, 0.7049, 0.7063 |

The plain baseline lands at ~0.77-0.79 — over an order of magnitude
worse than its own 0.026-0.046 range on the single-mode target, with
nothing else about the problem or training loop changed. Fourier
features give a real but small improvement (~0.70 vs ~0.78), nowhere
close to fixing it.

**A pointwise check confirms this is genuinely the spectral-bias
mechanism, not just a worse aggregate number.** Evaluating both trained
models at `t=0` across `x` values chosen to land on the `n=8` mode's
peaks/troughs (not its zero-crossings, which the two fields share):
both models' predictions track the smooth `n=1`-only envelope
(`0.5*sin(pi*x)`) almost exactly and are essentially blind to the true
field's `n=8` ripple (which swings roughly -0.4 to +0.99 at the same
points, vs. predictions that stay smoothly within the `n=1` envelope's
0.05-0.47 range). E.g. at `x=0.5625`: true field `0.9904`, `n=1`-only
component `0.4904`, plain-model prediction `0.4410`, Fourier-model
prediction `0.4687` — both models predicted almost exactly the `n=1`
envelope value and missed the `n=8` contribution entirely. This is
textbook spectral bias (cf. Tancik et al., "Fourier Features Let
Networks Learn High Frequency Functions in Low Dimensional Domains",
NeurIPS 2020, arXiv:2006.10739, Figure 3c, which shows the identical
shape of failure in a non-PDE 1D regression setting: a low-frequency
component converges while a simultaneously-present high-frequency
component doesn't).

**Why Fourier features only partially help, explained by the embedding's
own frequency support, not a mystery:** `FourierCavityPINN`'s embedding
(`src/em_piml/embeddings.py`) normalizes `x` to `[0,1]` and uses
frequencies `2^0*pi, 2^1*pi, ..., 2^(k-1)*pi` for `num_bands=k`. At the
shipped `num_bands=2` default, that's `{pi, 2pi}` — no basis component
at `8*pi`, which is exactly the frequency the `n=8` spatial mode needs
(`sin(8*pi*x/L)` becomes `sin(8*pi*u)` in normalized coordinates). The
pointwise check above confirms this precisely: the Fourier model's
predictions at the `n=8` peaks/troughs are nearly identical to the plain
model's, both tracking only the `n=1` envelope — `num_bands=2` doesn't
give the network anything new to work with for this specific target.
`num_bands=4` (frequencies up to `8pi`) would be the literature-predicted
fix; deliberately not tested here, per this issue's constraint to hold
the shipped `num_bands=2` default fixed and treat that as its own
finding rather than silently retuning mid-issue.

`tests/test_two_mode_superposition.py` asserts both models' relative L2
error stays `> 0.5` — not an accuracy bar (there's no bar to clear here)
but a regression check on this documented failure signature, with
~2.6-3.4x margin below the observed 0.70-0.79 range.

**Leads for whoever picks this up next:**
1. ~~Sweep `num_bands` up to `4`/`6`+ on this two-mode target~~ — done,
   see issue #25 below. Result: raising `num_bands` does **not** close
   this gap the way it closed the single-mode `num_bands=4` instability.
2. Issue #23 (long time-horizon/causality) is a queued, independent
   alternative failure mode — not blocked on this issue's outcome.
3. Now that there's a real, reproducible failure mode with a known
   mechanism (missing basis frequency), it's also a legitimate testbed
   to revisit issue #20's pseudo-sequence tokenization — PINNsFormer's
   own headline results are specifically on PDEs where a plain MLP
   already fails, which wasn't true of this project's original baseline.

## Does raising num_bands close the two-mode spectral-bias gap? (issue #25)

Issue #22 found `num_bands=2`'s missing `8*pi` basis frequency as the
likely reason it only partially fixes the two-mode target, and flagged
`num_bands=4` (whose basis `{pi, 2pi, 4pi, 8pi}` *does* include `8*pi`)
as the natural next test. This issue runs that test — and the result
overturns the naive "missing frequency" hypothesis rather than
confirming it.

**Step 1: plain Adam destabilizes at `num_bands>=4` on this target too,
exactly as issue #4 found on the single-mode baseline.** Using the
existing `train_fourier_cavity_two_mode` (32-hidden, Adam, unchanged
from issue #22) at `seed=0`:

| num_bands | relative L2 |
|---|---|
| 2 | 0.7029 |
| 4 | 1.0042 |
| 6 | 1.0309 |
| 8 | 1.0325 |

So testing `num_bands>=4` at all requires the optimizer/capacity fixes
this project already has from issues #6/#8/#10/#11 (L-BFGS or SOAP,
64-hidden for L-BFGS, `n_collocation=2000/n_boundary=400/n_initial=400`)
— unmodified from their shipped single-mode recipe, applied here via two
new functions mirroring the existing `train_fourier_cavity_two_mode`
pattern: `train_fourier_cavity_lbfgs_two_mode` and
`train_fourier_cavity_soap_two_mode` (`src/em_piml/train.py`). This
required threading `field_fn` through `_train_pinn_lbfgs`/
`_train_pinn_soap` the same way issue #22 already did for
`_train_pinn_adam`.

**Step 2: with stable optimizers, `num_bands=4` gives a small,
consistent improvement — but nowhere close to closing the gap. Going
past `4` doesn't help further, it destroys training.**
`src/em_piml/num_bands_sweep.py` (`uv run python3 -m
em_piml.num_bands_sweep`) sweeps `num_bands` in `{2, 4, 6, 8}` across
seeds `0/1/2/7`, both optimizers:

| num_bands | L-BFGS mean (stdev) | SOAP mean (stdev) |
|---|---|---|
| 2 | 0.7334 (0.0080) | 0.7302 (0.0055) |
| 4 | 0.7023 (0.0024) | 0.7128 (0.0021) |
| 6 | 1.0292 (0.0052) | 1.0757 (0.0691) |
| 8 | 1.0298 (0.0058) | 355.24 (292.3, range 1.02-751.55) |

Full per-seed numbers are in the sweep script's own output; the pattern
is tight and reproducible across all 4 seeds at every `num_bands`
value, not a one-seed artifact.

**This overturns the "missing `8*pi` basis frequency" hypothesis as
sufficient, though not as necessary.** `num_bands=4` does add the needed
`8*pi` component, and it does help slightly (0.70-0.73 down to
0.70-0.71) — consistently, under both optimizers, across all 4 seeds.
But it comes nowhere near the ~0.03 the single-mode baseline achieves,
and nowhere near the ~0.02-0.04 that L-BFGS/SOAP at `num_bands=4`
achieved on the single-mode target in issues #10/#11. Having the right
frequency available in the embedding is evidently necessary but not
sufficient — the network still can't learn to use it well on a target
that actually needs substantial weight on that frequency.

**And past `num_bands=4`, more embedding headroom actively destabilizes
training for *both* previously-robust optimizers** — `num_bands=6` and
`8` collapse to ~1.0-1.03 (worse than doing nothing) for L-BFGS, and
SOAP is even more extreme at `num_bands=8` (three of four seeds diverge
to relative L2 in the hundreds; the fourth stays near 1.0). This is a
different failure than the single-mode `num_bands=4/6` Adam instability
from issue #4, because L-BFGS/SOAP already *solved* that one (issues
#10/#11) — yet they fail here at a comparable or lower `num_bands`.

**Working interpretation:** issue #4/#6/#8/#10/#11's `num_bands=4`
instability was a network with *unused* high-frequency capacity next to
a purely low-frequency target — L-BFGS/SOAP could converge to a solution
that simply keeps those extra directions near zero. Here, the target
genuinely contains high-frequency content (the `n=8` mode), so the
network must actually learn nontrivial weight on the high-frequency
Fourier directions, not just avoid them — a qualitatively harder
optimization problem, consistent with Khodakarami et al.'s (issue #6)
NTK-eigenvalue-decay account of why gradient-based learning of
high-frequency content is intrinsically slow, not just an optimizer
implementation detail. Adding *more* unnecessary high-frequency
embedding dimensions on top of that (`num_bands=6/8`) doesn't give the
network more room to represent the `n=8` mode (which only needs `8*pi`) —
it just adds more directions for the optimizer to mismanage, and this
time there's no longer a "these directions can safely sit near zero"
escape hatch, because the loss landscape near the true (partially
high-frequency) solution is already harder to navigate.

`tests/test_num_bands_sweep.py` locks in the two headline findings as
regression checks (not accuracy bars — there's no bar to clear on this
target yet): `num_bands=4` under L-BFGS still exceeds the same `> 0.5`
failure-signature bound issue #22 used (observed 0.6986-0.7049), and
`num_bands=6` under L-BFGS exceeds `> 0.9` (observed 1.0205-1.0336,
confirming the collapse rather than an improvement). Both use L-BFGS
specifically (not SOAP) because its between-seed spread is tighter
(stdev <= 0.008 at every `num_bands` tested here, vs. SOAP's 0.069-292
at `num_bands=6/8`) — a more reliable single-seed regression signal.

**Leads for whoever picks this up next:**
1. This project has never tried widening capacity *specifically* for the
   two-mode target the way issue #10 did for the single-mode
   `num_bands=4` case — untried here, and not obviously the fix (the
   single-mode case was solving an "unused capacity" instability, not
   the "needs to actually learn high-frequency content" problem this
   target poses), but cheap to test given the machinery already exists.
2. NTK-based adaptive loss reweighting (see issue #20's leads, and
   Khodakarami et al.'s own mitigation guidelines in issue #6) remains
   the most literature-direct unexplored lever for a target that
   genuinely needs learned high-frequency content, as opposed to
   optimizer/capacity fixes aimed at unused-capacity instability.
3. Why does `num_bands=8` SOAP diverge so much harder and so much more
   seed-dependently (163-751 on 3 seeds, ~1.0 on the 4th) than
   `num_bands=8` L-BFGS (tight 1.02-1.03 across all 4 seeds)? Untouched
   here — could be about SOAP's Shampoo-style preconditioner behaving
   badly on an ill-conditioned Hessian at this embedding dimensionality,
   but this issue didn't dig into why.
4. Issue #23 (causality) remains an independent, unblocked alternative
   failure mode to this whole `num_bands` thread.

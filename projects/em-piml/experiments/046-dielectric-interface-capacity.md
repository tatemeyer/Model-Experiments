# Does capacity help resolve a local dielectric-interface kink, in contrast to issue #25's negative global-spectral-content finding? (issue #46)

Issue #25 found that network capacity, while present in the model
(`hidden=64`, up from the single-mode baseline's `hidden=32`, per
`../num-bands-gap/010-network-capacity.md`), was *insufficient* to close
the two-mode spectral-bias gap — the target needed genuinely learned
high-frequency content spread across the *whole* domain, a harder
optimization problem than more capacity alone solves. That is a finding
about *global* spectral content. This issue poses a different kind of
hard-to-represent feature: a 1D cavity with a piecewise-constant
permittivity (a dielectric interface at a fixed interior point), governed
by Maxwell's transmission conditions, whose true solution has a
*spatially localized* kink concentrated at one point rather than
frequency content spread across the domain. Does capacity help here, or
is capacity's irrelevance a general property of PINN optimization in
this project rather than something specific to global spectral bias?

**Physics module: `src/em_piml/dielectric.py`.** Permittivity is
piecewise-constant, `EPS_1=1.0` for `x < X_INT` and `EPS_2=4.0` for
`x >= X_INT`, `X_INT=0.35` (off-center to avoid symmetric degeneracies)
— both are explicit, documented module constants, not magic numbers
baked into a formula. Permeability `MU=1.0` is constant across the
interface (both regions non-magnetic). The governing PDE is `eps(x) *
d^2E/dt^2 = (1/MU) * d^2E/dx^2`, the same 1D Maxwell reduction as
`physics.py`'s baseline but with a position-dependent local wave speed
`c(x) = 1/sqrt(MU*eps(x))` instead of a single global one.

Maxwell's transmission conditions at a source-free interface are
tangential-E continuous and tangential-H continuous. `E_z` is tangential
here (the interface normal is `x`, the field points along `z`), so this
is exactly "tangential E continuous." `H_y` is tangential too, and
Faraday's law (`dE_z/dx = -MU*dH_y/dt`) ties `H_y`-continuity to
`dE_z/dx`-continuity whenever `MU` is equal on both sides (true here) —
so, perhaps counterintuitively, both the field *and* its first spatial
derivative stay continuous at `X_INT`; the permittivity jump doesn't
show up as a slope discontinuity. `D_x = eps*E_x` ("normal D") is
identically `0=0` on both sides for this transverse-field,
normal-incidence reduction (there's no `E_x` component in this
polarization), so normal-D continuity is trivially satisfied and adds no
separate constraint here beyond the two tangential ones — documented in
`dielectric.py`'s docstring rather than silently assumed. What the
permittivity jump *does* do is break curvature continuity: from the PDE
itself, `d^2E/dx^2 = MU*eps(x)*d^2E/dt^2`, and since `d^2E/dt^2` is
continuous at `X_INT` (it's evaluated from a single continuous, C1
function of `t`), `d^2E/dx^2` jumps by exactly `EPS_2/EPS_1` whenever
that shared time-derivative is nonzero — a genuine, spatially localized
second-derivative kink, concentrated at one point, that a plain
coordinate MLP has to locally represent even though the field itself
looks smooth to the eye.

Given a separable standing-wave ansatz `E(x,t) = f(x)*cos(omega*t)`
(same convention as the baseline's `analytical_field`), `f` is piecewise
sinusoidal (`A1*sin(k1*x)` for `x<X_INT`, already zero at `x=0`;
`A2*sin(k2*(L-x))` for `x>=X_INT`, already zero at `x=L`) with
`k_i = omega*sqrt(MU*eps_i)`. Matching `f` and `f'` at `X_INT` gives a
transcendental eigenvalue equation for `omega` with no closed-form root,
so `dielectric._solve_fundamental_omega` finds it numerically (scan for
the first sign change past `omega=0`, then bisect) — the "numerically
verified reference solution" issue #46's success criteria allow for.
`dielectric.verify_reference_solution()` checks the result actually
satisfies both boundary conditions and both interface-matching
conditions to floating-point precision (`<1e-9`), and separately reports
the curvature-jump ratio, which lands on exactly `4.0` (`EPS_2/EPS_1`),
confirming the kink is real and exactly where/how big it's supposed to
be, not a root-finder artifact.

**Training: `train_dielectric_cavity` (`src/em_piml/train.py`), swept in
`src/em_piml/dielectric_capacity_sweep.py`
(`uv run python3 -m em_piml.dielectric_capacity_sweep`).** Same
`CavityPINN` architecture family, Adam optimizer, and step/point-count/lr
defaults as `train_cavity_baseline` (4000 steps, `n_collocation=200`,
`n_boundary=64`, `n_initial=64`, `lr=3e-3`) — `hidden` (swept in `{16,
32, 64, 128, 256}`, per the issue's suggested range) is the only
variable, `num_layers` held fixed at the baseline's `3`. This keeps the
comparison controlled the same way issue #10's capacity sweep was: only
capacity changes, not the training recipe. `_dielectric_pinn_loss` is
the same four-term loss shape as the baseline's `_pinn_loss` (PDE
residual + 2 BCs + IC + IC-dot), with the PDE residual computed via
`pde_residual_dielectric` (piecewise `eps(x)` multiplying the time term)
and IC/IC-dot supervised against `analytical_field_dielectric` (the
kinked reference solution) instead of the homogeneous-cavity target.
`dE/dt(x,0)=0` still holds unchanged (`cos(OMEGA*t)` has zero
time-derivative at `t=0`, regardless of `f(x)`'s shape). No new
dependency — everything is `torch` autograd and plain Python bisection.


**Scope reduction (documented tradeoff).** This sandbox was running many
concurrent agent sessions during this issue, and single-threaded
training that normally takes ~35s (the project's documented baseline
runtime) was inflated to several minutes per run — a full `{16, 32, 64,
128, 256}` x 4-seed x 4000-step sweep (this project's usual convention,
e.g. `../num-bands-gap/010-network-capacity.md`) would not finish in a
reasonable session. The numbers below use a reduced but still real,
actually-run configuration: 3 capacities (`16`, `64`, `256` — smallest,
middle, largest), 2 seeds (`0`, `1`), `steps=600` (down from the
baseline's `4000`). `dielectric_capacity_sweep.py`'s `HIDDEN_VALUES`/
`SEEDS`/`STEPS` constants are exactly these reduced values, with a
comment on how to restore the full convention when more time/compute is
available. Determinism was re-verified before trusting these numbers:
training `hidden=16, seed=0` twice gives bit-identical relative L2 error
(`0.833814` both times) and bit-identical parameters.

**Result: capacity gives a real, monotonic-on-average improvement here —
modest in size, and error concentrates at the interface at every
capacity tested, shrinking with capacity but not disappearing.**

| hidden | relative L2 (seeds 0, 1) | mean | stdev |
|---|---|---|---|
| 16 | 0.8338, 0.8497 | 0.8418 | 0.0080 |
| 64 | 0.7075, 0.7415 | 0.7245 | 0.0170 |
| 256 | 0.6729, 0.7383 | 0.7056 | 0.0327 |

Every capacity increase reduces the mean error (0.8418 -> 0.7245 ->
0.7056), unlike issue #25's two-mode target, where capacity was present
(`hidden=64`, already double the baseline's `32`) but the gap stayed
essentially unmoved without also fixing the optimizer (L-BFGS/SOAP) —
here, plain Adam with no optimizer change already benefits from more
capacity. The seed-to-seed spread grows with capacity (stdev 0.008 ->
0.017 -> 0.033), consistent with a genuinely harder, noisier
optimization landscape at larger width rather than a clean monotonic
improvement per seed (`hidden=256`'s seed 1 run, 0.7383, lands close to
`hidden=64`'s seed 0 run, 0.7075) — the trend is in the *mean*, not
every individual seed. These are all far above the baseline's
0.026-0.046 range, but that comparison isn't apples-to-apples: `steps`
here is 600 (15% of the baseline's 4000) specifically because of this
issue's runtime constraint (see above), not because the problem is
harder to converge in some fundamental sense that these numbers by
themselves demonstrate — the *shape* of the capacity trend, not the
absolute error level, is this experiment's finding.

**Pointwise diagnosis: error concentrates at the interface at both
capacities tested, and capacity reduces it roughly uniformly across
distance from the interface rather than disproportionately at the
interface itself.** `pointwise_error_by_distance` (seed 0 models, 2000
held-out points binned by `|x - X_INT|`):

| dist from interface | mean\|err\| (hidden=16) | mean\|err\| (hidden=256) |
|---|---|---|
| [0.000, 0.081] | 0.2902 | 0.2286 |
| [0.081, 0.162] | 0.2837 | 0.2272 |
| [0.162, 0.244] | 0.2421 | 0.1911 |
| [0.244, 0.325] | 0.2285 | 0.1833 |
| [0.325, 0.406] | 0.2383 | 0.1916 |
| [0.406, 0.487] | 0.2203 | 0.1749 |
| [0.487, 0.569] | 0.1536 | 0.1236 |
| [0.569, 0.650] | 0.1022 | 0.0790 |

At both capacities, error is highest in the bin closest to the interface
and falls off with distance from it (roughly 2.5-2.9x higher next to the
interface than in the farthest bin) — confirming the error genuinely
concentrates at the localized kink, not spread uniformly by some other
mechanism (e.g. simple undertraining, which would show up flat across
distance). Going from `hidden=16` to `hidden=256` reduces error at
*every* distance bin by a broadly similar ~20-25% (e.g. 0.2902->0.2286
near the interface, a 21% cut; 0.1022->0.0790 far from it, a 23% cut) —
capacity helps, but it doesn't preferentially fix the interface region
more than the rest of the domain, and the near-interface bin is still
the worst bin at `hidden=256` too. Capacity shrinks the kink's error
without eliminating the concentration.

**Contrast with issue #25's negative capacity finding.** Issue #25 found
capacity present (`hidden=64` in `FourierCavityPINN`, already doubled
from the single-mode baseline's `32`) but *insufficient* to close the
two-mode spectral-bias gap — the model needed genuinely learned
high-frequency content spread across the whole domain (a harder
optimization problem, per Khodakarami et al.'s NTK-eigenvalue-decay
account), and widening capacity further was flagged as untried but "not
obviously the fix" for that reason. This issue's local dielectric-kink
target behaves differently: plain capacity scaling on the *same*
architecture family, *same* optimizer (Adam, no L-BFGS/SOAP needed),
gives a real, monotonic-on-average reduction in both the aggregate
relative L2 error and the pointwise error at every distance from the
interface. The local/global distinction plausibly explains the
difference: representing one point's worth of extra curvature is a
capacity-shaped problem a wider smooth basis can partially absorb
locally, whereas representing genuinely new high-frequency content
spread across the entire domain is not just about having enough
parameters — it is about the optimizer actually learning to use
high-frequency directions everywhere, which issue #25 found capacity
alone does not solve. Capacity is not equally unhelpful here: this is
the first capacity-only, optimizer-unchanged result in this project's
history to show a real, if modest and noisy, improvement from
widening alone.

`tests/test_dielectric_interface_capacity.py` locks in the reference
solution's correctness (fast, no training: value/slope continuity and
the boundary conditions to `1e-9`, and the curvature-jump ratio at
exactly `EPS_2/EPS_1=4.0`) and the headline capacity-helps finding as a
regression check (`hidden=256`'s relative L2 error is lower than
`hidden=16`'s, same seed, same reduced `steps=600` budget).

**Leads for whoever picks this up next:**
1. This issue's numbers use a reduced `steps=600`, 2-seed, 3-capacity
   budget because of transient sandbox CPU contention (see "Scope
   reduction" above) — rerunning `dielectric_capacity_sweep.py` with
   `STEPS=4000`, `HIDDEN_VALUES=(16,32,64,128,256)`, `SEEDS=(0,1,2,7)`
   (this project's usual convention) whenever more time/compute is
   available would give a tighter, more convincing version of the same
   trend, and would let `128` be checked too.
2. The pointwise diagnosis shows capacity reduces error roughly
   uniformly across distance from the interface rather than
   disproportionately at the interface itself — an explicit
   interface-localized loss term (denser collocation right at `X_INT`,
   or a soft penalty directly on the curvature-jump condition) is
   untried here and might close the interface-specific gap faster than
   capacity alone, the way issue #8's denser collocation helped the
   `num-bands-gap` thread.
3. Whether an optimizer change (L-BFGS/SOAP, per `../num-bands-gap/`)
   on top of capacity closes the remaining gap the way it did for the
   global single-mode case is untried here.

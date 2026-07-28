# Does a Neuro-Spectral Architecture (NeuSA) fix the long-horizon causality collapse? (issue #36)

Every fix tried against the long-horizon collapse so far (causal
reweighting #23, pseudo-sequence tokenization #30, curriculum training
#32, NTK-based adaptive loss reweighting #34) was a modification layered
on top of the same plain coordinate-input MLP (`CavityPINN`). Bizzi et
al., "Neuro-Spectral Architectures for Causal Physics-Informed Networks"
(NeurIPS 2025, [arXiv:2509.04966](https://arxiv.org/abs/2509.04966))
propose a genuinely different architecture family: project the PDE onto
a spectral basis, then integrate the resulting finite-dimensional
dynamics with an adapted Neural ODE. The paper reports this
simultaneously overcomes spectral bias (via the spectral basis) *and*
enforces causality (by inheriting the Neural ODE's inherently causal,
sequential integration structure), validated on wave-equation
benchmarks — the same PDE family as this project's cavity mode. This
issue tests NeuSA against this project's long-horizon collapse (issue
#23).

## Mechanism

Project `E(x,t)` onto the Dirichlet sine basis `b_k(x) = sin(k*pi*x/L)`,
`k=1..num_modes` — satisfies `E(0,t)=E(L,t)=0` for every `t` by
construction, no BC loss term ever needed. The wave equation
`E_tt = c^2 * E_xx` reduces (method of lines) to a first-order ODE for
spectral coefficients `(a_k(t), w_k(t)=da_k/dt)`:

```
d/dt [a; w] = A[a; w] + eps * F_theta(a, w),   A = [[0, I], [D, 0]],   D = -diag(omega_k^2)
```

`omega_k = k*pi*C/L` — the paper's own worked wave-equation example
(Appendix E). Since this project's PDE is genuinely linear/
translation-invariant, `A` is the *exact* dynamics matrix, not an
approximation the paper has to linearize around for a general nonlinear
PDE — `F_theta` (a small MLP, `_mlp([2*num_modes, hidden, hidden,
num_modes])`, scaled by a small `eps`) only has to learn a zero
correction for a perfect fit to exist. Only the velocity equation
(`dw/dt`) gets the learned correction — `da/dt = w` is a definitional
identity from converting a 2nd-order PDE to first-order form, matching
the paper's own `A` matrix having an exact, structural `I` block, not a
learned one.

Initial coefficients `(a(0), w(0))` are the *exact* numerical projection
of the true initial condition and its time-derivative onto the basis
(`_project_onto_sine_basis`, trapezoidal quadrature, `n_quad=2000`), not
learned — so IC compliance is exact and architectural, same as BC. **3
of the 4 loss terms every prior long-horizon experiment trained against
(BC, IC, `dE/dt(x,0)=0`) are eliminated by construction** — only a
PDE-residual-equivalent term remains.

**Training-in-coefficient-space simplification (derived and verified
empirically, not just asserted):** because the sine basis is orthogonal
on `[0,L]`, and only the velocity-coefficient dynamics carry the learned
correction, the physical-space PDE residual `E_tt - c^2 * E_xx`
integrated over `x` reduces, up to a constant factor, to the sum of
squared correction-term outputs along the trajectory — so training
minimizes `mean(correction(states)**2)` directly in coefficient space
(`_train_pinn_neusa`), never reconstructing the field at sampled `x`
points for the loss. This is legitimate specifically because the PDE is
linear and the basis is orthogonal — not a general-purpose PINN training
trick (the paper's own general loop does reconstruct in physical space
for nonlinear PDEs).

Integration is hand-rolled fixed-step RK4 (`_rk4_integrate`) — no new
dependency (`torchdiffeq`/`torchdyn` weren't needed for a small
fixed-dimensional near-linear ODE). RK4's stability region crosses the
imaginary axis at `|z| ~ 2.83`; this ODE's eigenvalues are `±i*omega_k`,
max at `k=num_modes`, so stability requires
`h < 2.83 / (num_modes*pi)` (`C=L=1`). This was verified empirically
(no NaN/blowup at every configuration tried below), not just trusted
from the derivation.

## Implementation and a critical runtime-budget deviation from the initial plan

`NeuSACavityPINN`/`_sine_basis`/`_project_onto_sine_basis`/
`_rk4_integrate`/`_interpolate_trajectory` (`src/em_piml/model.py`) and
`_train_pinn_neusa`/`train_cavity_neusa_long_horizon`
(`src/em_piml/train.py`) implement the mechanism above. Shipped config:
`num_modes=8` (spans this project's fundamental `n=1` mode and the
`n=8` mode used by the optional two-mode secondary check below),
`hidden=16`, `eps=0.1`, `steps_per_unit_time=20` (gives `h=0.05`, a
~2.2x stability margin at `num_modes=8`, `h_max=0.1125`), `lr=3e-3`.

**The originally planned defaults (`steps=1000`, training integrated
over the full `t_max = horizon_periods * PERIOD = 5*PERIOD` matching
every prior long-horizon function's train==eval-horizon convention) are
computationally infeasible for this project's CI budget and were
changed.** Measured directly: at those defaults, a single training step
(one full RK4 unroll over `n_steps=200` sub-steps, forward + backward)
took **~5.3 seconds** — a full `steps=1000` run would take **~88
minutes**, far beyond every other "slow" test in this project (35-100s).
The cost is dominated by backpropagating through the `O(n_steps)`
sequential Python-level RK4 loop (effectively a ~800-op-deep computation
graph per training step, since each RK4 step calls the correction MLP 4
times) — not FLOPs (the correction MLP is tiny), but per-op
autograd/Python dispatch overhead compounding over a long sequential
unroll.

**Fix: decouple the training integration horizon from the evaluation
horizon.** `train_cavity_neusa_long_horizon` gained a
`train_horizon_periods` parameter (default `1.0`, i.e. train by
integrating only one period) independent of `horizon_periods` (default
`5.0`, used only for the caller's eval `t_max`, exactly as in every
other `train_cavity_*_long_horizon` function). This is legitimate
specifically for NeuSA and not for any plain-MLP variant in this
thread: the `(a_1, w_1)` phase-space trajectory of this LTI system is a
periodic circular orbit, so training beyond ~1 period revisits the same
manifold of states the correction net has already seen — it isn't new
information the way sampling more of the time domain is for a
coordinate-input MLP (whose BC/IC/PDE terms are pointwise constraints
that must be satisfied *at* whatever points get sampled, which is why
every prior fix in this thread needed the full domain covered one way
or another). `NeuSACavityPINN.forward` re-integrates fresh from
`state0` for whatever `t_max` a query implies, entirely independent of
what horizon was used during training — this is exactly the paper's
claimed architectural-extrapolation property, not an assumption; the
empirical result below (trained on 1 period, evaluated at 5 periods,
solves it) is the direct test of that claim, and confirms it. Cutting
the training horizon 5x (and `steps` down to `100`, see below) brings a
full training run down to roughly ~1-2 minutes — see the per-seed
timings recorded in the regression test/results.csv.

`steps=100` (down from the plan's `1000`) was also empirically
sufficient: the correction MLP's target is exactly zero everywhere (the
`A` matrix alone is the exact solution), the easiest possible fitting
problem, so it converges fast. This was verified directly (see below),
not just assumed cheaper.

**A second, orthogonal runtime fix was also needed: single-threading.**
`_train_pinn_neusa` pins `torch.set_num_threads(1)` for the duration of
training (restored afterward), the same fix this project already applies
to `_train_pinn_soap` (issue #11) and
`_train_pseudo_sequence_pinn_adam` (issue #20) for the identical reason:
this workload is dominated by many small sequential ops (the RK4 unroll
calls the tiny correction MLP 4 times per sub-step; backprop walks the
whole chain), so default intra-op threading overhead dominates and this
measured highly sensitive to this sandbox's CPU oversubscription from
concurrent agent sessions (see CLAUDE.md issue #10's identical note).
With this project's default threading, a single training run's wall
time was observed varying wildly (tens of minutes) purely as a function
of how many other concurrent processes happened to be running on the
shared sandbox at the time — not a change to the algorithm or config.
Pinned to one thread, a full `train_cavity_neusa_long_horizon()` run
(`steps=100`, `train_horizon_periods=1.0`) measured **~105-135s** per
seed on this contended sandbox — still somewhat above this project's
typical "slow" test (35-100s), but in the same order of magnitude, and
per this project's own established precedent (issues #10/#11), these
exact wall-clock numbers are sandbox-load-dependent and not expected to
reproduce bit-for-bit on a quieter CI runner; rerun rather than assume a
regression if they differ substantially.

## Results

**Result: essentially solves it.** Unlike every prior fix in this
thread (all partial improvements at best, one actively worse), NeuSA
lands nearly three orders of magnitude below every prior approach.

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| uniform (issue #23) | 0.9225, 0.9255, 0.9229, 0.9249 |
| causal, epsilon=1.0 (issue #23) | 0.9251, 0.9230, 0.9236, 0.9249 |
| pseudo-sequence (issue #30) | 0.9973, 1.0844, 0.9792, 1.1015 |
| curriculum (issue #32) | 0.9141, 0.9120, 0.8985, 0.9111 |
| NTK-reweighted (issue #34) | 1.0089, 1.1336, 0.9999, 1.0275 |
| NeuSA (this issue) | 0.002300, 0.002245, 0.002289, 0.002259 |

NeuSA lands at 0.00225-0.00230 across seeds 0/1/2/7 — remarkably tight
(all four seeds within 2.5% of each other) and roughly 400x smaller
than the best prior fix (curriculum training's 0.8985-0.9141). This
matches the plan's expectation going in: the PDE is exactly LTI and the
single-mode target is exactly representable by the sine basis with a
correction that only has to learn to output ~0, so a near-perfect fit
exists and is easy to reach — no fix in this thread that only changed
representation, loss weighting, or training schedule around the same
plain coordinate MLP came close, because none of them removed the
actual degenerate-collapse escape hatch (a near-constant function
trivially satisfying the wave-equation residual). NeuSA doesn't need to
avoid that escape hatch — it doesn't have BC/IC/PDE-residual loss terms
pulling against each other in the first place; the only thing being
optimized is a correction the exact solution already sets to zero
everywhere. Not literally machine precision (~1e-3, not ~1e-7) — the
residual ~0.002-0.003 is consistent with a combination of the RK4
integrator's own O(h^4) local truncation error at the shipped `h=0.05`
and the correction MLP's own residual weight-decay-free convergence,
neither of which was tuned to push further given this result already
solves the problem by any practical standard.

## Pointwise verification

Evaluating the trained seed-0 model at `x=0.5` across increasing `t`
(same checkpoints issues #23/#30/#32/#34 used, `PERIOD=2`):

| t | true field | predicted |
|---|---|---|
| 0 (`0*PERIOD`) | 1.0000 | 1.0000 |
| 0.5 (`0.25*PERIOD`) | -0.0000 | 0.0000 |
| 1.0 (`0.5*PERIOD`) | -1.0000 | -1.0000 |
| 2.0 (`1.0*PERIOD`) | 1.0000 | 1.0000 |
| 3.0 (`1.5*PERIOD`) | -1.0000 | -1.0000 |
| 4.0 (`2.0*PERIOD`) | 1.0000 | 1.0000 |
| 6.0 (`3.0*PERIOD`) | 1.0000 | 1.0000 |
| 8.0 (`4.0*PERIOD`) | 1.0000 | 0.9999 |
| 9.8 (`4.9*PERIOD`) | 0.8090 | 0.8089 |

**No collapse anywhere in the domain — the collapse is fully avoided,
not just delayed or reduced.** Every prior fix in this thread (issues
#23/#30/#32/#34) tracked the true field closely only near `t=0` before
decaying toward a near-constant plateau within roughly one period, with
curriculum training's delayed-collapse-to-~1.5-2*PERIOD (issue #32) the
best of them. NeuSA matches the true field to 3-4 significant figures at
*every* checkpoint tested, including the very last one at `t=4.9*PERIOD`
— there is no point in the domain where prediction and truth visibly
diverge. This is a qualitatively different outcome, not a quantitative
improvement on the same collapse: the mechanism that caused every prior
architecture to find the trivial near-constant solution (a near-constant
function trivially satisfies the wave-equation residual, so gradient
descent finds it as a "free" low-loss solution) has no equivalent escape
hatch here — the exact solution the correction net needs to converge to
is "output zero everywhere," not "avoid a degenerate attractor while
also satisfying pointwise BC/IC constraints elsewhere," and there is no
alternative low-loss solution for the optimizer to find instead.

## Determinism

Re-verified per this project's standing rule: same seed (0) run twice
produces bit-identical model parameters and identical relative L2 error
— see `tests/test_neusa_long_horizon.py`'s
`test_neusa_long_horizon_is_deterministic`.

`tests/test_neusa_long_horizon.py` locks in the finding:
`test_neusa_solves_long_horizon_collapse` asserts relative L2 error
`< 0.05` — a ~20x margin above the observed 0.00225-0.00230 range
(loose enough to tolerate ordinary run-to-run/seed variance without
flaking, tight enough to catch any regression back toward the ~0.9-1.1
failure range every prior fix landed in).
`test_neusa_long_horizon_is_deterministic` (fast, not `@pytest.mark.slow`
— uses `steps=5`, just enough to exercise one optimizer step) re-verifies
same-seed bit-identical parameters as a permanent regression check,
separate from the one-off manual verification above.

**Leads for whoever picks this up next:**
1. The optional secondary two-mode check (`analytical_field_two_mode`,
   issue #22/#25) was **not attempted** — the primary long-horizon
   result already needed real runtime-budget debugging (see the
   deviation section above) and this issue's time was spent making that
   solid rather than also pursuing the optional secondary check.
   `num_modes=8` (the shipped default) already exactly spans the
   two-mode target's `n=1`/`n=8` modes, so trying `field_fn=
   analytical_field_two_mode` against `train_cavity_neusa_long_horizon`
   directly is cheap to attempt next — genuinely interesting given
   issues #22/#25 found the two-mode target induces spectral bias in
   every plain-MLP variant tried, a different pathology than the
   long-horizon collapse this issue addressed.
2. `train_horizon_periods=1.0` was chosen as "one period," not swept —
   whether an even shorter training horizon (a fraction of a period)
   still generalizes, or whether some PDEs/targets in this project
   would need more than one period of training coverage, is untested.
3. `hidden=16`/`eps=0.1`/`steps=100`/`steps_per_unit_time=20` were
   chosen empirically to fit this project's CI budget while solving the
   problem, not swept for a minimal sufficient configuration — there is
   likely room to go even cheaper (fewer steps, fewer modes for the
   single-mode-only case) without losing accuracy.
4. This is the first fix in the long-horizon thread that does more than
   partially help — worth revisiting whether NeuSA (or its
   coefficient-space training simplification specifically) generalizes
   to this project's other open failure modes (two-mode spectral bias,
   issues #22/#25) where the target is no longer a single LTI mode the
   architecture's basis exactly spans without any correction needed.

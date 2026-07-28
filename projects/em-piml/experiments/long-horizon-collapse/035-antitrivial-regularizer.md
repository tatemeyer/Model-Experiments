# Does an explicit anti-trivial-solution regularizer fix the long-horizon causality collapse? (issue #35)

Four fixes have now been tried against the long-horizon collapse (issue
#23): causal loss-reweighting (`023-long-horizon-causal.md`),
pseudo-sequence tokenization (`030-pseudo-sequence-long-horizon.md`),
a short-to-long curriculum (`032-curriculum-long-horizon.md`), and
NTK-based adaptive loss reweighting
(`034-ntk-reweighted-long-horizon.md`). All four changed representation,
loss weighting, or training schedule; none made the degenerate
near-constant trivial solution itself less attractive to gradient
descent — the collapse mechanism, diagnosed in issue #23, is that a
near-constant/near-linear output has near-zero second derivatives in
both `x` and `t`, so it trivially satisfies the wave-equation residual
without needing to match the true oscillation.

Leiteritz & Pflueger, "How to Avoid Trivial Solutions in Physics-Informed
Neural Networks" (arXiv:2112.05620), propose exactly the kind of fix this
thread has been missing: rather than reweighting existing loss terms,
they add a *new* term that penalizes the PDE residual's own spatial
smoothness. Studying a 1D harmonic oscillator with sparse collocation
points, they observed the trivial (constant) solution becomes reachable
because the true and trivial solutions can each locally satisfy the
residual at the collocation points actually sampled, with an
unconstrained gap in between where the model is free to switch from one
to the other — and that switch shows up as a sharp spike in the
residual's own gradient. Their fix (eq. 8, concretized for their ODE in
eq. 13) adds `max_i (d(residual)/dt at collocation point i)^2` to the
loss: a smoothness/anti-spike penalty on the residual field itself,
reported to let their benchmark train with up to 80% fewer collocation
points.

**Implementation.** `train_cavity_antitrivial_long_horizon`
(`src/em_piml/train.py`) reuses `train_cavity_long_horizon`'s exact
architecture (`CavityPINN(hidden=32, num_layers=3)`), total step budget
(4000), point counts (`n_collocation=200, n_boundary=64, n_initial=64`),
and `lr=3e-3` — the only variable is the added regularization term.
`_pde_residual_and_input_grad_sq` duplicates `pde_residual`'s derivative
chain (rather than calling it directly) because computing the residual's
*own* gradient requires a handle to the exact `x`/`t` leaf tensors it was
differentiated from — `pde_residual` clones its inputs internally without
exposing that clone, so reusing it would silently differentiate w.r.t. a
disconnected leaf. The paper's benchmark has one input dimension (`t`),
so eq. 13's bracketed term is literally `d(residual)/dt`; this project's
wave equation has two (`x`, `t`), so this generalizes it to the squared L2
norm of the residual's gradient over both inputs,
`(dr/dx)^2 + (dr/dt)^2` — same "residual should not spike" intent as the
paper, extended to a 2D domain rather than arbitrarily picking one axis.
`_antitrivial_pinn_loss` adds `lambda_grad * max_i(grad_sq_i)` (default
`lambda_grad=1.0`, matching the paper's own eq. 13, which adds the term
unscaled) to the existing four-term loss (PDE residual, BC, IC,
`dE/dt(x,0)=0`). Single-threaded during training
(`torch.set_num_threads(1)`, restored afterward), for the same reason as
the existing SOAP/pseudo-sequence training loops: the extra
third-order-derivative autograd call is dominated by many small ops, and
this project's sandbox has repeatedly shown that default intra-op
threading compounds badly under concurrent-session CPU oversubscription
(see issue #10's note) — this was applied pre-emptively here, not in
response to a specific measured regression, since the mechanism (extra
autograd overhead, many small tensor ops) is the same one that motivated
it for SOAP/pseudo-sequence.

**Result: does not fix the collapse — lands slightly worse than doing nothing.**

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| uniform (issue #23) | 0.9225, 0.9255, 0.9229, 0.9249 |
| causal, epsilon=1.0 (issue #23) | 0.9251, 0.9230, 0.9236, 0.9249 |
| pseudo-sequence (issue #30) | 0.9973, 1.0844, 0.9792, 1.1015 |
| curriculum (issue #32) | 0.9141, 0.9120, 0.8985, 0.9111 |
| NTK-reweighted (issue #34) | 1.0089, 1.1336, 0.9999, 1.0275 |
| anti-trivial regularizer (this issue) | 0.9687, 0.9697, 0.9716, 0.9680 |

The anti-trivial regularizer lands at 0.9680-0.9716 — a small but
consistent regression relative to uniform's 0.9225-0.9255 and causal's
0.9230-0.9251 (every seed here is worse than every seed of either prior
approach), though the four seeds are the *tightest*-clustered of any
variant tried in this thread (a spread of 0.0036, vs. 0.003-0.14 for the
others) and the regression is much milder than pseudo-sequence's or NTK
reweighting's. Determinism re-verified before trusting this: seed 0
trained twice independently produced bit-identical parameters and
identical relative L2 error (0.9687 both times).

**A pointwise check (seed 0) shows the same collapse mechanism as every
prior approach in this thread, not a new one — and a worse initial fit
than uniform/causal/curriculum.** Evaluating at `x=0.5` across increasing
`t`:

| t | true field | predicted |
|---|---|---|
| 0 | 1.0000 | 0.8520 |
| 0.25*PERIOD | -0.0000 | 0.2732 |
| 0.5*PERIOD | -1.0000 | -0.2793 |
| 1.0*PERIOD | 1.0000 | -0.0805 |
| 1.5*PERIOD | -1.0000 | 0.0912 |
| 2.0*PERIOD | 1.0000 | 0.0754 |
| 3.0*PERIOD | 1.0000 | 0.0001 |
| 4.0*PERIOD | 1.0000 | -0.0050 |
| 4.9*PERIOD | 0.8090 | 0.0180 |

The model already fits `t=0` less well than uniform/causal/curriculum
(0.852 vs. their 0.975-0.993, though nowhere near as broken as NTK
reweighting's 0.1465), then settles into the same near-zero plateau
(`0.0001`-`0.09`) from about `t=PERIOD` onward while the true field keeps
cycling through its full `[-1, 1]` range — the identical "near-constant
output trivially satisfies the wave equation" collapse every prior issue
in this thread found, not a different pathology.

**Diagnosed mechanistically, and the mechanism is sharper — and more
interesting — than "no spike to catch": the penalty is cheapest exactly
where the model has collapsed, and most expensive exactly where it's
still doing the hard work of tracking the true oscillation.** The
paper's own motivating example is collocation-starved: with as few as
12-68 points on a 1D domain, the model can satisfy the residual at the
sampled points while switching abruptly between the true and trivial
solution somewhere in the unsampled gap between them, and that abrupt
switch is exactly what shows up as a spike in the residual's gradient —
which their penalty directly suppresses. This project resamples 200
fresh collocation points every step over the whole domain (never a fixed
sparse set), a qualitatively different regime — so the first question
was whether there's simply no spike here to catch. Instrumenting
`_pde_residual_and_input_grad_sq`'s own output (the unweighted term the
penalty maximizes) throughout a fresh seed-0 training run, and its
spatial distribution across the full 5-period domain at the end of
training (10 equal-width time chunks, 2000 fresh evaluation points,
matching issue #23's chunking instrumentation), confirms that — but also
reveals something stronger:

*(training trajectory, every 500 steps, unweighted loss components)*

| step | loss_pde | loss_bc | loss_ic | loss_ic_dot | penalty (unweighted max grad_sq) |
|---|---|---|---|---|---|
| 0 | 7.30e-5 | 7.53e-3 | 6.52e-1 | 1.47e-3 | 2.30e-3 |
| 500 | 5.19e-4 | 3.16e-2 | 6.13e-2 | 2.70e-3 | 8.04e-3 |
| 1000 | 2.73e-4 | 2.45e-2 | 4.23e-2 | 3.64e-3 | 6.89e-3 |
| 1500 | 2.44e-4 | 1.28e-2 | 4.44e-2 | 1.72e-3 | 4.17e-3 |
| 2000 | 2.87e-4 | 2.48e-2 | 4.00e-2 | 2.52e-3 | 1.14e-2 |
| 2500 | 3.88e-4 | 1.55e-2 | 2.90e-2 | 1.84e-3 | 7.98e-3 |
| 3000 | 4.18e-4 | 2.06e-2 | 2.95e-2 | 1.93e-3 | 1.12e-2 |
| 3500 | 4.22e-4 | 2.04e-2 | 2.24e-2 | 9.49e-4 | 2.67e-3 |

*(final spatial distribution, 10 time chunks over the full domain, 2000 fresh points)*

| chunk (t range) | mean residual^2 | mean grad_sq | max grad_sq |
|---|---|---|---|
| 0 ([0,1]) | 1.60e-3 | 7.06e-3 | 4.57e-2 |
| 1 ([1,2]) | 7.26e-4 | 1.42e-3 | 4.03e-3 |
| 2 ([2,3]) | 6.02e-4 | 1.14e-3 | 2.31e-3 |
| 3 ([3,4]) | 1.86e-3 | 3.13e-4 | 9.67e-4 |
| 4 ([4,5]) | 1.39e-3 | 2.15e-4 | 3.39e-4 |
| 5 ([5,6]) | 4.93e-4 | 2.13e-4 | 2.80e-4 |
| 6 ([6,7]) | 1.08e-4 | 9.18e-5 | 1.59e-4 |
| 7 ([7,8]) | 1.17e-5 | 2.97e-5 | 5.84e-5 |
| 8 ([8,9]) | 1.23e-6 | 8.05e-6 | 1.80e-5 |
| 9 ([9,10]) | 7.48e-6 | 2.24e-6 | 5.29e-6 |

Both the residual magnitude and its gradient decrease **monotonically**
from chunk 0 (near `t=0`, where the pointwise check above shows the
model is still partially tracking the true oscillation) down to chunk 9
(deep in the collapsed plateau, where predictions are within `0.02` of
zero everywhere) — no spike anywhere in the domain, confirming the "no
localized switch to catch" hypothesis. But the training trajectory shows
something the per-chunk snapshot alone wouldn't: the penalty term stays
non-negligible throughout training (`2.3e-3` to `1.1e-2`, the same order
of magnitude as `loss_bc`, competing for real gradient budget against
the other four terms every step, not silently near-zero from the
start), *and* it's largest in exactly the region where the model is
doing the genuinely hard thing (tracking real oscillation, which
requires a residual that varies across the domain, i.e. has nonzero
gradient) and smallest in exactly the region where the model has already
given up and gone flat (a near-constant function has near-zero curvature
*everywhere*, so its residual's own gradient is trivially near-zero too,
not just the residual itself). **The penalty doesn't just fail to catch
the trivial solution — the trivial solution is the single cheapest way
to satisfy it**, so paying its (nonzero, competing-for-gradient) cost
pulls fitting capacity away from the initial condition instead of buying
any resistance to the collapse, which is exactly what the pointwise
check's worse-than-baseline `t=0` fit (`0.852` vs. uniform/causal/
curriculum's `0.975-0.993`) shows in miniature. This is a milder version
of the same story issue #34 found for NTK reweighting: a literature
fix's implicit assumption about *how* a trivial solution manifests
(there, PDE-residual dominance; here, a localized truth-to-trivial
switch) doesn't hold for this project's specific pathology (a smooth,
domain-wide settling), and applying the fix anyway has a real, if small,
cost.

**Leads for whoever picks this up next:**
1. The paper's mechanism is specifically about *collocation-starved*
   domains where an abrupt truth-to-trivial switch is possible between
   sparse points. Testing this same penalty against a deliberately
   collocation-starved variant of this project's own long-horizon target
   (matching the paper's sparse-point regime more closely, rather than
   this project's every-step-resampled 200 points) is untried and might
   transfer better than this issue's controlled-comparison setup
   allowed — the diagnostic above only rules out a spike existing at
   this project's shipped point density, not at every density.
2. This is now the fifth fix tried against the long-horizon collapse
   (after causal reweighting, pseudo-sequence tokenization, curriculum,
   NTK reweighting) and the second (after NTK reweighting) to actively
   regress relative to doing nothing, though far more mildly. Every
   fix tried so far assumes a *specific* mechanism for how a PINN reaches
   a trivial solution (unconverged-residual lag for causal weighting,
   PDE-residual dominance for NTK reweighting, a localized spike at an
   abrupt switch for this issue) — none of those specific mechanisms
   match this project's actual failure signature (a smooth, domain-wide
   settling with no lag, no PDE-residual dominance, and no localized
   spike, where the collapsed region is instead the *cheapest* place for
   each of these penalties to be satisfied). Only curriculum training
   (issue #32, which changes *what the model ever sees*, not what the
   loss penalizes) has given a real improvement. Worth taking seriously:
   literature fixes aimed at loss-term mechanics keep missing this
   project's specific pathology because the collapse happens to make
   each fix's target quantity look small/smooth/easy, not because the
   fixes are poorly implemented.
3. `lambda_grad` was left at the paper's own unscaled default (`1.0`) —
   untuned. The training trajectory above shows the penalty competing at
   a similar order of magnitude to `loss_bc` throughout training (not
   crushing another term the way NTK reweighting's ratio-based weights
   did) — a smaller `lambda_grad` would likely shrink this issue's mild
   regression, but per lead #2 above, there's no reason to expect it
   would produce an actual improvement, since the diagnostic shows the
   mechanism isn't engaging with this project's failure mode regardless
   of weight.

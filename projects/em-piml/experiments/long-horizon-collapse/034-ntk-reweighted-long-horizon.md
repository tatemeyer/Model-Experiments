# Does NTK-based adaptive loss-term reweighting fix the long-horizon causality collapse? (issue #34)

Issues #20, #25, and #30 each independently flagged the same unexplored
lead: NTK-based adaptive loss-term reweighting (Wang, Teng, Perdikaris,
"Understanding and Mitigating Gradient Flow Pathologies in
Physics-Informed Neural Networks," SIAM J. Sci. Comput. 2021,
arXiv:2001.04536) — motivated by NTK eigenvalue theory, this is a
different mechanism than every fix already tried against the
long-horizon collapse: it doesn't reweight the PDE-residual term across
time chunks (issue #23's causal reweighting), change the input
representation (issue #30's pseudo-sequence tokenization), or change
the training schedule (issue #32's curriculum) — it rebalances the
*global* weight of each loss term (PDE residual, BC, IC, `dE/dt(x,0)=0`)
based on the relative magnitude of that term's own backpropagated
gradient.

`_train_pinn_adam_ntk_reweighted`/`train_cavity_ntk_reweighted_long_horizon`
(`src/em_piml/train.py`) implement the paper's Algorithm 1 ("learning
rate annealing"): every `update_every=10` steps, each non-PDE term
`i`'s weight is nudged via an exponential moving average
(`alpha=0.9`) toward `hat_lambda_i = ||grad(loss_pde)|| /
||grad(loss_i)||` — i.e. the PDE-residual term's own gradient norm
divided by term `i`'s gradient norm, computed via `torch.autograd.grad`
on each loss component separately (`_pinn_loss` was split into
`_pinn_loss_components` to expose the four terms individually without
changing any existing call site's behavior). `loss_pde`'s own weight
stays fixed at 1.0, the reference every other term is balanced against,
per the paper. Same architecture, step budget, point counts, and `lr`
as `train_cavity_long_horizon` (issue #23) — loss weighting is the only
variable.

**Result: not just a failure to help — measurably worse than doing
nothing.**

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| uniform (issue #23) | 0.9225, 0.9255, 0.9229, 0.9249 |
| causal, epsilon=1.0 (issue #23) | 0.9251, 0.9230, 0.9236, 0.9249 |
| pseudo-sequence (issue #30) | 0.9973, 1.0844, 0.9792, 1.1015 |
| curriculum (issue #32) | 0.9141, 0.9120, 0.8985, 0.9111 |
| NTK-reweighted (this issue) | 1.0089, 1.1336, 0.9999, 1.0275 |

NTK-reweighted training lands at 0.9999-1.1336 — worse than uniform's
0.9225-0.9255 on every seed, and no better than pseudo-sequence
tokenization's already-bad 0.9792-1.1015. This is the first fix in the
long-horizon thread that actively *regresses* relative to doing
nothing, not merely a fix that doesn't transfer.

**Diagnosed mechanistically, not just reported as a bare null result —
inspecting the adaptive weights during training explains why.**
Instrumenting `g_pde`/`g_bc`/`g_ic`/`g_ic_dot` and the resulting
weights every 500 steps of a fresh seed-0 run:

| step | g_pde | g_bc | g_ic | g_ic_dot | w_bc | w_ic | w_ic_dot |
|---|---|---|---|---|---|---|---|
| 0 | 2.6e-3 | 0.531 | 2.553 | 0.084 | 0.105 | 0.101 | 0.128 |
| 500 | 1.4e-3 | 0.462 | 2.190 | 0.120 | 0.013 | 0.011 | 0.023 |
| 1000 | 2.0e-3 | 0.993 | 1.907 | 0.122 | 0.003 | 0.002 | 0.017 |
| 2000 | 3.8e-3 | 1.521 | 1.549 | 0.092 | 0.003 | 0.002 | 0.040 |
| 3500 | 5.1e-3 | 1.466 | 1.200 | 1.6e-3 | 0.003 | 0.004 | 2.935 |

The PDE-residual loss's gradient norm (`g_pde`) stays tiny (~1e-3 to
5e-3) throughout training, while BC/IC gradient norms stay one to
three orders of magnitude larger (~0.5-2.5) — the **opposite** of the
regime this reweighting scheme was designed for. Wang et al.'s scheme
assumes the PDE-residual term is typically the *hard-to-satisfy* one
that dominates and overwhelms the BC/IC terms' gradients, so it
down-weights the PDE term (or equivalently up-weights BC/IC) to
compensate. But this project's specific failure mode inverts that: the
degenerate near-constant collapse makes the PDE residual trivially
*easy* to satisfy from early in training (near-zero second derivatives
in both `x` and `t`), so its gradient is already small — and the
formula `weight_i = g_pde / g_i` responds to a small `g_pde` by
crushing `weight_bc`/`weight_ic` down to ~0.002-0.013 (vs. uniform
weighting's implicit 1.0), instead of upweighting a neglected term.
This removes almost all of the pull toward matching the true initial
condition — exactly the constraint that was fighting the collapse in
the first place — which is consistent with the pointwise check below
showing this model fits the initial condition *worse* than every prior
approach, not just failing to sustain the oscillation over time.
(`w_ic_dot` additionally spikes to `2.935` at step 3500 when
`g_ic_dot` itself collapses toward zero — a `dE/dt(x,0)=0` term that's
easy to satisfy exactly, per this project's baseline discussion in
`../../CLAUDE.md` — showing the EMA-smoothed ratio is generally
unstable whenever any individual term's own gradient shrinks, not just
for the PDE term.)

**A pointwise check (seed 0) confirms the model no longer even fits the
initial condition well, unlike every prior approach in this thread:**

| t | true field | predicted |
|---|---|---|
| 0 | 1.0000 | 0.1465 |
| 0.25*PERIOD | -0.0000 | 0.1444 |
| 0.5*PERIOD | -1.0000 | 0.1389 |
| 1.0*PERIOD | 1.0000 | 0.1197 |
| 1.5*PERIOD | -1.0000 | 0.0960 |
| 2.0*PERIOD | 1.0000 | 0.0744 |
| 3.0*PERIOD | 1.0000 | 0.0458 |
| 4.0*PERIOD | 1.0000 | 0.0314 |
| 4.9*PERIOD | 0.8090 | 0.0246 |

Every prior approach in this thread (uniform, causal, pseudo-sequence,
curriculum) tracked the true field closely at `t=0` (0.975-0.993 vs.
true 1.0) before collapsing later. This model never gets `t=0` right in
the first place (`0.1465` vs. true `1.0`) — consistent with the IC
weight being crushed to ~0.002-0.013 throughout training, exactly as
diagnosed above.

`tests/test_ntk_reweighted_long_horizon.py` locks in the finding as a
regression check (not an accuracy bar): seed 0, default config, asserts
relative L2 error `> 0.5` — comfortable margin below the observed
0.9999-1.1336 range.

**This is now the fourth fix tried against the long-horizon collapse
(after causal reweighting, pseudo-sequence tokenization, curriculum
training) and the first one that actively makes things worse rather
than merely failing to help.** Consistent with this project's standing
pattern (see `030-pseudo-sequence-long-horizon.md`'s leads): a
literature fix's assumptions about *which* failure mode it addresses
matter — this scheme assumes PDE-residual dominance, which is the
opposite of this project's degenerate-collapse pathology, so applying
it uncritically actively hurts rather than merely not helping.

**Leads for whoever picks this up next:**
1. The paper's formula could be inverted or symmetrized for this
   specific failure mode (e.g. weight terms by `mean(all gradients) /
   g_i` rather than always normalizing against `g_pde` specifically),
   since this project's problem structurally has the PDE term as the
   *easy* one, not the hard one the paper assumes — untried here, and
   would be a deliberate departure from the paper's literal formulation
   after confirming (as done here) that a faithful implementation
   doesn't transfer.
2. `update_every=10` and `alpha=0.9` were taken directly from the
   paper's own suggested defaults, untuned for this project's specific
   loss landscape — the instability at `w_ic_dot` (spiking to `2.935`)
   suggests either less-frequent updates or a lower `alpha` (slower
   EMA) might reduce the swings, though it's unclear that would change
   the qualitative direction (crushing IC weight) that's actually
   driving the regression.
3. Combining an anti-trivial-solution mechanism (queued as issue #35)
   with a *symmetrized* version of NTK reweighting (lead #1 above)
   might avoid this issue's specific failure — the IC weight only got
   crushed because the PDE term looked artificially "easy"; a mechanism
   that makes the trivial solution itself harder to reach could change
   what the gradient-norm ratios look like entirely, not just how
   they're used.
4. This project now has four fixes tried against the long-horizon
   collapse (causal, pseudo-sequence, curriculum, NTK reweighting)
   targeting loss-weighting, architecture, and training schedule
   respectively, with curriculum training (`032-curriculum-long-horizon.md`)
   the only one giving a real (if modest) improvement. Issues #35-#41
   queue up several more untried directions (anti-trivial-solution
   regularization, Neuro-Spectral Architectures, R3 sampling, and
   others) — see `../../CLAUDE.md`'s open-leads section for what's next.

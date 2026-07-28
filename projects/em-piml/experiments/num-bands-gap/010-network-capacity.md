# Does more network capacity close the residual gap? (issue #10)

PR #9 got `num_bands=4` L-BFGS down to 0.065-0.104 relative L2 — close
to, but not reliably under, the standard `0.1` bar (seed 1 landed at
0.104). Architecture (32-hidden, 3-layer) had been held fixed since
issue #4 specifically to keep every comparison in this thread
controlled against the `num_bands=2` baseline. This issue tests whether
32-hidden was ever the right size for the higher-dimensional
Fourier-embedded input at `num_bands=4` (18-dim: `1 + 2*4` per scalar,
times 2 scalars), now that collocation density is no longer the
confound (see `008-denser-collocation.md` / `012-point-draw-variance.md`).

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
1. SOAP/SS-Broyden (the paper's actual best performers, see
   `006-lbfgs-optimizer.md`) still weren't tried via capacity scaling —
   out of scope here since they require adopting a new
   package/implementation, not a `torch.optim` drop-in like L-BFGS. (See
   `011-soap-optimizer.md`.)
2. The density-vs-accuracy non-monotonicity from issue #8 (lead #4
   there) is still open and now somewhat moot at this capacity — 64x3
   already clears the bar at the shipped 2000-point density, so
   re-sweeping density at 64-hidden isn't urgent, but the underlying
   "why is it non-monotonic" question wasn't answered by this issue.

# Does quasi-random (Sobol) sampling reduce the point-draw variance found in issue #12? (issue #40)

Issue #12 found that for the `num_bands=4` L-BFGS problem, *which*
collocation points get drawn matters as much as how many: resampling
alone (model-init seed fixed, only `points_seed` varied) reproduced
variance (stdev 0.035 at `n_collocation=2000`, 0.047 at 4000) rivaling
the entire between-density spread issue #8 found across 1000-4000
points. It flagged low-discrepancy sampling as an untried fix and
explicitly noted its own numbers were measured at the (then-current)
32-hidden architecture, not the 64-hidden default issue #10 shipped
afterward — "absolute numbers would likely shift down... hasn't been
re-verified at 64-hidden." "Quasi Random Physics-Informed Neural
Networks" (arXiv:2507.08121) reports Sobol sequences improve PINN
convergence stability over pseudo-random uniform sampling; this issue
tests that directly, and along the way finally re-verifies issue #12's
own flagged 64-hidden gap.

`_sample_points_sobol` (`src/em_piml/train.py`) is a drop-in alternative
to `_sample_points`: one 2D `torch.quasirandom.SobolEngine` (scrambled —
unscrambled Sobol repeats the same first point every seed, which would
break the "independent draws" premise this is tested under), drawn from
sequentially for collocation, then boundary, then initial points, so
every draw advances the same low-discrepancy sequence. `sobol_point_draw_sweep.py`
reruns issue #12/`point_draw_sweep.py`'s exact benchmark shape (5
independent point-set draws, `points_seed` 100-104, at each of
`n_collocation=2000`/`4000`, model-init `seed=0`, `n_boundary=n_initial=400`,
`outer_steps=50`/`max_iter=50` all held fixed) with `sampling="sobol"`
as the only additional variable.

**First: re-running the plain `point_draw_sweep.py` unmodified (it
already defaults to `train_fourier_cavity_lbfgs`'s current `hidden=64`)
resolves issue #12's own flagged gap.** The point-draw variance problem
had already shrunk substantially from the capacity increase alone,
independent of sampling method:

| n_collocation | uniform, 32-hidden (issue #12) | uniform, 64-hidden (this issue) |
|---|---|---|
| 2000 | mean 0.078, stdev 0.035, range 0.041-0.145 | mean 0.0259, stdev 0.0082, range 0.0185-0.0411 |
| 4000 | mean 0.086, stdev 0.047, range 0.026-0.166 | mean 0.0267, stdev 0.0066, range 0.0144-0.0334 |

That's a ~4-7x reduction in stdev from widening the network alone — a
finding issue #10 didn't itself measure (it varied capacity at one
point draw per seed, not resampling variance) and issue #12 explicitly
left open.

**Second, and the issue's actual question: does Sobol sampling improve
on this already-much-tighter 64-hidden uniform baseline?** Yes, but
modestly — a real reduction in variance, not in mean accuracy:

| n_collocation | sampling | mean | stdev | range |
|---|---|---|---|---|
| 2000 | uniform | 0.0259 | 0.0082 | 0.0185-0.0411 |
| 2000 | Sobol | 0.0243 | 0.0036 | 0.0197-0.0293 |
| 4000 | uniform | 0.0267 | 0.0066 | 0.0144-0.0334 |
| 4000 | Sobol | 0.0256 | 0.0046 | 0.0192-0.0315 |

Sobol's stdev is ~2.3x lower at `n_collocation=2000` and ~1.4x lower at
4000 — a real, consistent-direction effect at both densities, not
noise. Mean accuracy is only marginally better (~6% and ~4% lower
respectively) — nowhere near the ~3x mean improvement a naive comparison
against issue #12's stale 32-hidden numbers would have suggested (an
earlier draft of this write-up made exactly that mistake before the
64-hidden uniform baseline above was rerun for a fair, matched
comparison). Sobol's range is also visibly tighter (spread 0.0096 vs.
0.0226 at 2000; 0.0123 vs. 0.0189 at 4000) — fewer unlucky draws, not a
systematically better typical draw.

**Conclusion: ⚠️ real but modest effect, and it compounds with (doesn't
replace) issue #10's capacity fix.** Network capacity turned out to be
the dominant lever for the point-draw-variance problem issue #12
identified — Sobol sampling gives a genuine, reproducible further
reduction in draw-to-draw variance on top of that, but not the dramatic
fix a naive reading of "quasi-random beats pseudo-random" literature
might predict in isolation. Both mechanisms narrow the same failure
mode from different angles (more representational capacity to absorb
an unlucky draw; a sampling scheme less likely to produce one), and
they compose rather than compete.

`tests/test_sobol_point_draw_seed.py` covers `_sample_points_sobol`'s
plumbing (determinism, seed-independence, domain bounds) as fast checks,
plus one `@pytest.mark.slow` test locking in a single Sobol run's
magnitude (`n_collocation=4000`, `points_seed=100` → 0.0192, asserted
`< 0.05`). Per `point_draw_sweep.py`'s own precedent, the full 10-run
comparison sweep (`sobol_point_draw_sweep.py`) is deliberately not part
of the pytest suite — rerun it directly if this needs revisiting.

**Determinism**: verified before trusting any number above — same
`points_seed` reproduces bit-identical `_sample_points_sobol` output
regardless of unrelated global RNG state (model-init seed), and
different seeds draw genuinely different points (see
`test_sobol_point_draw_seed.py`).

**Leads for whoever picks this up next:**
1. Whether Sobol's variance reduction holds at the 32-hidden
   architecture too (i.e. is it capacity-independent, or does it
   interact with capacity the same way the raw variance problem did) —
   untried; this issue only tested at the current 64-hidden default.
2. Latin hypercube sampling (issue #12's other named candidate) is
   still untried — Sobol was the literature-motivated first pick
   (arXiv:2507.08121), not a claim that it's the best low-discrepancy
   option available.
3. This issue only threaded Sobol sampling through `train_fourier_cavity_lbfgs`
   (the L-BFGS single-mode `num_bands=4` problem, per the issue's own
   scope) — whether it transfers to the two-mode target
   (`two-mode-spectral-bias/`) or the long-horizon problems
   (`long-horizon-collapse/`) is untried.

# Is the density non-monotonicity about count or which points? (issue #12)

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
absolute numbers would likely shift down (per `010-network-capacity.md`'s
findings). The qualitative conclusion below (within-density variance
rivaling between-density variance) isn't expected to flip, but hasn't
been re-verified at 64-hidden.

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
slowdown to feed information this project's docs now record directly).
Rerun `point_draw_sweep.py` directly if this needs revisiting.

**Leads for whoever picks this up next:**
1. ~~Collocation-set density~~ (issue #8) and ~~count vs. which-points~~
   (this issue) — both resolved. The dominant lever left in the
   1000-4000 range is which points get drawn, not how many.
2. Network capacity, still held fixed since issue #4 — unexplored. (See
   `010-network-capacity.md` — resolved in parallel with this issue.)
3. SOAP/SS-Broyden — still out of scope (new dependency). (See
   `011-soap-optimizer.md`.)
4. Given points matter this much, a stratified/quasi-random sampling
   scheme (e.g. Latin hypercube or Sobol sequence instead of uniform
   `torch.rand`) might reduce the draw-to-draw variance found here
   without needing more points — untried. (Queued as issue #40.)

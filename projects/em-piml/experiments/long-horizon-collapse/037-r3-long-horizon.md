# Does R3 (Retain-Resample-Release) adaptive sampling fix the long-horizon causality collapse? (issue #37)

Six fixes have now been tried against the long-horizon collapse (issue
#23): causal loss-reweighting (`023-long-horizon-causal.md`),
pseudo-sequence tokenization (`030-pseudo-sequence-long-horizon.md`), a
short-to-long curriculum (`032-curriculum-long-horizon.md`), NTK-based
adaptive loss reweighting (`034-ntk-reweighted-long-horizon.md`), an
anti-trivial-solution regularizer (`035-antitrivial-regularizer.md`), and
Neuro-Spectral Architecture (`036-neusa-long-horizon.md`, which
essentially solved it). None of the five that stayed on the plain
coordinate-input MLP attacked the collocation-*sampling* strategy itself
— every one resampled collocation points uniformly at random every step,
identical to the baseline. Daw, Bu, Wang, Perdikaris, Karpatne,
"Mitigating Propagation Failures in Physics-Informed Neural Networks
using Retain-Resample-Release (R3) Sampling" (ICML 2023,
[arXiv:2207.02338](https://arxiv.org/abs/2207.02338)) frames their target
failure mode as a "propagation failure" leading to a "trivial solution" —
language that matches this project's own diagnosed mechanism (issue #23:
a near-constant output trivially satisfies the wave-equation residual)
closely — and fixes it purely by changing which collocation points get
trained on, not the loss, architecture, or schedule. This issue tests R3
against the long-horizon target as a genuinely different (sampling-
strategy) mechanism, still on the same plain MLP as #23/#30/#32/#34/#35.

**Algorithm (`_r3_update_pool`/`_train_pinn_adam_r3`,
`src/em_piml/train.py`).** R3 maintains a fixed-size pool of `n_collocation`
points across training steps instead of redrawing it fresh every step.
Each step: compute the PDE residual on the current pool, retain every
point whose `|residual|` exceeds the pool's own mean `|residual|` (the
paper's threshold `tau_i`, recomputed from the current population every
iteration — not a fixed constant), release the rest, and resample fresh
uniform-random points to refill the pool back to `n_collocation`. Using a
strict `>` against the pool's own mean guarantees the resampled set is
never empty (the paper's "Non-Empty Theorem," Theorem 4.2): not every
point in a finite population can exceed its own mean. This reuses the
residual already computed for that step's gradient — no extra forward
pass, matching the paper's own claim of negligible overhead over uniform
resampling. Only the *collocation* points get this treatment; BC/IC
points stay freshly uniform-random every step, unchanged from every other
long-horizon variant in this thread, per issue #37's "sampling strategy is
the only variable" constraint. `train_cavity_r3_long_horizon` otherwise
reuses `train_cavity_long_horizon`'s exact architecture
(`CavityPINN(hidden=32, num_layers=3)`), step budget (4000), point counts
(`n_collocation=200, n_boundary=64, n_initial=64`), and `lr=3e-3` — R3 vs.
uniform resampling is the only variable. This issue implements base R3,
not the paper's Causal R3 extension (a `tanh`-gated causally-biased
variant) — Causal R3 would reintroduce a causal-weighting mechanism
issue #23 already tested and found didn't transfer here, confounding the
comparison; base R3 isolates the sampling-strategy question issue #37
actually asks.

**Result: does not fix the collapse — a small, consistent regression,
similar in shape to issue #35's anti-trivial regularizer but milder.**

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| uniform (issue #23) | 0.9225, 0.9255, 0.9229, 0.9249 |
| causal, epsilon=1.0 (issue #23) | 0.9251, 0.9230, 0.9236, 0.9249 |
| pseudo-sequence (issue #30) | 0.9973, 1.0844, 0.9792, 1.1015 |
| curriculum (issue #32) | 0.9141, 0.9120, 0.8985, 0.9111 |
| NTK-reweighted (issue #34) | 1.0089, 1.1336, 0.9999, 1.0275 |
| anti-trivial regularizer (issue #35) | 0.9687, 0.9697, 0.9716, 0.9680 |
| NeuSA (issue #36) | 0.002300, 0.002245, 0.002289, 0.002259 |
| R3 (this issue) | 0.9343, 0.9305, 0.9340, 0.9303 |

R3 lands at 0.9303-0.9343 — every seed here is worse than every seed of
uniform (0.9225-0.9255) or causal (0.9230-0.9251), a small but consistent
regression (R3's *best* seed, 0.9303, is still worse than uniform's
*worst* seed, 0.9255). The four seeds are tightly clustered (spread
0.0040), similar to antitrivial's tight clustering, and the regression is
milder than antitrivial's (0.968-0.972) and far milder than NTK
reweighting's (0.9999-1.1336) or pseudo-sequence's (0.9792-1.1015).
Determinism re-verified before trusting these numbers: seed 0 trained
twice independently (reduced `steps=50` for a fast check) produced
bit-identical parameters and identical relative L2 error.

**Pointwise check (seed 0) shows the identical collapse mechanism as
every prior variant in this thread — R3 doesn't change the qualitative
failure, just its magnitude slightly.** Evaluating at `x=0.5` across
increasing `t`:

| t | true field | predicted |
|---|---|---|
| 0 | 1.0000 | 0.9718 |
| 0.5 (`0.25*PERIOD`) | -0.0000 | 0.0451 |
| 1.0 (`0.5*PERIOD`) | -1.0000 | -0.3855 |
| 2.0 (`1.0*PERIOD`) | 1.0000 | 0.0595 |
| 3.0 (`1.5*PERIOD`) | -1.0000 | 0.0513 |
| 4.0 (`2.0*PERIOD`) | 1.0000 | -0.0034 |
| 6.0 (`3.0*PERIOD`) | 1.0000 | -0.0230 |
| 8.0 (`4.0*PERIOD`) | 1.0000 | -0.0101 |
| 9.8 (`4.9*PERIOD`) | 0.8090 | 0.0002 |

The model tracks the true field reasonably well near `t=0` (0.9718 vs.
1.0), partially through `t=1.0` (-0.3855 vs. -1.0, weaker than the
uniform baseline's typical `t=1.0` fit but not yet collapsed), then
settles into the same near-zero plateau (`-0.02` to `0.06`) from roughly
`t=2*PERIOD` onward while the true field keeps cycling through its full
`[-1, 1]` range — the identical "near-constant output trivially satisfies
the wave equation" collapse issue #23 diagnosed, not a different
pathology.

**Mechanistic diagnosis: R3 correctly identifies the highest-residual
region — but that region is *not* where the collapse actually lives, and
concentrating collocation budget there starves the collapsed region of
the (already-thin) sampling pressure uniform resampling gave it.**
Instrumenting a fresh seed-0 training run's retained-point set (binned
into 10 equal-width time chunks) at snapshots throughout training, and
the final per-chunk residual at the end of training (2000 fresh
held-out points):

*(retained-point counts per chunk, snapshots every 500 steps — chunk 0 is
`t in [0,1)`, chunk 9 is `t in [9,10)`; only chunks with any nonzero count
across all snapshots are shown, all others are 0 at every snapshot)*

| step | chunk 0 | chunk 1 | chunk 2 | chunk 3 | retained / 200 |
|---|---|---|---|---|---|
| 0 | 23 | 18 | 0 | 6 | 47 (24%) |
| 500 | 51 | 53 | 0 | 0 | 104 (52%) |
| 1000 | 35 | 7 | 2 | 13 | 57 (29%) |
| 1500 | 40 | 11 | 14 | 2 | 67 (34%) |
| 2000 | 48 | 19 | 0 | 0 | 67 (34%) |
| 2500 | 32 | 61 | 1 | 0 | 94 (47%) |
| 3000 | 27 | 48 | 0 | 0 | 75 (38%) |
| 3500 | 47 | 9 | 25 | 1 | 82 (41%) |

At every snapshot across the entire 4000-step run, R3 retains points
*exclusively* from chunks 0-3 (the first 40% of the domain) — never once
from chunks 4-9 (the back 60%). The final pool (after the last
retain-resample-release) is somewhat more spread (`[37, 33, 56, 14, 12,
8, 12, 9, 11, 8]` across the 10 chunks) but still 63% concentrated in
chunks 0-2 alone.

*(final per-chunk mean residual^2, 2000 fresh held-out points, matching
issue #23's chunking convention)*

| chunk (t range) | mean residual^2 |
|---|---|
| [0,1] | 2.11e-3 |
| [1,2] | 3.82e-4 |
| [2,3] | 2.79e-4 |
| [3,4] | 9.61e-5 |
| [4,5] | 2.47e-5 |
| [5,6] | 4.70e-5 |
| [6,7] | 3.28e-5 |
| [7,8] | 1.85e-5 |
| [8,9] | 9.32e-6 |
| [9,10] | 4.64e-6 |

The residual decreases **monotonically** from chunk 0 (highest, where the
pointwise check shows the model is still doing genuine work tracking the
true oscillation) to chunk 9 (lowest, deep in the collapsed plateau) — the
identical per-chunk residual signature issues #23/#34/#35 already found
for this project's collapse. R3's retain criterion is *working exactly as
designed*: it correctly finds and concentrates on the genuinely
highest-residual region. The problem is that this project's collapse
mechanism makes the collapsed region look like the *easiest*, not the
*hardest*, part of the domain — a near-constant function has near-zero
residual almost everywhere by construction, so R3's residual-based signal
has no way to distinguish "already correctly solved" from "cheaply
satisfied by giving up." R3 therefore spends an outsized share of its
collocation budget (24-52% of the 200-point pool, every snapshot)
re-refining the fit in chunk 0-1, which was already the *best*-fit region
under uniform sampling, while implicitly reducing how often chunks 4-9
get sampled at all relative to uniform's flat 1/10-per-chunk rate — the
opposite of what the collapsed region needs. This is the same
underlying story as issues #34's NTK reweighting and #35's anti-trivial
regularizer: a residual-based adaptive mechanism, faithfully implemented,
targets exactly the wrong region because this project's specific failure
signature (a *cheap*, not *expensive*, degenerate solution) inverts the
assumption every one of these literature fixes makes about where PINN
training difficulty concentrates.

`tests/test_r3_long_horizon.py` locks in the finding: `train_cavity_r3_long_horizon`
(seed 0, default `horizon_periods=5.0`) asserts relative L2 error `> 0.5`
— the same failure-signature bound issue #23's test uses, not an accuracy
bar (there's no bar to clear on this target yet).

**Leads for whoever picks this up next:**
1. R3's causal extension (the paper's "Causal R3": a `tanh`-gated
   time-window that biases sampling toward *earlier* unresolved time
   regardless of raw residual magnitude, per section 4.3 of the paper)
   was deliberately not implemented here (see "Algorithm" above) to keep
   this issue's comparison to base R3's sampling-strategy question
   uncontaminated by a second, already-separately-tested causal-weighting
   mechanism. Given issue #23 found causal weighting alone doesn't help
   here, Causal R3 is not an obviously promising follow-up, but it wasn't
   directly tested.
2. This is now the third residual-based adaptive mechanism (after NTK
   reweighting, issue #34, and the anti-trivial-solution regularizer,
   issue #35) to fail specifically because this project's collapse
   presents as a *low*-residual region, not a high-residual one, to
   *any* residual-based signal. A mechanism that instead flags
   "suspiciously smooth/near-constant output" directly (rather than
   inferring difficulty from residual magnitude) is the one class of fix
   in this thread's history not yet tried on that specific axis — issue
   #35's anti-trivial regularizer targeted residual-*gradient* spikes,
   still a residual-derived signal, not output smoothness/variance
   directly.
3. Whether R3's chunk-0-concentration behavior is itself informative —
   e.g. deliberately *excluding* the top-residual chunk from the retain
   criterion (or capping the retain fraction per chunk) to force
   redistribution toward the rest of the domain — is untried and would
   be a modification of R3's own algorithm, not a faithful
   implementation of the paper.
4. Network capacity was held fixed here (`CavityPINN(hidden=32,
   num_layers=3)`, matching every other long-horizon variant) — still
   untested for this specific sampling-strategy axis, consistent with
   the broader capacity gap this thread has flagged since issue #23.

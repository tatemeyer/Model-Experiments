# Does raising num_bands close the two-mode spectral-bias gap? (issue #25)

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
this project already has from `../num-bands-gap/` (L-BFGS or SOAP,
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
achieved on the single-mode target in `../num-bands-gap/010-network-capacity.md`
/ `../num-bands-gap/011-soap-optimizer.md`. Having the right frequency
available in the embedding is evidently necessary but not sufficient —
the network still can't learn to use it well on a target that actually
needs substantial weight on that frequency.

**And past `num_bands=4`, more embedding headroom actively destabilizes
training for *both* previously-robust optimizers** — `num_bands=6` and
`8` collapse to ~1.0-1.03 (worse than doing nothing) for L-BFGS, and
SOAP is even more extreme at `num_bands=8` (three of four seeds diverge
to relative L2 in the hundreds; the fourth stays near 1.0). This is a
different failure than the single-mode `num_bands=4/6` Adam instability
from issue #4, because L-BFGS/SOAP already *solved* that one
(`../num-bands-gap/010-network-capacity.md` /
`../num-bands-gap/011-soap-optimizer.md`) — yet they fail here at a
comparable or lower `num_bands`.

**Working interpretation:** the earlier `num_bands=4` instability (issues
#4/#6/#8/#10/#11) was a network with *unused* high-frequency capacity next
to a purely low-frequency target — L-BFGS/SOAP could converge to a solution
that simply keeps those extra directions near zero. Here, the target
genuinely contains high-frequency content (the `n=8` mode), so the
network must actually learn nontrivial weight on the high-frequency
Fourier directions, not just avoid them — a qualitatively harder
optimization problem, consistent with Khodakarami et al.'s
(`../num-bands-gap/006-lbfgs-optimizer.md`) NTK-eigenvalue-decay account
of why gradient-based learning of high-frequency content is intrinsically
slow, not just an optimizer implementation detail. Adding *more*
unnecessary high-frequency embedding dimensions on top of that
(`num_bands=6/8`) doesn't give the network more room to represent the
`n=8` mode (which only needs `8*pi`) — it just adds more directions for
the optimizer to mismanage, and this time there's no longer a "these
directions can safely sit near zero" escape hatch, because the loss
landscape near the true (partially high-frequency) solution is already
harder to navigate.

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
   two-mode target the way `../num-bands-gap/010-network-capacity.md`
   did for the single-mode `num_bands=4` case — untried here, and not
   obviously the fix (the single-mode case was solving an "unused
   capacity" instability, not the "needs to actually learn
   high-frequency content" problem this target poses), but cheap to test
   given the machinery already exists. (Queued as issue #41, PirateNets.)
2. NTK-based adaptive loss reweighting (see
   `../020-pseudo-sequence-tokenization.md`'s leads, and Khodakarami et
   al.'s own mitigation guidelines in `../num-bands-gap/
   006-lbfgs-optimizer.md`) remains the most literature-direct
   unexplored lever for a target that genuinely needs learned
   high-frequency content, as opposed to optimizer/capacity fixes aimed
   at unused-capacity instability. (Tried against the *long-horizon*
   collapse instead in `../long-horizon-collapse/
   034-ntk-reweighted-long-horizon.md` — never tried against this
   two-mode target specifically.)
3. Why does `num_bands=8` SOAP diverge so much harder and so much more
   seed-dependently (163-751 on 3 seeds, ~1.0 on the 4th) than
   `num_bands=8` L-BFGS (tight 1.02-1.03 across all 4 seeds)? Untouched
   here — could be about SOAP's Shampoo-style preconditioner behaving
   badly on an ill-conditioned Hessian at this embedding dimensionality,
   but this issue didn't dig into why.
4. ~~Issue #23 (causality) remains an independent, unblocked alternative
   failure mode to this whole `num_bands` thread.~~ Done, see
   `../long-horizon-collapse/`.

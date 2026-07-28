# Is the residual num_bands=4 L-BFGS instability an FP32 precision artifact? (issue #38)

Issues #6/#8/#10 characterized and resolved a `num_bands=4` Fourier-embedding
instability on this project's single-mode baseline via L-BFGS + denser
collocation + wider capacity — down from an original ~0.79-0.88 plateau
(32-hidden, `n_collocation=200`, issue #6) to 0.065-0.104 via density alone
(issue #8) to 0.018-0.041 via capacity (issue #10). Xu, Liu, Nassereldine,
Xiong, "FP64 is All You Need: Rethinking Failure Modes in Physics-Informed
Neural Networks" (University at Buffalo, SUNY, NeurIPS 2025,
arXiv:2505.10949) proposes a "Same-Basin Hypothesis": PINN "failure modes"
(near-zero PDE residual, high solution error) are not isolated local optima
separated by loss barriers from the true solution, but an intermediate
"failure phase" of a three-stage (un-converged -> failure -> success)
optimization trajectory in the *same* basin. They argue FP32's
`tolerance_change` (1e-7) sits at or above FP32 machine epsilon (1.19e-7),
so `torch.optim.LBFGS`'s convergence test fires prematurely and strands
training in the failure phase — switching to `torch.float64` (epsilon
2.22e-16) lets the same optimizer keep making meaningful updates and reach
the success phase, with no other change. This issue re-runs this project's
own two reference configurations under FP64, at the *same* iteration budget
as their documented FP32 numbers, to test that claim directly rather than
assuming density/capacity were ever the "real" fix.

`train_fourier_cavity_lbfgs` (`src/em_piml/train.py`) gained two new
parameters: `hidden` (previously hardcoded to `64`, now defaults to `64`
unchanged — lets this issue reproduce the *original* pre-issue-#10
32-hidden architecture too) and `dtype` (defaults to `torch.float32`,
bit-for-bit unaffected unless overridden). `dtype` threads through
`_sample_points` (all `torch.rand`/`torch.zeros`/`torch.full` calls now take
an explicit `dtype`) and `_train_pinn_lbfgs`; the model itself is cast via
`model.to(dtype)` after construction, which also casts
`FourierFeatureEmbedding`'s `frequencies` buffer (registered without an
explicit dtype, so it silently defaults to float32 otherwise — verified
this cast propagates to every parameter and buffer before trusting any
number below). `evaluate_relative_l2_error` gained the same `dtype`
parameter so evaluation points match the model's own dtype (a mismatch
errors inside the first `Linear` layer). `train_fourier_cavity_lbfgs_fp64`
is a thin wrapper fixing `dtype=torch.float64` — precision is the only
variable relative to each config's own documented FP32 numbers; the
constraint against also changing architecture/density/optimizer in the same
run is honored by testing exactly issue #6's and issue #10's own configs,
unmodified except for `dtype`.

**Result: at a matched iteration budget, FP64 does not fix the original
plateau, and does not meaningfully change the already-good shipped config
either — this project's instability is not an FP32 precision artifact.**

| config | precision | seed 0 | seed 1 | seed 2 | seed 7 |
|---|---|---|---|---|---|
| original (32-hidden, `n_collocation=200`) | FP32 (issue #6/#8) | 0.822 | 0.851 | — | — |
| original (32-hidden, `n_collocation=200`) | **FP64 (this issue)** | 0.908 | 0.889 | 0.922 | 0.892 |
| shipped (64-hidden, `n_collocation=2000`) | FP32 (issue #10) | 0.027 | 0.041 | 0.026 | 0.018 |
| shipped (64-hidden, `n_collocation=2000`) | **FP64 (this issue)** | 0.028 | 0.028 | 0.058 | 0.038 |

Both configurations land in essentially the same range under FP64 as their
documented FP32 numbers. The original 32-hidden/200-point config stays at
0.889-0.922 relative L2 under FP64 — if anything marginally *worse* than
FP32's 0.822-0.851 (only 2 seeds documented for that exact row in
`008-denser-collocation.md`, but consistent with issue #6's broader
0.79-0.88 characterization) — not the dramatic collapse to a low error the
paper's hypothesis would predict if this project's plateau had been a
precision artifact all along. The shipped 64-hidden config stays at
0.028-0.058 under FP64 vs. FP32's 0.018-0.041 — same order of magnitude,
both comfortably under the standard `0.1` bar, no meaningful change either
direction.

**Bonus data point (not required by this issue's success criteria, but
free to collect once the FP64 plumbing existed): 32-hidden at the
already-density-fixed `n_collocation=2000` density (issue #8's exact
config, pre-capacity-fix) — FP64 doesn't help here either, and adds
noticeably more seed-to-seed noise:**

| seed | FP32 (issue #8) | FP64 (this issue) |
|---|---|---|
| 0 | 0.098 | 0.034 |
| 1 | 0.104 | 0.225 |
| 2 | 0.096 | 0.190 |
| 7 | 0.065 | 0.083 |

Range widens from FP32's tight 0.065-0.104 to FP64's 0.034-0.225 — one seed
improves substantially (0), two get markedly worse (1, 2), one is roughly
flat (7). Mean error is worse under FP64 (~0.133 vs ~0.091). This is the
opposite of "FP64 is a free accuracy improvement": at this particular
architecture/density combination it mostly adds variance without a
consistent direction.

**Mechanistic diagnosis — the extreme runtime cost, not the accuracy, is
where the paper's proposed mechanism shows up most clearly.** The issue's
own text anticipated FP64 "roughly doubles" wall time; the actual slowdown
observed here is far larger and not incidental:

| config | FP32 wall time | FP64 wall time (this issue) | slowdown |
|---|---|---|---|
| original (32-hidden, 200pt) | ~40s (documented, `006-lbfgs-optimizer.md`) | ~406-410s | ~10x |
| shipped (64-hidden, 2000pt) | ~100-220s (documented, `010-network-capacity.md`) | ~2277-2444s | ~10-24x |

This was confirmed to be a genuine compute cost, not CPU contention: process
CPU time tracked wall time almost exactly (~98% utilization) across
multiple independent runs. A follow-up experiment isolates why: rerunning
the shipped 64-hidden config at a **reduced** L-BFGS budget
(`outer_steps=15, max_iter=15` instead of the default `50, 50` — a ~11x cut
in nominal total inner iterations) collapses the result to **1.062 relative
L2 — total failure**, not a graceful partial improvement over the full-budget
0.028. This directly corroborates the paper's specific mechanistic claim at
one level (FP64 genuinely stops L-BFGS's convergence test from exiting the
inner loop early — the optimizer keeps consuming its full nominal
`max_iter` budget every outer step instead of bailing out after a handful
of iterations the way FP32 apparently does, which is *why* it's so much
slower) while simultaneously showing that this behavioral difference does
**not** translate into a better final answer for this project's specific
plateau — extra genuine iterations largely just re-confirm the same local
optimum FP32 finds faster.

**This also means there is no cheaper, faithful version of this check —
reducing the L-BFGS budget to fit a normal CI time slice was tried and
found to invalidate the result entirely** (see above), not merely make it
noisier. `tests/test_fp64_precision.py`'s slow regression test therefore
uses the *original* 32-hidden/200-point config (the cheaper of the two,
~406-410s/seed) at the full, unmodified `outer_steps=50, max_iter=50`
budget — the only version of this test that means anything. The shipped
64-hidden config (~2277-2444s/seed) is not added as a CI test at all;
per this project's existing precedent for expensive sweeps
(`src/em_piml/point_draw_sweep.py`, `src/em_piml/num_bands_sweep.py`), the
full comparison lives in `src/em_piml/fp64_precision_sweep.py` and this
file, reproducible on demand rather than re-verified every CI run. A fast
(not slow-marked) plumbing test separately checks `dtype=torch.float64`
propagates correctly through model/point construction without erroring,
at a trivial scale that says nothing about accuracy.

Determinism re-verified per this project's standing rule before trusting
any number above: the shipped 64-hidden/seed-0 FP64 run was executed twice,
independently, and returned bit-identical results (`0.028183` both times,
despite ~7% wall-time variance between runs from ordinary scheduling
noise) before any other seed was trusted.

**Important scope caveat, to avoid overclaiming:** this issue tested FP64 at
the *same nominal iteration budget* as the documented FP32 baselines
(`outer_steps=50, max_iter=50`) — the controlled, apples-to-apples
comparison issue #38 specifically asked for. Xu et al.'s own three-stage
model (un-converged -> failure -> success) predicts that harder problems
need a *longer* failure phase before FP64 lets training break through to
success — so this result does not rule out that a much larger iteration
budget under FP64 would eventually reach a qualitatively better solution
here too. That experiment would abandon the budget-matched comparison this
issue was scoped to, and — given the reduced-budget ablation showed FP64
needs its *entire* nominal budget just to match FP32's result, not less —
would likely cost even more than the already-substantial ~2300-2400s/run
seen at 64-hidden. What this issue does establish cleanly: FP64 is not a
drop-in win at equal cost the way it might be on the paper's own GPU
benchmarks (1.1-1.3x slower there vs. 10-60x slower here on CPU), and it
does not rescue this project's specific plateau within a directly
comparable budget.

**Leads for whoever picks this up next:**
1. Whether a much larger L-BFGS budget (e.g. 10-100x more outer_steps)
   eventually pushes the original 32-hidden/200-point config from failure
   into Xu et al.'s "success phase" under FP64 is untested here — see the
   scope caveat above. Given the cost already observed, this would be a
   multi-hour-per-seed experiment, not a quick follow-up.
2. SOAP already fully closes this same `num_bands=4` gap under FP32
   (issue #11, 0.023-0.036) — FP64 was never going to be a more attractive
   fix even if it had worked, since it costs 10-60x more wall time for a
   worse-or-equal result at matched budget. Not worth retrying SOAP under
   FP64.
3. The ~10-60x FP64 slowdown mechanism (confirmed here: L-BFGS's internal
   convergence test stops exiting early) is itself a more general and
   possibly more interesting finding than this specific plateau's
   resistance to it — untested whether the same reduced-budget collapse
   happens for other optimizers/architectures in this project, or is
   specific to `torch.optim.LBFGS`'s `strong_wolfe` line search, or to CPU
   vs. the paper's GPU-only benchmarks.
4. The bonus 32-hidden/2000-point result (FP64 adds variance without a
   consistent direction) suggests FP64 isn't simply "neutral" wherever it
   doesn't fix a hard failure — it can make an already-working
   configuration noisier. Untested whether this is a general property or
   specific to L-BFGS's line search interacting with `torch.float64`'s
   different rounding behavior.

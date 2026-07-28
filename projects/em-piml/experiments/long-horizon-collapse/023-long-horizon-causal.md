# Does a longer time horizon break causality, and does causal loss-reweighting fix it? (issue #23)

An independent failure mode from the `num_bands` thread (see
`../two-mode-spectral-bias/`), queued alongside issue #22: does the plain
single-mode baseline (the original `analytical_field`, unchanged — no
two-mode superposition here) degrade if trained/evaluated over several
periods (`t` up to `5*PERIOD`) instead of one, and does the causal
loss-reweighting scheme from Wang, Sankaran, Perdikaris ("Respecting
Causality is All You Need for Training Physics-Informed Neural
Networks," arXiv:2203.07404) fix it?

`train_cavity_long_horizon` (`src/em_piml/train.py`) reuses
`train_cavity_baseline`'s exact shipped defaults (architecture, steps,
`lr`, point counts) — the only variable is `t_max` (via a new `t_max`
parameter threaded through `_sample_points`/`_train_pinn_adam`/
`evaluate_relative_l2_error`, defaulting to the original `PERIOD` so
every existing call site is bit-for-bit unaffected).

**Result: yes, it breaks badly — even a modest 2-period extension is
enough, and point density isn't the explanation.** At `horizon_periods=5`
(the issue's own suggested value): relative L2 error **0.9592-0.9633**
across seeds 0/1/2/7, vs. the single-period baseline's 0.026-0.046 — a
total failure, not a modest degradation. Before attributing this to
causality specifically, density was ruled out as a confound (this
project's own `../num-bands-gap/008-denser-collocation.md` precedent:
same `n_collocation=200` spread over a 5x larger domain is 5x sparser
per unit time) — scaling `n_collocation`/`n_boundary`/`n_initial`
proportionally with `horizon_periods` (200→1000 at 5 periods) made no
meaningful difference (0.9601 unscaled vs. 0.9550 density-matched). Even
a much milder 2-period extension already fails just as badly (0.9131
unscaled, 1.2508 density-matched-worse) — the failure isn't about
running out of points, it shows up almost immediately as the horizon
extends at all.

**Mechanism, confirmed pointwise and via per-chunk residual
instrumentation: the model tracks the true solution near `t=0`, then
collapses to a near-constant low-amplitude plateau instead of continuing
the oscillation** — not "learns the wrong frequency" or "gets noisier
with distance from `t=0`," but a qualitative collapse. At `x=0.5`, one
seed-0 2-period run: `t=0` predicted `0.975` (true `1.0`, good), `t=1.4`
predicted `-0.21` (true `-0.32`, still reasonable), then from `t=2.8`
onward the true field keeps swinging through its full `[-1, 1]` range
while the prediction flattens to a `0.05-0.09` plateau and stays there.
Instrumenting the per-time-chunk PDE residual (10 chunks across the
domain) during training showed *why* uniform weighting doesn't fix
this: residual for the later chunks is already trivially near-zero from
essentially the start of training (a near-constant/near-linear function
has near-zero second derivatives in both `x` and `t`, so it satisfies
the wave equation almost by construction, without needing to match the
true field) — the model finds this "free" low-residual collapse instead
of doing the harder work of propagating the true oscillation forward
from the initial condition.

**Causal loss-reweighting, implemented per the paper's own formulation,
does not fix this — and the reason is diagnosable, not just "it didn't
work."** `_causal_pinn_loss`/`_sample_points_causal`/
`_train_pinn_adam_causal` (`src/em_piml/train.py`) stratify collocation
points into `n_chunks=10` equal-width temporally-ordered bins each step
(`n_per_chunk=20`, so `n_per_chunk * n_chunks = 200` matches
`train_cavity_long_horizon`'s collocation budget exactly — chunking is
the only variable), compute a per-chunk mean-squared PDE residual, and
weight chunk `i`'s contribution to the loss by
`exp(-epsilon * sum(chunk_losses[:i]))` (paper eq. 11-13; only the PDE
residual term is causally weighted, BC/IC stay uniformly weighted, per
the paper's own scoping) — `train_cavity_causal_long_horizon` wraps this
with the same shipped defaults otherwise. Swept `epsilon` across
**0.1, 1, 5, 20, 100, 500, 5000** (four orders of magnitude) at both
2-period and 5-period horizons, seeds 0/1/2/7 at the 5-period/default
epsilon values:

| variant | relative L2 (seeds 0/1/2/7, horizon=5) |
|---|---|
| uniform | 0.9601, 0.9592, 0.9633, 0.9632 |
| causal, epsilon=1.0 | 0.9593, 0.9607, 0.9644, 0.9598 |
| causal, epsilon=100.0 | 0.9596, 0.9612, 0.9664, 0.9571 |
| causal, epsilon=500.0 | 0.9618, 0.9621, 0.9679, 0.9599 |

**Erratum (found and fixed during issue #32):** the table above was
evaluated over the wrong domain. This project's `PERIOD` is `2` (not
`2*pi` — `OMEGA = pi`, and `PERIOD = 2*pi/OMEGA = 2`), but the
evaluation `t_max` used to produce every number in this table was
computed as `5.0 * (2 * math.pi) ≈ 31.4` — i.e. ~15.7 periods, not the
intended 5. Re-evaluated at the correct `t_max = 5.0 * PERIOD = 10`:
**uniform 0.9225, 0.9255, 0.9229, 0.9249; causal (epsilon=1.0) 0.9251,
0.9230, 0.9236, 0.9249** across seeds 0/1/2/7. The qualitative
conclusion is unchanged (still total failure, still statistically
indistinguishable from uniform), and `tests/test_causal_long_horizon.py`
now uses the corrected `T_MAX`, but the `epsilon in {100, 500, 5000}`
rows above and the 2-period numbers in the paragraph below were **not**
re-verified at the correct domain — treat them as evaluated over ~15.7
periods, not 5, until re-checked. See `032-curriculum-long-horizon.md`
for the full comparison table (uniform/causal/pseudo-sequence/curriculum)
at the corrected domain, and for how this was discovered.

Statistically indistinguishable from uniform weighting at every
`epsilon` tried — causal weighting gives essentially zero improvement,
not even a partial one like `../two-mode-spectral-bias/
025-num-bands-sweep.md`'s `num_bands=4` result. (At the milder 2-period
horizon, `epsilon=500` gave a small improvement, 0.8917 vs. 0.9131
uniform — real, but nowhere near recovering accuracy, and gone again by
`epsilon=5000`, 0.9480.)

**Why the fix doesn't transfer, diagnosed from the same per-chunk
residual instrumentation used to characterize the failure above:** the
causal weight for chunk `i` is a function of *how large* the earlier
chunks' residual loss is — the whole mechanism is designed to detect
"later chunks are being solved out of order while earlier ones remain
unconverged" and suppress that. But this project's collapse isn't that
pathology: the later chunks' residual is already trivially *small*
(near-zero, via the degenerate near-constant collapse) from early in
training, not stubbornly large. Since the gating signal
(`cumulative earlier-chunk loss`) never grows large, `exp(-epsilon *
cumulative)` never meaningfully shrinks below ~1 for any tested
`epsilon`, so the causal weights stay close to uniform throughout
training regardless of `epsilon` — the gate never actually closes. This
is a clean, mechanistic explanation for a null result, not just "we
tried it and it didn't help": Wang et al.'s causal weighting targets
*unconverged-residual lag*; this project's specific long-horizon failure
is a *degenerate low-residual collapse*, a different pathology the same
formula's gating signal can't detect.

`tests/test_causal_long_horizon.py` locks in both findings as regression
checks (not accuracy bars — there's no bar to clear on this target yet):
both the uniform and causal variants (seed 0, default `epsilon=1.0`,
`horizon_periods=5.0`) assert relative L2 error `> 0.5` — comfortable
margin below the observed 0.923-0.926 range for both (corrected domain,
per the erratum above).

**Leads for whoever picks this up next:**
1. A causal-weighting variant that also gates on *whether the field
   value itself* (not just the residual) is converging — e.g. weighting
   by IC/BC-consistency-propagated error rather than pure PDE residual —
   might catch the degenerate-collapse pathology this issue's literal
   implementation of the paper's formula couldn't. Untried here; not
   obviously literature-grounded (would be a departure from the paper's
   own formulation, which this issue deliberately implemented faithfully
   first).
2. ~~An explicit curriculum — train on `t in [0, PERIOD]` to convergence
   first, then progressively extend the domain~~ — tried, see
   `032-curriculum-long-horizon.md`: real but modest improvement.
3. Network capacity/architecture was held fixed throughout (per this
   project's controlled-comparison convention) — whether a
   larger/deeper network is simply less prone to this kind of
   degenerate collapse (more capacity to represent genuine oscillation
   without it being the "cheap" solution) is untested.
4. This is now the second failure mode in this project (after
   `../two-mode-spectral-bias/025-num-bands-sweep.md`'s two-mode
   spectral-bias gap) where a literature-prescribed fix, faithfully
   implemented, didn't transfer — both times because this project's
   specific failure mechanism turned out to differ subtly from the
   mechanism the fix targets. Worth remembering as a pattern: confirming
   *which* failure mode is actually present (via instrumentation, as
   done here and there) before assuming a literature fix will apply is
   doing real work in this project, not a formality.

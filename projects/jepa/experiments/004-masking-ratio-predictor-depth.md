# Do masking ratio or predictor depth move the collapse boundary? (issue #107)

Arc 1's premise was that three levers prevent representation collapse: EMA
momentum, masking ratio/strategy, and predictor depth/capacity. Slice 2 (issue
#97) swept the first and found it near-irrelevant — a flat healthy plateau from
momentum 0.0 to 0.999, with **stop-gradient** rather than smoothing doing the
work. This slice sweeps the other two, jointly, and closes Arc 1.

Varying them jointly rather than one at a time was deliberate: the most
plausible remaining hypothesis was an *interaction* — a deeper predictor having
the capacity to absorb a more aggressive masking ratio — which a one-at-a-time
sweep would have hidden by construction.

No new paper is cited; `../LITERATURE.md`'s I-JEPA row is updated with the
verdict on its masking claim at this scale.

## Implementation

`train_jepa` gained `predictor_depth`, `num_target_blocks`, and
`target_scale_range` (`src/jepa/train.py`), defaulting to the values every prior
slice ran at. That the defaults are a **bit-for-bit no-op** was verified, not
assumed — running the unmodified pre-change code produced identical
`effective_rank` at seeds 0/1/2 (2.8376 / 2.3622 / 2.7932). Without that check
the new grid's `default`/`depth=2` cell could not be compared to Slices 1 and 2.

`src/jepa/masking_depth_sweep.py` runs depth ∈ {1, 2, 4} × masking ∈ {light,
default, heavy} × seeds {0, 1, 2} at 3000 steps, recording `effective_rank`,
`embedding_std`, final loss, and a loss slope.

**`probe_r2` is deliberately not collected.** Issue #104 showed it is saturated
on this task — an untrained encoder scores 0.9767 against a ~0.978 ceiling — so
it cannot separate variants and would only add a column of 0.977s.

**The masking knob is not the masking ratio.** The parameter is a per-block
scale range, and four blocks are sampled independently and unioned, so overlap
makes the realized masked fraction sublinear in the knob. Measured over 2000
draws:

| config | per-block scale | realized target fraction |
|---|---|---|
| light | (0.05, 0.10) | 0.258 |
| default | (0.15, 0.20) | 0.493 |
| heavy | (0.30, 0.40) | 0.702 |

So the sweep spans roughly a quarter to seven-tenths of the grid masked — a wide
span, not a marginal one.

## Result: neither axis moves `effective_rank` at all — the effect is smaller than seed noise

Cell means (3 seeds each):

| depth | light | default | heavy |
|---|---|---|---|
| 1 | 2.632 | 2.852 | 2.151 |
| 2 | 2.778 | 2.655 | 2.543 |
| 4 | 2.576 | 2.699 | 2.711 |

The table invites pattern-reading — depth 1 looks bad at heavy masking, depth 4
looks like it recovers — but the variance decomposition says none of it is real:

| quantity | value |
|---|---|
| between-cell spread of means | 0.701 |
| **mean within-cell (seed) spread** | **0.752** |
| max within-cell spread | 1.333 |
| MS_between / MS_within (**F**, df 8,18) | **0.674** |

**F is below 1**: the variation *between* the nine configurations is smaller
than the variation *within* them across three seeds. Marginal means say the same
thing — depth spans 2.545 → 2.662 (0.117) and masking spans 2.468 → 2.735
(0.267), against a within-cell standard deviation of ~0.4.

The apparent depth × masking interaction is exactly the shape noise takes at
n=3. It is recorded here as *not supported*, not as a lead.

## What did move: training stability, on the masking axis only

The masking ratio has a clean, monotone, nearly-unanimous effect — on the loss
trend rather than on the representation:

| masking | realized fraction | mean loss slope | runs with loss *rising* |
|---|---|---|---|
| light | 0.258 | **−0.021** | 2 / 9 |
| default | 0.493 | +0.008 | 6 / 9 |
| heavy | 0.702 | **+0.063** | **9 / 9** |

A positive slope means the loss was still *increasing* when training stopped. At
the heavy end that happened in every single run. So masking ratio is not an
inert knob — it makes the prediction task harder and training less stable — it
simply does not translate into the collapse metric Arc 1 cares about. That
distinction is the useful part: an axis can visibly affect optimization and
still be irrelevant to representation collapse.

## The result that dwarfs both axes: step count

Placing this grid against the reference points from earlier slices:

| configuration | `effective_rank` |
|---|---|
| no stop-gradient (`use_ema=False`), Slice 1 | 1.25 – 1.46 |
| **this entire 27-run grid, 3000 steps** | **1.94 – 3.44** |
| untrained random-init encoder, Slice 1 | 2.44 – 2.93 |
| Slice 2 momentum plateau, 6000 steps | 3.15 – 4.03 |

Two things fall out. First, at 3000 steps the trained models *straddle an
untrained encoder* — every cell in this grid overlaps the random-init band.
Second, Slice 2's plateau at 6000 steps clears it outright. **Doubling the step
count moves `effective_rank` further than any setting of either axis studied
here.** Arc 1 named three architectural levers and the dominant variable turned
out to be how long you train.

`tests/test_masking_depth_sweep.py` locks this in. "No effect" has no threshold,
so the null is asserted **comparatively** instead: sweeping either axis across
its full swept range at a fixed seed must move `effective_rank` by less than the
0.75 seed-noise scale. That is the slice's actual claim, and it is what a future
change making either axis matter would break. Stating it against a noise scale
rather than an absolute rank also makes it robust to the torch-version drift
noted below.

Two things are deliberately *not* asserted. The loss-slope effect is clean at
3000 steps (9/9 runs) but its margin at a test-sized budget is ~0.007 — measured,
and too thin to encode without a flaky test. And no test asserts an absolute
"healthy" rank floor: at the reduced budget every variant sits at 1.6–1.9,
uncomfortably near the no-stop-gradient collapse band (1.25–1.46), so such a
threshold would have little margin. The remaining tests guard the plumbing —
that the defaults are a no-op, that `predictor_depth` actually reaches
`Predictor`, and that the masking knob actually changes the realized fraction —
which is the failure mode that would otherwise make a real effect *look* like
this slice's null.

**A reproducibility caveat worth recording.** These runs do not reproduce the
`effective_rank` values recorded on 2026-08-03 exactly (seed 0: 2.8376 now vs
2.779 then; seeds 1/2 differ by +0.010 and +0.002). The unmodified pre-change
code reproduces the *new* numbers exactly, so the drift is the environment —
`torch 2.13.0+cu126` (issue #60) rather than the build those rows were recorded
under — not this slice's changes. Regression thresholds in this project need
margin for torch-version drift of order 0.06 on this metric; the seed spread
(~0.75) is an order of magnitude larger, so no conclusion here is affected.

**Leads for whoever picks this up next:**

1. **Step count is the untested axis that actually matters.** Slice 2 varied it
   only as a checkpoint ladder while sweeping momentum; nothing has studied it
   as the primary variable. Given it beat both of this slice's axes, "how does
   `effective_rank` evolve with training length, and where does it saturate" is
   the obvious next mechanism question — and it is cheap.
2. **n=3 is not enough for this metric.** Within-cell seed spread reached 1.333,
   comparable to the entire range across configurations. Any future sweep on
   `effective_rank` needs more seeds per cell or it will keep producing F < 1
   grids that cannot answer their own question. This is the single most useful
   methodological lesson from the slice.
3. Heavy masking's unanimous positive loss slope means those runs had not
   converged at 3000 steps. Whether heavy masking eventually *does* separate,
   given enough steps to converge, is not answered here and is entangled with
   lead 1.

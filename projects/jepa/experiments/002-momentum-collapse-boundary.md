# Where is the EMA-momentum collapse boundary, and how does it move with training length? (issue #97)

Slice 1 (issue #69, `001-baseline-collapse-avoidance.md`) established that EMA
prevents representation collapse at exactly one operating point:
`ema_momentum = 0.996`, 3000 steps. `effective_rank` separated the full model
(2.35–2.79) from a no-EMA ablation (1.25–1.46) with no seed overlap.

That is one point on a two-dimensional surface. This experiment maps it.

The two axes are not separable, which is why they are swept together. Slice 1
measured the full model's `effective_rank` climbing 1.62 → 1.97 → 2.78 → 3.61
at 300/1000/3000/6000 steps while the no-EMA ablation *declined* 1.61 → 1.47 →
1.39 → 1.34 — **at 300 steps the two are indistinguishable.** A momentum sweep
at a single fixed step count would measure that choice rather than momentum.
Slice 1's own lead #1 says exactly this.

Motivating literature: Assran et al., I-JEPA (arXiv:2301.08243) for the
stop-gradient/EMA target design; Chen & He, SimSiam (arXiv:2011.10566), which
showed that stop-gradient is the load-bearing ingredient and a momentum encoder
is *not* required; and Grill et al., BYOL (arXiv:2006.07733), whose own target
decay-rate ablation points the other way (see "Relation to prior results"
below).

## Implementation

`train_jepa()` (`src/jepa/train.py`) gained a **checkpoint-evaluation hook**
(`checkpoint_steps`, `on_checkpoint`), so one 6000-step run yields metrics at
four step counts instead of retraining per cell — a 4× saving.

**The hook exposed a wrong assumption worth recording.** This slice's design
asserted that training never consumes torch's global RNG (batches come from an
explicit `torch.Generator`, masks from a numpy one), and therefore that a
callback could not perturb training. That is false: `Predictor`'s
`nn.TransformerEncoderLayer` stack carries **six `Dropout` modules at the
PyTorch default `p=0.1`**, and dropout samples from the global RNG every step.
Measured directly — a no-op callback leaves training bit-identical, a callback
that reseeds the global RNG diverges from step one. The hook therefore saves
and restores the global RNG state around the callback, and
`tests/test_train.py::test_checkpoint_hook_does_not_perturb_training` asserts
this with a deliberately hostile callback. Without that guard every number
below would have been silently corrupted by the act of measuring it.

Slice 1's evaluation harness was promoted out of its test file into
`src/jepa/harness.py` so the sweep could reuse it rather than duplicate it;
verified behaviour-preserving by bit-identical comparison of frames, targets
and embeddings, and by reproducing Slice 1's published random-init numbers
exactly.

`src/jepa/momentum_steps_sweep.py` runs the grid: momentum
`(0.0, 0.9, 0.99, 0.996, 0.999, 0.9999)` × seeds `(0, 1, 2)`, each trained once
to 6000 steps with `effective_rank`, `embedding_std`, final loss and loss slope
recorded at 300/1000/3000/6000, plus the `use_ema=False` and random-init
controls. Rows are flushed to disk as they complete so an interrupted sweep
resumes rather than restarting.

**Momentum 0.0 is not the same as `use_ema=False`.** At momentum 0.0 the target
is the online encoder copied every step: **stop-gradient preserved, zero
smoothing**. With `use_ema=False` the target is a separate network trained by
ordinary gradient descent: **no stop-gradient at all**. Separating these two is
what makes the finding below possible.

**Result: the collapse boundary is at the *high*-momentum end, not the low one — and stop-gradient, not EMA smoothing, is what prevents collapse.**

## Results

`effective_rank`, mean over seeds 0/1/2:

| momentum | 300 | 1000 | 3000 | 6000 |
|---|---|---|---|---|
| 0.0 | 1.649 | 2.054 | 2.933 | **3.829** |
| 0.9 | 1.643 | 2.027 | 2.921 | **3.809** |
| 0.99 | 1.635 | 1.938 | 2.737 | **3.768** |
| 0.996 | 1.635 | 1.773 | 2.655 | **3.552** |
| 0.999 | 1.635 | 1.502 | 2.426 | **3.604** |
| 0.9999 | 1.635 | 1.484 | 1.403 | **1.938** |
| `no_ema` control | 1.640 | 1.467 | 1.366 | **1.302** |
| random-init control | — | — | — | 2.611 |

Per-seed at 6000 steps, showing the separation is not an artifact of averaging:

| variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.0 | 3.899 | 3.765 | 3.822 |
| 0.9 | 3.802 | 3.695 | 3.929 |
| 0.99 | 3.478 | 3.795 | 4.029 |
| 0.996 | 3.681 | 3.189 | 3.785 |
| 0.999 | 3.753 | 3.149 | 3.910 |
| **0.9999** | **1.999** | **1.939** | **1.878** |
| **`no_ema`** | **1.336** | **1.387** | **1.182** |
| random-init | 2.934 | 2.461 | 2.437 |

**Three regimes, with no seed overlap between any of them:**

1. **Healthy plateau, momentum 0.0 → 0.999** — 3.15–4.03 at 6000 steps.
   Momentum is very nearly irrelevant across four orders of magnitude, and
   momentum **0.0** is the *best* cell in the grid.
2. **Frozen-target collapse, momentum 0.9999** — 1.88–2.00, *below even the
   untrained random-init encoder* (2.44–2.93).
3. **No-stop-gradient collapse, `use_ema=False`** — 1.18–1.39, the worst.

## The high-momentum confound, and how the loss evidence settles it

This was flagged in the design as the thing most likely to produce a
confidently wrong conclusion. Slice 1 measured random-init `effective_rank` at
2.44–2.93, *higher* than its trained model's 2.35–2.79 — so a healthy-looking
rank at high momentum could equally mean "learned well" or "never moved from
initialization," and rank alone cannot distinguish them. Every cell therefore
also recorded loss.

Final loss at 6000 steps, mean over seeds:

| variant | final loss | loss slope |
|---|---|---|
| 0.0 | 0.09892 | −0.012854 |
| 0.9 | 0.10012 | −0.015772 |
| 0.99 | 0.14002 | −0.013631 |
| 0.996 | 0.14035 | −0.021385 |
| 0.999 | 0.17089 | −0.035797 |
| **0.9999** | **0.00793** | **+0.002210** |
| **`no_ema`** | **0.00000** | — |

The confound resolves cleanly, and **not** in the direction the design
anticipated. Momentum 0.9999's rank is not "healthy but untrained" — it is
genuinely collapsed, *below* random init. Its loss is **10–20× lower** than
every healthy cell, with a slope that has gone positive.

**Mechanistic diagnosis.** At momentum 0.9999 the target retains
`0.9999^6000 ≈ 0.55` of its initial weights after the full run — it is
substantially frozen. A nearly-frozen target is trivial to predict, so the
prediction task itself degenerates: the loss collapses toward zero because
there is nothing left to learn, and the online encoder drifts to a
low-dimensional solution that suffices for the trivial task. The `no_ema`
control reaches the same endpoint by a different route — with no stop-gradient,
both encoders co-adapt to a constant, driving the loss to *exactly* 0.00000.

**Both failure modes present as anomalously LOW loss.** That inversion is the
same signature `projects/em-piml`'s `long-horizon-collapse` thread documents
for its own trivial solution, where every residual-derived adaptive method
failed because the collapse presents as *low* residual rather than high. Two
unrelated problem domains in this repo, same trap: **the degenerate solution is
the cheapest place to be, so the loss looks best exactly where the model is
worst.**

## What this says about Arc 1's question

Arc 1 asks what actually prevents representation collapse. The answer here is
**the stop-gradient**, not the EMA smoothing:

- Momentum 0.0 keeps the stop-gradient and removes *all* smoothing — and is the
  healthiest cell measured (3.83).
- `use_ema=False` removes the stop-gradient — and is the least healthy (1.30).
- Everything between momentum 0.0 and 0.999 is a plateau; the momentum value
  itself buys essentially nothing on this task.

EMA smoothing only becomes decisive at the extreme, and there it *hurts*: past
some point the target is too stale, the prediction task degenerates, and the
representation collapses below random initialization. So momentum has a safe
operating range with a hard ceiling, rather than a "more is better" gradient.

### Relation to prior results — one agreement, one disagreement

This lands squarely on **SimSiam**'s side (Chen & He, arXiv:2011.10566): stop-
gradient is the essential ingredient, and a momentum encoder is not required.
Momentum 0.0 here is close to SimSiam's setup — a stop-gradient copy with no
momentum — and it is the best cell in the grid.

It **disagrees with BYOL** (Grill et al., arXiv:2006.07733), whose target
decay-rate ablation found τ=0 substantially degrades ImageNet performance
(18.8% vs 72.5% at τ=0.99) rather than matching it. Both cannot be universally
true, so the difference is worth naming rather than glossing: the most likely
explanation is task scale. This environment is a single Gaussian blob on an
otherwise-empty 32×32 canvas with a shallow encoder, where the representation
needed is simple enough that EMA's stabilization buys nothing; BYOL's regime is
ImageNet with a ResNet, where a rapidly-moving target plausibly does destabilize
learning. That makes the flatness of the plateau a claim about *this* task, not
about JEPA in general — see lead #4.

Two validity checks passed. At 300 steps every momentum is flat (1.635–1.649),
independently reproducing Slice 1's "the gap is invisible at 300 steps" and
confirming the nested-checkpoint design was necessary. And momentum 0.996 at
3000 steps gives 2.655, inside Slice 1's published 2.35–2.79 band — the
checkpoint hook measures the same thing Slice 1 measured.

A third check came free: an interrupted sweep re-ran several cells, and the
duplicate rows were **bit-identical** to the originals, confirming determinism
across process boundaries.

`tests/test_momentum_collapse_boundary.py` locks in both halves of the finding
— the rank collapse at momentum 0.9999 relative to 0.996, and the degenerate
low-loss signature that explains it.

**Leads for whoever picks this up next:**

1. **Slice 3** — masking ratio and predictor depth, the two axes deliberately
   split out of this slice. Given how flat the momentum plateau turned out to
   be, these may matter more than momentum did.
2. **The boundary between 0.999 and 0.9999 is uncharted.** The grid jumps an
   order of magnitude in staleness across the only interval where anything
   happens. Where exactly the plateau ends — and whether the transition is
   sharp or gradual — is unanswered.
3. **Momentum schedules** (BYOL-style ramps) were deliberately deferred until
   the static surface was known. Now it is: since the plateau is broad and flat,
   a schedule that ramps *toward* 0.9999 would be ramping toward the one
   configuration that collapses. That reframes the question from "what schedule
   is optimal" to "does any schedule beat a flat 0.0."
4. **Test whether the stop-gradient conclusion survives a harder task.** This
   environment's frames are a single blob on an empty canvas; the plateau's
   flatness may be a property of the task's easiness rather than of JEPA.
5. Slice 1's lead #2 (probe R² against a deeper encoder) remains open, and is
   now more attractive: `linear_probe_r2` was additionally found to be
   **nondeterministic** on this data (`torch.linalg.lstsq` returned different
   values on bit-identical inputs for an ill-conditioned 4000×2049 system), so
   any future probe work needs that addressed first.

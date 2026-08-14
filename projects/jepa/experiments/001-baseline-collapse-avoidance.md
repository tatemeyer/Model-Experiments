# Does the full Slice 1 JEPA pipeline avoid collapse and out-probe naive baselines? (issue #69)

Tasks A-D (bouncing-ball generator, encoder/EMA-target/predictor/masking,
training loop, collapse-metric + linear-probe eval harness) built Slice 1's
infrastructure but never ran it end-to-end against a control. This is Arc
1's first real research payoff: does the full EMA-target JEPA actually earn
its complexity, against two cheap ablations — a no-EMA variant (target
encoder trained by ordinary gradient descent, no stop-gradient, the classic
Siamese/BYOL collapse failure mode motivating I-JEPA's own design, Assran et
al. arXiv:2301.08243) and a random-init encoder (no training at all)?

## Implementation

`train_jepa()` (`src/jepa/train.py`) gained a `use_ema: bool = True`
parameter. When `False`, `target_encoder` is a second, independently
initialized `PatchEncoder` (not `EMATargetEncoder`) — its parameters are
added to the optimizer alongside the online encoder and predictor, and the
per-step `target_encoder.update(...)` EMA call is skipped, so the target
moves only via backprop through the same prediction loss. Everything else
(masking, batching, optimizer, seeding order) is untouched — this reuses
the existing training loop with one branch, not a parallel implementation.
The random-init baseline needed no production code: an untrained
`PatchEncoder`, evaluated directly.

**The eval harness needed two non-obvious choices Task D's `eval.py`
deliberately leaves to its caller** (`eval.py` is dataset/model-agnostic by
design — see its docstring):

1. **Pooling patch embeddings into one embedding per input.** Mean-pooling
   across all 64 patches (I-JEPA's own choice for whole-image
   classification) was tried first and drowns the signal here: this task's
   frames are ~55-60/64 pure-background patches, so the mean is dominated
   by the "empty patch" embedding almost regardless of ball position. Two
   alternatives were used instead, for two different purposes: **per-patch
   embeddings treated as independent samples** — `(N * 64, embed_dim)` —
   for collapse metrics (this is the literal "do representations vary
   meaningfully" question, undiluted by averaging); and **all patches
   flattened per frame** — `(N, 64 * embed_dim = 2048)` — for the linear
   probe, since flattening is the only pooling tried that preserves *which*
   patch carried the signal (max-pooling was also tried and destroys
   positional information almost entirely — R² near/below 0 across every
   variant).
2. **Probe train-set size.** Flattened embeddings are 2048-dimensional;
   with a few hundred training rows (`ordinary least squares`, `D≈N`) the
   probe overfits badly and R² swings wildly by seed in *either* direction,
   including runs where an untrained random encoder outscored the trained
   one purely from OLS noise. `n_train=4000` (`D << N`) was needed before
   R² numbers stopped being dominated by this artifact.

Both discoveries came from directly observing seed-to-seed sign flips under
the naive (mean-pool, small-N) harness before trusting any comparison —
consistent with this project's standing "verify before trusting a seed
comparison" rule (`train.py:56-58`,
`projects/em-piml/CLAUDE.md`'s mirrored rule).

**Result: partial — the collapse-avoidance mechanism is real but slow to
appear, and the probe-R² hypothesis does not hold at this scale.**

| variant | metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| full (EMA) | effective_rank (patch-level, 3000 steps) | 2.779 | 2.352 | 2.791 |
| no_ema (ablation) | effective_rank (patch-level, 3000 steps) | 1.389 | 1.459 | 1.250 |
| random_init | effective_rank (patch-level) | 2.934 | 2.461 | 2.437 |
| full (EMA) | embedding_std (patch-level, 3000 steps) | 0.4951 | 0.5311 | 0.4933 |
| no_ema (ablation) | embedding_std (patch-level, 3000 steps) | 0.4779 | 0.3439 | 0.1640 |
| random_init | embedding_std (patch-level) | 0.0179 | 0.0156 | 0.0114 |
| full (EMA) | probe_r2 (position+velocity, held-out) | 0.1491 | 0.4542 | 0.1688 |
| no_ema (ablation) | probe_r2 (position+velocity, held-out) | 0.1883 | 0.1257 | 0.1660 |
| random_init | probe_r2 (position+velocity, held-out) | 0.9762 | 0.1001 | 0.1741 |

All numbers above come directly from `tests/test_baseline_collapse_avoidance.py`'s harness
(`COLLAPSE_STEPS=3000`, `ema_momentum=0.996` default, probe `n_train=4000`/`n_test=300`) and are
reproduced exactly by running it — determinism of both the EMA and no-EMA training paths was
re-verified (bit-identical loss curves across two runs of the same seed) before trusting any
seed-to-seed comparison, per this project's standing rule.

**Collapse metric: `effective_rank` cleanly separates full from the no-EMA ablation, with no
overlap across any seed** (full: 2.35–2.79; no_ema: 1.25–1.46) — the issue's first hypothesis
holds, once `effective_rank` rather than `embedding_std` is used as the discriminator.
`embedding_std` does *not* reliably separate them: at seed 0 no_ema's std (0.478) is nearly
identical to full's (0.495), and at seed 2 no_ema's std (0.164) drops well below full's (0.493) —
inconsistent across seeds, unlike the effective_rank gap. Both trained variants clear
random_init's std by 10–30×, so *some* signal in embedding_std distinguishes "trained" from
"untrained" — it just doesn't distinguish full from no_ema specifically.

**Linear-probe R²: does not confirm the issue's second hypothesis.** Full's probe R² (0.149–0.454)
is not reliably above random_init's (0.100–0.976) — at seed 0, random_init actually scores highest
of all nine cells in the table (0.976), while at seeds 1/2 all three variants land within the same
0.10–0.45 band with no consistent ordering. This directly contradicts "full model clears a
threshold that random-init fails to clear."

**Mechanistic diagnosis.**

*Why effective_rank and not embedding_std detects the collapse gap:* the no-EMA ablation isn't
collapsing to a single point (embedding_std stays well above zero, sometimes above full's) — it's
collapsing onto a low-dimensional *subspace*, large variance concentrated along one or two
directions rather than spread across the embedding's 32 dimensions. This is "dimensional
collapse," a documented SSL failure mode distinct from total collapse (Hua et al., "On Feature
Decorrelation in Self-Supervised Learning", ICCV 2021, arXiv:2105.00470; Jing, Vincent, LeCun,
Tian, "Understanding Dimensional Collapse in Contrastive Self-supervised Learning",
arXiv:2110.09348) — per-dimension variance (embedding_std) is blind to it by construction,
since correlated dimensions can each carry high individual variance while jointly spanning almost
no volume; effective_rank's SVD-entropy formulation is exactly the metric built to catch this.
Without a stop-gradient target, nothing stops the online+target pair from jointly finding a
"cheap" solution that satisfies the prediction loss along a narrow direction (e.g. a rough
foreground/background split) without needing the full richness of the embedding space — a milder,
slower-forming version of full collapse. It's slow-forming: at `train.py`'s STEPS=300 default,
neither variant has moved far enough for the gap to appear (full and no_ema's ranks are both
~1.6); the full model's rank keeps climbing with more training (1.62 → 1.97 → 2.78 → 3.61 at
300/1000/3000/6000 steps, seed 0) while no_ema's stays flat or declines (1.61 → 1.47 → 1.39 →
1.34) — the two only pull apart once training runs long enough for EMA's smoothing to matter.

*Why the probe-R² hypothesis fails:* this task's frames are a single Gaussian blob on an
otherwise-empty canvas — ball position is close to a linear function of pixel intensities
(intensity-weighted centroid), so a shallow, *even untrained*, patch encoder (one strided conv +
linear projection) already approximately preserves this via something close to a random linear
projection of the input (a Johnson–Lindenstrauss-style argument: random projections tend to
preserve linearly-recoverable structure). JEPA training doesn't need to — and apparently doesn't —
add much on top of what a random projection already gives a sufficiently-large flattened linear
probe access to, at least for `PatchEncoder`'s current shallow architecture and this task's
simplicity.

`tests/test_baseline_collapse_avoidance.py`'s single combined slow test locks in both findings:
the collapse-metric result via `COLLAPSE_RANK_THRESHOLD = 1.8` (full and no_ema on opposite sides,
verified across seeds 0/1/2), and the probe-R² negative result via `PROBE_R2_FLOOR = 0.05` (a
regression floor — both variants must clear it, but full is deliberately *not* asserted to beat
random_init, since it doesn't reproducibly).

**Leads for whoever picks this up next:**
1. Slice 2 (EMA momentum / masking ratio / predictor depth sweep) should
   specifically vary `ema_momentum` and `steps` jointly — this experiment
   found the collapse gap needs ~3000 steps at momentum 0.996 to appear at
   all, suggesting momentum tuned for the training-length budget (BYOL's
   own schedule ramps momentum up over training rather than holding it
   fixed) may reveal the effect at a fraction of the compute.
2. The probe-R² negative result may be an artifact of `PatchEncoder`'s
   shallow architecture (one strided conv + linear projection) rather than
   JEPA training itself — a near-linear random projection of this task's
   pixels may already retain most of the linearly-recoverable position
   signal (ball position is close to a linear function of pixel
   intensities, an intensity-weighted-centroid argument), leaving little
   headroom for training to improve on. Worth re-testing probe R² against
   a deeper/nonlinear-probe-resistant encoder before concluding JEPA
   training doesn't help position-decodability here.
3. Velocity is not included as a separate breakdown in the table above but
   is structurally unrecoverable from any of these variants — these are
   single-frame (`n_frames=1`) samples with no motion cues, so the
   combined position+velocity R² is a lower bound relative to a
   position-only probe. Confirmed directly: velocity-only R² was
   consistently ≤0 for every variant tested during this investigation.

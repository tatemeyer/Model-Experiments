# Does a PirateNets-style adaptive-residual architecture close the two-mode spectral-bias gap? (issue #41)

Network capacity was widened for the two-mode target's own predecessor
problem (issue #10, single-mode `num_bands=4`) but never for this
target specifically — and issue #10's fix was a *wider* plain MLP, not
a *different depth mechanism*. Wang, Li, Chen, Perdikaris, "PirateNets:
Physics-Informed Deep Learning with Residual Adaptive Networks" (JMLR
2024, arXiv:2402.00326) propose a depth mechanism designed specifically
for PINNs: a stack of adaptive-residual blocks, gated by fixed random
Fourier features, that starts as an effectively shallow network and
progressively deepens during training via a learnable per-block scalar
(rather than needing careful initialization to train stably at depth at
all). This issue tests whether that mechanism — not just more width —
closes the two-mode gap.

`PirateNetCavityPINN` (`src/em_piml/model.py`) implements the paper's
own formulation (eq. 4.1-4.8) directly: a fixed (non-trained), random
Fourier embedding `Phi(x) = [cos(Bx); sin(Bx)]`, `B ~ N(0,
fourier_scale^2)`, feeds two gate networks `U`/`V` (computed once),
reused by every `PirateNetBlock`; each block computes two gated dense
transforms from `U`/`V` and blends its own output with its input via a
learnable scalar `alpha`, **initialized to 0** so the block starts as an
exact identity map. `fourier_scale=1.0` is an explicitly untuned default
— the paper reports no wave-equation benchmark of its own to inherit a
value from (only advection, Navier-Stokes, diffusion), same situation
RWF's `mu`/`sigma` was in (issue #39).

**Structural check, not just a training-run number**: at construction,
with every block's `alpha=0`, the whole network must reduce exactly to
`W_out @ Phi(x)` (eq. 4.8) — a fast (no training), exact (not
approximate) test of the paper's own "effectively shallow at init"
claim, verified directly (`test_shallow_at_init_matches_paper_eq_4_8`,
`torch.equal`, not `allclose`) before trusting any trained-model number
below.

**A real, project-specific constraint that changed the shipped config:
PirateNets' reference depth (4 blocks x 3 dense layers = 12 sequential
layers, each needing 2nd-order autograd through the whole chain for the
PDE residual) is far more expensive per step than this project's
existing 3-layer bodies.** At the paper-scale defaults (`num_blocks=4`,
`steps=4000`, matching `train_cavity_two_mode`'s step budget), a single
seed measured **767-796s** — over an order of magnitude past this
project's "well under a minute" convention (`CLAUDE.md`, "Model and
training"), and the issue's own constraint explicitly allows scaling
down rather than skipping the runtime check in exactly this situation.
**Shipped config: `num_blocks=2`, `steps=1000`** (~110-115s/seed) — half
the paper's block count, a quarter of the step budget.

**Result: a real improvement over the plain baseline, but does not beat
either existing Fourier-embedding fix, at this reduced budget.**

| variant | relative L2 (seeds 0/1/2/7) |
|---|---|
| plain `CavityPINN` (`022-...md`) | 0.7699, 0.7876, 0.7944, 0.7947 |
| `FourierCavityPINN` `num_bands=2` (`022-...md`) | 0.6995, 0.7029, 0.7049, 0.7063 |
| `FourierCavityPINN` `num_bands=4` L-BFGS (`025-...md`) | 0.7023-0.7128 |
| PirateNets (`num_blocks=2`, `steps=1000`) | 0.7278, 0.7288, 0.7338, 0.7407 |

PirateNets' range (0.7278-0.7407) sits clearly below the plain
baseline's (0.7699-0.7947) — a real, consistent improvement — but above
both Fourier-embedding variants' ranges (0.6995-0.7063,
0.7023-0.7128). The adaptive-residual depth mechanism helps relative to
doing nothing architecturally, same shape of result as RWF's weight
reparameterization (issue #39), but a different, deeper mechanism than
RWF and still short of what the existing (much cheaper) Fourier
embedding already achieves.

**Supplementary, not the shipped comparison — partial paper-scale
numbers, gathered as a side data point while the reduced-budget sweep
above ran**: `num_blocks=4`, `steps=4000` reached 0.7151 (seed 0) and
0.7195 (seed 1) before this write-up's time budget moved on (2 of 4
seeds only — not a complete, decision-grade comparison). Both are
better than the shipped reduced-budget numbers, consistent with more
capacity/training genuinely helping somewhat, and edge closer to (but,
on these two seeds, still don't clearly beat) the Fourier-embedding
baselines' range. Whether the full 4-seed paper-scale run would
actually close the gap is a real open question this issue's own runtime
budget didn't allow settling — see leads below.

**Pointwise check (same method as `022-...md`/`039-...md`) at seed 0's
shipped-config model confirms the same failure mechanism as every prior
fix in this thread.** At `x=0.5625, t=0` (an `n=8` peak, true field
`0.9904`, `n=1`-only envelope `0.4904`): PirateNets predicts `0.5011` —
tracking the `n=1`-only envelope almost exactly, missing the `n=8`
contribution entirely, same shape of failure as every model in this
thread so far. Plausible mechanism (untested directly): `fourier_scale=1.0`
means `B`'s entries are typically `O(1)` in magnitude, so the embedding's
random frequencies are mostly far below the `~8*pi` (`≈25`) the `n=8`
mode needs — a random-frequency analogue of `num_bands=2`'s fixed
`{pi, 2pi}` basis missing `8*pi` (`022-...md`). Unlike a fixed discrete
basis, a large enough `fourier_scale` *could* in principle place
non-trivial density near the needed frequency by chance; `fourier_scale=1.0`
just doesn't.

`tests/test_piratenets_two_mode.py`: the fast structural test above,
plus one `@pytest.mark.slow` regression test (seed 0, shipped config)
asserting `relative_l2 > 0.5`, matching `test_two_mode_superposition.py`'s/
`test_rwf_two_mode.py`'s `FAILURE_LOWER_BOUND` convention. The full
4-seed sweep (`piratenets_sweep.py`) is deliberately not part of the
pytest suite, same precedent as `point_draw_sweep.py`/`rwf_sweep.py`.

**Determinism**: verified before trusting any number above — same seed
reproduces a bit-identical `state_dict` across two calls to
`train_piratenets_two_mode` (spot-checked at a small step count before
the full sweep).

**Leads for whoever picks this up next:**
1. Whether the full paper-scale config (`num_blocks=4`, `steps=4000`)
   actually beats the Fourier-embedding baselines is genuinely open —
   only 2 of 4 seeds were gathered (0.7151, 0.7195), both better than
   the shipped reduced-budget numbers but not conclusively past
   0.6995-0.7128. Finishing that 4-seed run (~50 min total at this
   sandbox's per-seed cost) would settle it.
2. `fourier_scale` is untuned (borrowed default, no wave-equation
   precedent in the paper) — per the pointwise diagnosis above, a
   larger value (raising the typical sampled frequency magnitude toward
   the `n=8` mode's `~8*pi` need) is a concrete, testable lever neither
   this issue nor RWF's analogous `mu`/`sigma` sweep touched.
3. Combining PirateNets' depth mechanism with the project's own
   deterministic Fourier embedding (guaranteeing `8*pi` basis coverage,
   rather than relying on random-frequency luck) is architecturally
   straightforward and untried — would isolate "does adaptive depth
   help *given* adequate frequency coverage" from "does PirateNets'
   own embedding provide adequate coverage."

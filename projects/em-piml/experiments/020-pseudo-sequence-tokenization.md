# Does pseudo-sequence tokenization beat the raw-coordinate baseline? (issue #20)

Every embedding experiment up to this point (issues #4/#6/#8/#10/#11, see
`num-bands-gap/`) varied the Fourier-feature band count, the optimizer, or
the network capacity — never the *architecture* consuming the input. This
issue tests the project's actual motivating research question
(tokenization/embedding for PIML, see issue #2's origin) via a literature
pass and a genuinely different axis: does turning each pointwise `(x, t)`
input into a short *token sequence*, processed by a small Transformer, do
better than a plain MLP?

**Literature pass (alphaXiv):** three tokenization schemes were evaluated for
fit against this project's 1D, single-fixed-equation, CPU-only scale:

1. **Equation tokenization** (Lorsung et al., "Physics Informed Token
   Transformer" / PITT, arXiv:2305.08757) tokenizes the *governing equation
   itself* as a symbolic sequence, to condition an operator-learning model
   across a *family* of parametric PDE instances (varying viscosity, forcing
   amplitude, etc.). Ruled out: em-piml's baseline is one fixed equation
   (fixed `L`, `c`, `n`) — there's no varying equation instance to tokenize or
   condition on. Would only become relevant if the project generalizes to a
   family of cavity problems.
2. **Patch-based multi-scale tokenization** (e.g. MeshTok, arXiv:2606.04366)
   adaptively patchifies large gridded 2D/3D PDE fields for Transformer
   foundation models (AMR-inspired, refining high-activity regions). Ruled
   out: em-piml's input is scalar `(x, t)` point samples, not a discretized
   field with spatial structure to patchify.
3. **Pseudo-sequence tokenization** (Zhao et al., "PINNsFormer", ICLR 2024,
   arXiv:2307.11833) expands each `(x, t)` into a short sequence of `k` nearby
   timesteps `{[x,t], [x,t+dt], ..., [x,t+(k-1)dt]}`, processed by a small
   encoder-decoder Transformer with a sequential PINN loss and a `Wavelet`
   activation (`omega1*sin(x) + omega2*cos(x)`, learnable `omega1`/`omega2` —
   the paper's own ablation found this necessary; ReLU/Sigmoid fail outright,
   plain `Sin` is inconsistent). This one fit: no equation family or spatial
   field needed, and their own closest benchmark (a 1D wave equation,
   `d^2u/dt^2 - beta^2 d^2u/dx^2 = 0`, sinusoidal IC, Dirichlet BCs) is
   essentially the same equation family as our cavity mode. Implemented as
   `PseudoSequenceCavityPINN` (`src/em_piml/model.py`) — no LayerNorm, per the
   same ablation (it didn't help and sometimes caused NaN paired with
   Wavelet); decoder has no self-attention, reusing the encoder's own
   embeddings as its query, per the paper's design.

**The sequential loss needed a non-obvious derivative trick.** The encoder's
self-attention mixes information across the `k` sequence positions, so a
naive `torch.autograd.grad(u, t_seq, grad_outputs=ones)` call sums
cross-position contributions instead of isolating each position's own
derivative (`d(u_j)/d(t_seq_j)`, per PINNsFormer eq. 5) — `_sequence_derivative`
(`src/em_piml/train.py`) extracts this per-position Jacobian diagonal via one
backward pass per sequence position. **Verified directly against finite
differences before trusting any result from it** (perturbed a single sequence
position's input, held the rest fixed, compared to the analytical gradient —
matched to 4 significant figures). This makes the sequential PDE residual (2nd
order in both `x` and `t`) cost `O(4k)` backward-style passes per collocation
batch, on top of the forward pass.

**Result: does not beat the baseline — performs markedly worse, and not for
lack of trying.** Shipped config (`train_pseudo_sequence_cavity`: `d_model=16,
heads=2, ff_dim=32, num_layers=1, k=3, dt=1e-3`, Adam `lr=3e-3`, `steps=600`,
`n_collocation=30/n_boundary=16/n_initial=16`) reaches relative L2 error
**0.958-1.383 across seeds 0/1/2/7** — worse than the raw-coordinate baseline's
0.026-0.046 by more than an order of magnitude, and comparable to or worse
than the total-failure `num_bands=4` Adam case from `num-bands-gap/
004-fourier-embedding.md` (~1.0-1.04).

This was **not** a case of insufficient tuning — the following were all tried
and none closed the gap, each isolating a different candidate explanation:
- **A bug in the derivative math** — ruled out by the finite-difference check
  above.
- **Overfitting a small fixed L-BFGS collocation set** (this repo's own
  `num-bands-gap/006-lbfgs-optimizer.md` / `008-denser-collocation.md`
  precedent for why L-BFGS can fail this way) — ruled out by switching to
  Adam with fresh point resampling every step (removes the "memorize a fixed
  set" failure mode entirely) and running up to 1500 steps; same ~1.0-1.4
  plateau, reached within the first few hundred steps and flat thereafter
  despite the model's own training loss (PDE residual + BC + IC) converging
  to ~1e-4 — i.e. the model satisfies its own loss almost exactly while still
  not matching the analytical field.
- **`dt` mismatched to this problem's timescale** (paper's own benchmarks are
  mostly on `t` in `[0, 1]`; our `PERIOD` is ~6.28) — ruled out by rerunning
  with `dt` scaled up 6x; same outcome.
- **Insufficient training** — ruled out by both the loss-convergence evidence
  above and by denser collocation (tried up to 400 points) making no
  qualitative difference.

**Working interpretation:** the sequential loss's PDE-residual/BC/IC terms are
inherently local/pointwise constraints (as they are for the plain baseline
too) — satisfying them at every sampled point does not, by itself, guarantee
recovering the *unique* global solution, and this architecture apparently
finds an alternate function that satisfies them without matching the true
field elsewhere. The plain baseline avoids this in practice, but this
architecture's much higher per-step cost (`~15-30x` the baseline, from the
`O(k)` derivative extraction above) forces a far smaller model and point
budget within a CI-suitable wall-clock time, which may be part of why. It's
also worth noting PINNsFormer's own reported gain on their closest analog
(1D-wave) was modest without additional machinery: 0.335 (plain PINN) to 0.283
(PINNsFormer alone) — their headline 0.058 result required also adding NTK
adaptive loss-term reweighting on top, which neither this implementation nor
anything else in this repo currently has. A faithful implementation might
credibly have landed closer to "no clear win" even in the best case; landing
substantially *worse* than the baseline is the more surprising part of this
finding.

`tests/test_pseudo_sequence.py` asserts relative L2 error stays under `2.0` —
not an accuracy bar (there isn't one to clear here) but a regression check on
this documented negative result, with margin above the observed 1.383 worst
case. Determinism (same seed -> same result) was re-verified per this
project's standing lesson before trusting any of the numbers above.

**Leads for whoever picks this up next:**
1. ~~NTK-style adaptive loss-term reweighting~~ — the one lever the paper's
   own ablation shows matters most (0.283 -> 0.058) and was entirely
   unexplored at the time. Tried later against the long-horizon collapse in
   `long-horizon-collapse/034-ntk-reweighted-long-horizon.md` — result: it
   actively hurt that failure mode. Never separately re-tried against *this*
   issue's specific baseline-comparison target.
2. The `O(k)` per-position Jacobian-diagonal extraction is the compute
   bottleneck (see above) — a vectorized/batched implementation (e.g. via
   `torch.func.jacrev`/`vmap`, untried here to keep this issue's scope to the
   tokenization question itself) could afford a much larger model/collocation
   budget within the same wall-clock time, which this issue couldn't rule out
   as the actual fix.
3. This toy problem is a single low-frequency mode with a known closed form —
   PINNsFormer's own headline results are on *harder* PDEs (high-frequency
   convection, PINN failure-mode benchmarks) where a plain MLP already
   struggles. It's possible pseudo-sequence tokenization's real benefit only
   shows up once the baseline itself is failing for a reason this
   architecture specifically addresses (spectral bias / temporal
   error-propagation) — this lead was later followed up against the
   long-horizon collapse specifically, see
   `long-horizon-collapse/030-pseudo-sequence-long-horizon.md`.

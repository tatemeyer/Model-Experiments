# Does the optimizer explain the num_bands=4 instability? (issue #6)

["Spectral bias in physics-informed and operator learning: Analysis and
mitigation guidelines"](https://www.alphaxiv.org/abs/2602.19265)
(Khodakarami et al., Brown/Karniadakis group, Feb 2026) argues via NTK
theory that this kind of instability under higher-frequency inputs is
primarily *dynamical* (an Adam/first-order-optimizer limitation — each
frequency mode's effective learning rate scales with its NTK
eigenvalue, which decays sharply with frequency), not a representational
failure of the embedding, and reports quasi-second-order optimizers
(SOAP, L-BFGS, SS-Broyden) largely resolving it.

`train_fourier_cavity_lbfgs` (`src/em_piml/train.py`) tests this
directly: same `num_bands=4` `FourierCavityPINN`, same loss
construction, `torch.optim.LBFGS` (built into PyTorch, no new
dependency) instead of Adam. Since L-BFGS assumes a fixed/deterministic
objective across its internal line-search evaluations, collocation
points are sampled once per run rather than resampled every step (see
`_train_pinn_lbfgs`), unlike the Adam path.

**Result: partial support, not full resolution.**

- Adam at `num_bands=4`: relative L2 error ~1.0-1.04 — doesn't learn the
  solution at all, with or without more training steps.
- L-BFGS at `num_bands=4` (from a fresh random init): converges to
  ~0.79-0.88 relative L2 across seeds 0/1 and a sweep of
  `outer_steps`/`max_iter` budgets (10-100 outer steps, 20-100 inner
  iterations each) — a real, substantial improvement over Adam's total
  failure, but the error **plateaus** there; more iterations stop
  helping past `outer_steps=50, max_iter=50` (the shipped default, ~40s).
- Tried Adam-warmup-then-L-BFGS too (the paper mentions L-BFGS is "often
  used after Adam warm-up"): 1000 Adam steps first (still ~1.03-1.04,
  consistent with Adam's failure above) then L-BFGS refinement converges
  to the *same* ~0.86 plateau as starting L-BFGS from scratch. Warmup
  doesn't change the outcome — this looks like a genuine local optimum
  of this loss landscape for this architecture/point-budget, not an
  initialization sensitivity.

So at the time: optimizer choice mattered a lot (L-BFGS clearly did
something Adam couldn't) but didn't fully explain the instability by
itself at 200 points. See `008-denser-collocation.md` — lead #1 there
turned out to be most of the answer.

**Update (issue #38):** the "genuine local optimum" interpretation above
was tested directly, not just assumed — "FP64 is All You Need" (Xu et al.,
NeurIPS 2025, arXiv:2505.10949) argues failure modes like this one are
often FP32 precision artifacts (L-BFGS's convergence test firing
prematurely), not real local optima. Rerunning this exact 32-hidden/
200-point configuration under `torch.float64` at the same iteration budget
still plateaus at 0.889-0.922 relative L2 (seeds 0/1/2/7) — no better than
the FP32 numbers above. Not a precision artifact; see
`038-fp64-precision.md` for the full comparison and a mechanistic note on
why FP64 costs ~10x more wall time here without changing the outcome.

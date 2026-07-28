# Does SOAP close the rest of the num_bands=4 gap? (issue #11)

Note: developed in parallel with issue #10 — at the time this was
written, the `num_bands=4` gap was still open (issues #6/#8's
0.065-0.104). Issue #10 independently closed that same gap via network
capacity instead (see `010-network-capacity.md`). Both are documented;
SOAP is a genuinely different mechanism (optimizer, not architecture)
and remains a valid, separately useful result even though it's no
longer the only fix.

Issues #6/#8 deliberately stayed dependency-free (`torch.optim.LBFGS` only)
and got `num_bands=4` from a ~0.8 plateau down to 0.065-0.104 — a real
improvement, but short of the standard `0.1` bar (worst seed landed at
0.104). Khodakarami et al. report SOAP and SS-Broyden as substantially
stronger than L-BFGS for exactly this failure mode. This issue tests SOAP,
with a new dependency justified for the first time in this thread.

**Dependency adopted: `pytorch-optimizer` (PyPI, `kozistr/pytorch_optimizer`).**

- **What:** a 100+-optimizer PyTorch collection; we use exactly one class,
  `pytorch_optimizer.SOAP` (Vyas et al., "Improving and Stabilizing Shampoo
  using Adam", arXiv:2409.11321) — the official SOAP paper's own
  formulation, not a from-scratch reimplementation risk.
- **Why trusted:** actively maintained (v3.10.1, released within the last
  two months of this issue; 92 releases total, not a one-off drop), 421
  GitHub stars, CI + Codecov badges, Apache-2.0, single maintainer but a
  multi-year (since 2021) consistent release cadence — considered against
  the alternative of vendoring the paper authors' reference
  implementation (`nikhilvyas/SOAP`, which isn't published to PyPI and
  explicitly tells users to copy the file in, i.e. take on the
  maintenance burden ourselves) or `heavyball` (also credible, 334 stars,
  but younger project, more actively-changing API surface per its own
  migration-guide history). `pytorch_optimizer` was the better-established
  choice of the two PyPI-published options.
- **What it costs:** one new transitive-dependency-free package (only
  depends on `torch`/`numpy`, both already present); adds ~300KB wheel.
  We only use `SOAP`, none of the other 99+ optimizers it ships, so most
  of its surface area is dead weight we don't exercise or test — accepted
  because the alternative (reimplementing Shampoo-preconditioned SOAP
  in-repo) is far more maintenance risk than importing one class from an
  established package.

`train_fourier_cavity_soap` (`src/em_piml/train.py`) mirrors
`train_fourier_cavity_lbfgs`'s controlled-comparison setup exactly: same
`num_bands=4` `FourierCavityPINN`, same fixed (not resampled)
`n_collocation=2000/n_boundary=400/n_initial=400` point set, only the
optimizer swapped — `pytorch_optimizer.SOAP` (library defaults: `lr=3e-3`,
`betas=(0.95, 0.95)`, `precondition_frequency=10`) run for `steps=2000`
plain (no closure/line-search — unlike L-BFGS, SOAP doesn't need one).

**Result: fully closes the gap, doesn't just clear the bar.**

| seed | L-BFGS (issue #8) | SOAP (this issue) |
|---|---|---|
| 0 | 0.098 | 0.0357 |
| 1 | 0.104 | 0.0304 |
| 2 | 0.096 | 0.0233 |
| 7 | 0.065 | 0.0232 |

SOAP lands at 0.0232-0.0357 across seeds 0/1/2/7 — not merely under the
standard `0.1` bar, but in the *same range as the plain-coordinate Adam
baseline* (0.026-0.046, see `../../CLAUDE.md`) and tighter/more consistent
across seeds than L-BFGS's spread. This matches Khodakarami et al.'s
claim that SOAP is a stronger fix than L-BFGS for this exact spectral-bias
failure mode. `tests/test_fourier_soap.py` asserts `< 0.08` (~2.2-3.4x
margin above the observed 0.0357 worst case, matching the margin style of
`test_baseline_cavity.py`).

**Performance note (sandbox-specific, not a general claim):** SOAP
recomputes a Shampoo-style eigenbasis preconditioner every
`precondition_frequency` steps (extra `eigh` calls on top of an Adam-like
update), which made it dramatically more sensitive to CPU oversubscription
than Adam/L-BFGS when this issue was developed — on a contended sandbox
shared with unrelated concurrent processes, per-step wall time dropped
~200x (6.6s to 0.03s) after pinning `torch.set_num_threads(1)` for the
duration of the SOAP training call (see `_train_pinn_soap`), restored
afterward so it doesn't leak into other tests. 2000 steps takes ~60-80s
single-threaded, comparable to the existing L-BFGS test's ~40s. Revisit if
a quieter CI runner makes this pinning unnecessary or counterproductive —
it was not re-validated on an uncontended machine.

**Leads still open**, in case this is picked up further:
1. `SS-Broyden` (the paper's other top performer) wasn't tried — SOAP
   already met the bar, so there was no need to justify a second new
   dependency in the same issue.
2. SOAP hyperparameters (`lr`, `betas`, `precondition_frequency`) were
   left at library defaults — untuned. Given the result already beats
   the baseline range, tuning wasn't pursued, but there may be room to
   reduce `steps` below 2000 without losing accuracy.
3. Network capacity, embedding `num_bands` values other than 4, and the
   density non-monotonicity from issue #8 remain unexplored with SOAP —
   out of scope for this issue's specific question (does SOAP close the
   `num_bands=4` gap at the existing shipped point-set default).

# Does a two-mode superposition break the baseline, and does Fourier embedding fix it? (issue #22)

Every embedding experiment up to issue #20 shared a blind spot: the
baseline problem (a single low-frequency fundamental mode, `n=1`) is
already solved well by a plain coordinate MLP (0.026-0.046 relative L2),
so there was no real failure mode for a fancier architecture to fix —
issue #4's Fourier features were "neutral," issue #20's pseudo-sequence
tokenization was actively worse, but neither result distinguished
"tokenization doesn't help" from "there was nothing here to help with."
This issue tests a harder target designed specifically to induce a
well-documented PINN failure mode: **spectral bias** (Rahaman et al.
2019; Xu et al.'s F-Principle, arXiv:1901.06523) — coordinate MLPs learn
low-frequency content fast and high-frequency content far more slowly,
or not at all.

`analytical_field_two_mode` (`src/em_piml/physics.py`) adds a second,
much higher spatial mode (`N_MODE_2=8`) on top of the existing
fundamental (`n=1`), equal amplitude (`0.5` each). This is still exact —
the wave equation is linear and both terms individually satisfy the
same Dirichlet BCs and `dE/dt(x,0)=0`, so their sum does too — no
numerical solver, no new dependency. `PERIOD`/`OMEGA` stay derived from
`N_MODE=1` only, so the existing training/eval domain (one period of the
fundamental) already spans 8 full spatial half-cycles and 8 full
temporal cycles of the added mode with no domain-size change.
`train_cavity_two_mode` and `train_fourier_cavity_two_mode`
(`src/em_piml/train.py`) reuse `train_cavity_baseline`/
`train_fourier_cavity_baseline`'s exact shipped defaults (steps,
`n_collocation`, `lr`, architecture, `num_bands=2`) — the *only*
variable is the target field, via a new `field_fn` parameter threaded
through `_pinn_loss`/`_train_pinn_adam`/`evaluate_relative_l2_error`
(defaults to the original `analytical_field`, so every existing call
site and test is bit-for-bit unaffected).

**Result: the baseline does fail here, exactly as predicted — and
Fourier features only partially help, also exactly as predicted.**

| model | relative L2 (seeds 0/1/2/7) |
|---|---|
| plain `CavityPINN` | 0.7699, 0.7944, 0.7947, 0.7876 |
| `FourierCavityPINN` (`num_bands=2`) | 0.7029, 0.6995, 0.7049, 0.7063 |

The plain baseline lands at ~0.77-0.79 — over an order of magnitude
worse than its own 0.026-0.046 range on the single-mode target, with
nothing else about the problem or training loop changed. Fourier
features give a real but small improvement (~0.70 vs ~0.78), nowhere
close to fixing it.

**A pointwise check confirms this is genuinely the spectral-bias
mechanism, not just a worse aggregate number.** Evaluating both trained
models at `t=0` across `x` values chosen to land on the `n=8` mode's
peaks/troughs (not its zero-crossings, which the two fields share):
both models' predictions track the smooth `n=1`-only envelope
(`0.5*sin(pi*x)`) almost exactly and are essentially blind to the true
field's `n=8` ripple (which swings roughly -0.4 to +0.99 at the same
points, vs. predictions that stay smoothly within the `n=1` envelope's
0.05-0.47 range). E.g. at `x=0.5625`: true field `0.9904`, `n=1`-only
component `0.4904`, plain-model prediction `0.4410`, Fourier-model
prediction `0.4687` — both models predicted almost exactly the `n=1`
envelope value and missed the `n=8` contribution entirely. This is
textbook spectral bias (cf. Tancik et al., "Fourier Features Let
Networks Learn High Frequency Functions in Low Dimensional Domains",
NeurIPS 2020, arXiv:2006.10739, Figure 3c, which shows the identical
shape of failure in a non-PDE 1D regression setting: a low-frequency
component converges while a simultaneously-present high-frequency
component doesn't).

**Why Fourier features only partially help, explained by the embedding's
own frequency support, not a mystery:** `FourierCavityPINN`'s embedding
(`src/em_piml/embeddings.py`) normalizes `x` to `[0,1]` and uses
frequencies `2^0*pi, 2^1*pi, ..., 2^(k-1)*pi` for `num_bands=k`. At the
shipped `num_bands=2` default, that's `{pi, 2pi}` — no basis component
at `8*pi`, which is exactly the frequency the `n=8` spatial mode needs
(`sin(8*pi*x/L)` becomes `sin(8*pi*u)` in normalized coordinates). The
pointwise check above confirms this precisely: the Fourier model's
predictions at the `n=8` peaks/troughs are nearly identical to the plain
model's, both tracking only the `n=1` envelope — `num_bands=2` doesn't
give the network anything new to work with for this specific target.
`num_bands=4` (frequencies up to `8pi`) would be the literature-predicted
fix; deliberately not tested here, per this issue's constraint to hold
the shipped `num_bands=2` default fixed and treat that as its own
finding rather than silently retuning mid-issue.

`tests/test_two_mode_superposition.py` asserts both models' relative L2
error stays `> 0.5` — not an accuracy bar (there's no bar to clear here)
but a regression check on this documented failure signature, with
~2.6-3.4x margin below the observed 0.70-0.79 range.

**Leads for whoever picks this up next:**
1. ~~Sweep `num_bands` up to `4`/`6`+ on this two-mode target~~ — done,
   see `025-num-bands-sweep.md`. Result: raising `num_bands` does **not**
   close this gap the way it closed the single-mode `num_bands=4`
   instability.
2. Issue #23 (long time-horizon/causality, see
   `../long-horizon-collapse/023-long-horizon-causal.md`) is a queued,
   independent alternative failure mode — not blocked on this issue's
   outcome.
3. Now that there's a real, reproducible failure mode with a known
   mechanism (missing basis frequency), it's also a legitimate testbed
   to revisit issue #20's pseudo-sequence tokenization
   (`../020-pseudo-sequence-tokenization.md`) — PINNsFormer's own
   headline results are specifically on PDEs where a plain MLP already
   fails, which wasn't true of this project's original baseline.

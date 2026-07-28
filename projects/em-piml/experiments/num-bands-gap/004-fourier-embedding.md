# Embedding experiment #1: Fourier features (issue #4)

`FourierCavityPINN` (`src/em_piml/model.py`) embeds `(x, t)` — each
normalized to `[0, 1]` — via a NeRF/Tancik-style positional encoding
(`src/em_piml/embeddings.py`) before the *same* MLP body shape as
`CavityPINN`: `[u, sin(2^0 pi u), cos(2^0 pi u), ..., sin(2^(k-1) pi u),
cos(2^(k-1) pi u)]` per scalar, for `x` and `t` independently.
`train_fourier_cavity_baseline` reuses the exact same training loop,
loss construction, steps, and optimizer as the baseline — the only
variable is the input representation.

**Finding:** at `num_bands=2` (the shipped default), performance is
statistically indistinguishable from the raw-coordinate baseline on
this problem at this scale — relative L2 error 0.033-0.043 across
seeds 0/1/2/7, versus the baseline's 0.026-0.046. Fourier features
neither clearly help nor hurt here.

**More interesting finding:** `num_bands=4` and `num_bands=6` — more
expressive embeddings — *destabilize* training at the same fixed
learning rate/step budget as the baseline (relative L2 error ~0.95-1.06,
i.e. it doesn't learn the solution at all). This wasn't chased further
because doing so (e.g. lowering the learning rate for higher band
counts) would break the controlled-comparison premise of this issue —
but it's a concrete, actionable lead for whoever picks up the next
embedding iteration: naive higher-frequency Fourier features need their
own optimization treatment (lower LR, warmup, or SIREN-style init), they
don't drop in for free at a fixed budget tuned for raw coordinates.

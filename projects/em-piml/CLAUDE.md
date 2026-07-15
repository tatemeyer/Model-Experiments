# em-piml

Physics-informed ML for electromagnetics. Motivating research interest:
tokenization/embedding schemes for PIML on EM problems (see issue #2 for
origin). This baseline exists to have a verified walking skeleton before
that research work branches off it — it is intentionally a plain
coordinate-input MLP, not a tokenization/embedding experiment.

## Problem being solved (baseline)

The fundamental mode of a 1D perfect-electric-conductor (PEC) cavity of
length `L` — a direct 1D reduction of Maxwell's equations to the wave
equation:

```
d^2E/dt^2 = c^2 * d^2E/dx^2,   E(0,t) = E(L,t) = 0
```

with closed-form solution `E(x,t) = A * sin(n*pi*x/L) * cos(omega*t)`,
`omega = n*pi*c/L` (see `src/em_piml/physics.py`; `L = c = n = A = 1`).
Closed-form means verification is exact, not eyeballed.

## Model and training

`CavityPINN` (`src/em_piml/model.py`) is a 3-layer, 32-wide tanh MLP
taking `(x, t)` and predicting `E_z`. `train_cavity_baseline`
(`src/em_piml/train.py`) minimizes four loss terms via Adam: the PDE
residual (via double autograd), the two boundary conditions, the
initial field `E(x,0)`, and `dE/dt(x,0) = 0` (true for this
standing-wave mode). Defaults (4000 steps, lr=3e-3, small
collocation/boundary/initial batches) were tuned empirically — checked
convergent and stable across training seeds 0/1/2/7 (relative L2 error
0.026-0.046 against the analytical solution), ~35s on CPU. Don't
increase `steps` or network size without re-checking CI runtime stays
well under a minute.

Reproduce: `uv run python3 -m em_piml.train` (prints relative L2 error
over 500 random held-out points).

## Verification

`tests/test_baseline_cavity.py` trains with a fixed seed (0) and
asserts relative L2 error against the analytical solution, evaluated at
500 random `(x, t)` points sampled with a *different* seed (123) than
training — genuinely held-out, not memorized collocation points.
Tolerance is `0.1`, chosen with ~2-4x margin above the empirically
observed 0.026-0.046 range so the test doesn't flake on ordinary
run-to-run variance.

Pitfall already hit once: don't evaluate relative L2 error at a single
`(x, t)` snapshot — if it lands on a zero-crossing of the analytical
solution (e.g. `t = PERIOD/4`, where `cos(omega*t) = 0`), the near-zero
denominator inflates the metric to nonsense. Always evaluate over many
points spanning the domain.

## Data

Ground-truth grid is registered via `mx-data`, not computed ad hoc:
`uv run mx-data fetch em-piml-1d-cavity-analytical`. The generator
(`tools/datasets/registry/generators/em_piml_1d_cavity_analytical.py`)
imports `em_piml.physics.analytical_field` rather than duplicating the
formula — it works because `uv sync --all-packages` installs every
workspace member (including this one) into the one shared venv.

## Known deferred items

- `torch` is installed from plain PyPI (bundles CUDA deps, larger than
  necessary for a CPU-only baseline). The dedicated CPU-only wheel index
  (`https://download.pytorch.org/whl/cpu`, via `[tool.uv.sources]` +
  `[[tool.uv.index]]`) is the right fix but couldn't be verified from
  this session's sandboxed network — untested config wasn't worth
  shipping blind. Revisit if CI install time/size becomes a problem.
- Tokenization/embedding experimentation is explicitly out of scope
  here — next issue's job, once this baseline is the comparison point.

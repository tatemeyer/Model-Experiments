# Does collocation-set density explain the rest? (issue #8)

Swept the fixed collocation/boundary/initial point-set size for
`train_fourier_cavity_lbfgs` on the same `num_bands=4` configuration —
architecture, optimizer, iteration budget all held fixed, density is
the only variable:

| n_collocation | seed 0 | seed 1 | seed 2 | seed 7 |
|---|---|---|---|---|
| 200 (original) | 0.822 | 0.851 | - | - |
| 1000 | 0.090 | 0.143 | - | - |
| 2000 | 0.098 | 0.104 | 0.096 | 0.065 |
| 3000 | 0.142 | 0.163 | - | - |
| 4000 | 0.055 | 0.129 | - | - |

**Density was most of the answer** — an order-of-magnitude improvement
over the 200-point plateau at every density tried from 1000 up. But the
relationship is **noisy, not monotonic**: 3000 points did *worse* than
2000, and 4000 was a mix of the best (0.055) and a middling (0.129)
result. This is a single fixed point sample per (seed, density) — which
specific points get drawn matters as much as the nominal count, at
least in this 1000-4000 range.

**Shipped default: `n_collocation=2000, n_boundary=400, n_initial=400`**
— chosen because it's the most thoroughly tested (4 seeds) and most
consistent (0.065-0.104, all within roughly 2x of each other, unlike
the wider spread at other densities). `tests/test_fourier_lbfgs.py`
asserts `< 0.15` (comfortable margin above the observed 0.104 worst
case) — tighter than issue #6's `0.95` partial-improvement bound by
over 6x, but still not the standard `0.1` baseline bar, because seed 1
landed at 0.104 — just over it. Don't tighten this further without
re-sweeping; the noise between seeds/densities above means a single
lucky seed isn't good evidence of a real margin.

**Leads at the time**, in rough order of how cheap they are to test:
1. ~~Collocation-set density~~ — largely resolved by this issue.
2. ~~Network capacity~~ — resolved, see `010-network-capacity.md`.
3. SOAP/SS-Broyden (the paper's actual best performers) weren't tried —
   out of scope here since they require adopting a new
   package/implementation, not a `torch.optim` drop-in like L-BFGS.
4. ~~Understand *why* density vs. accuracy is non-monotonic~~ — see
   `012-point-draw-variance.md`: it's the point draw, not the count.

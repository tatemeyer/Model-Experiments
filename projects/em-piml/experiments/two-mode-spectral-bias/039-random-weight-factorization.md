# Does Random Weight Factorization close the two-mode spectral-bias gap? (issue #39)

Every fix tried against the two-mode spectral-bias target so far
(`022-two-mode-superposition.md`'s Fourier embedding, `025-num-bands-sweep.md`'s
higher `num_bands`) changed the *input embedding*. Wang, Wang, Seidman,
Perdikaris, "Random Weight Factorization Improves the Training of
Continuous Neural Representations" (arXiv:2210.01274) instead
reparameterizes the network's *linear layer weights themselves*: each
weight matrix `W` becomes `diag(s) @ V`, a per-output-neuron scalar
factor `s` and a same-shaped matrix `V`, both trained directly instead
of `W`. This is a mechanism the paper claims is independent of, and
complementary to, the input embedding — this issue tests whether it
closes the gap alone and combined with the existing Fourier embeddings.

`RWFLinear` (`src/em_piml/model.py`) is a drop-in replacement for
`nn.Linear`: it draws a standard `nn.Linear` init, then factorizes it
row-wise so the initial effective weight `diag(s) @ v` exactly equals
that init (only training dynamics differ, not the first forward pass).
`s`'s init distribution is `exp(mu + sigma * z)`, `z ~ N(0,1)` per the
paper's own formulation (arXiv:2210.01274, sec. 3.1) — this project's
wave-equation target isn't among the paper's own benchmarks (only
advection, Navier-Stokes, and diffusion PDEs are), so `mu=0.5, sigma=0.1`
(the paper's own Navier-Stokes setting, Table 4/Appendix D — the
nearest PDE task among its ablations) was used rather than its stated
general-purpose default (`mu=1.0, sigma=0.1`); this is an untuned,
borrowed choice for this project's specific problem, not a value the
paper validated here. `RWFCavityPINN`/`RWFFourierCavityPINN` apply it to
every linear layer of the existing `CavityPINN`/`FourierCavityPINN`
body. `train_cavity_rwf_two_mode`/`train_fourier_cavity_rwf_two_mode`/
`train_fourier_cavity_rwf_lbfgs_two_mode` (`src/em_piml/train.py`) reuse
the exact shipped recipes of the two-mode functions they replace
(`train_cavity_two_mode`, `train_fourier_cavity_two_mode`,
`train_fourier_cavity_lbfgs_two_mode`) — weight parameterization is the
only variable in each comparison.

**Result: a real but small effect in two of three configurations, and
none comes close to closing the gap.**

| variant | relative L2 (seeds 0/1/2/7) | vs. non-RWF baseline |
|---|---|---|
| RWF alone (no Fourier) | 0.7471, 0.7513, 0.7547, 0.7688 | 0.7699-0.7947 (`022-...md`) — small improvement |
| RWF + `num_bands=2` (Adam) | 0.7018, 0.7084, 0.7104, 0.7190 | 0.6995-0.7063 (`022-...md`) — slightly *worse* |
| RWF + `num_bands=4` (L-BFGS) | 0.6991, 0.6994, 0.7012, 0.7028 | 0.7023-0.7128 (`025-...md`) — small improvement, much lower variance (stdev 0.0015 vs. the wider spread implied by the baseline's range) |

RWF alone and RWF + `num_bands=4` both give a real, if modest,
improvement over their respective non-RWF baselines — the `num_bands=4`
combination in particular is both slightly more accurate on average and
markedly more seed-stable. RWF + `num_bands=2` is the one configuration
where it doesn't help: its range (0.7018-0.7190) mostly sits *above*
plain `num_bands=2`'s range (0.6995-0.7063), i.e. RWF made this specific
combination a little worse, not better. None of the three gets anywhere
near the 0.026-0.046 achievable on the single-mode target — this is not
a fix for the underlying spectral-bias gap, just a small, mixed
perturbation on top of it.

**A pointwise check (same method as `022-...md`) confirms all three
variants are still blind to the `n=8` mode, same mechanism as every
prior fix.** Evaluating each (seed 0) at `x=0.5625, t=0` — an `n=8`
peak, true field `0.9904`, `n=1`-only envelope value `0.4904`:

| variant | prediction |
|---|---|
| RWF alone | 0.4565 |
| RWF + `num_bands=2` | 0.5003 |
| RWF + `num_bands=4` | 0.4774 |

All three predictions cluster tightly around the `n=1`-only envelope
value (0.4904), essentially identical in shape to issue #22's plain
(0.4410) and Fourier (0.4687) predictions at the same point — RWF
changes *how the weights are parameterized during optimization*, not
*what frequencies the network's input/architecture can represent*.
Since the representational gap here is the Fourier embedding's missing
`8*pi` basis frequency at `num_bands<4` (established in `022-...md`),
a weight-reparameterization mechanism with no effect on that basis was
never mechanistically positioned to close it — the small, mixed
accuracy deltas above are consistent with RWF doing what it's actually
claimed to do (loss-landscape conditioning / per-neuron adaptive
step size) rather than adding missing frequency content.

`tests/test_rwf_two_mode.py` asserts all three variants' relative L2
error stays `> 0.5`, matching `test_two_mode_superposition.py`'s
`FAILURE_LOWER_BOUND` convention — a regression check on this
documented failure signature, not an accuracy bar.

**Determinism**: verified before trusting any number above — same seed
(0) reproduces a bit-identical `state_dict` across two separate calls to
`train_cavity_rwf_two_mode` (and by extension every `RWFLinear`-based
variant, since they share the same `torch.manual_seed`-before-construction
pattern as every other `train_*` function in this file).

**Leads for whoever picks this up next:**
1. `mu`/`sigma` are untuned for this project's problem (borrowed from
   the paper's Navier-Stokes setting, the nearest available PDE
   ablation) — a small sweep over `mu`/`sigma` might shift RWF alone's
   or `num_bands=4`'s numbers further, though it's very unlikely to
   close the gap outright given the pointwise diagnosis above (RWF
   doesn't touch the missing-frequency mechanism at all).
2. RWF + `num_bands=4` combined with SOAP (rather than L-BFGS) is
   untried — issue #25 documented both L-BFGS and SOAP for plain
   `num_bands=4`; only L-BFGS was tested here.
3. Given the pointwise diagnosis, the two-mode-spectral-bias thread's
   real remaining lever is still representational (issue #41's
   PirateNets-style adaptive-residual architecture, still untried) —
   RWF is now the second mechanism (after `num_bands`, issue #25) shown
   to leave the missing-basis-frequency problem essentially untouched.

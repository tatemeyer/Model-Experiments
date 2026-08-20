# Does `--device cuda` actually place tensors on the GPU, and is the GPU worth using here? (issue #60)

Slice 3 of the `device-abstraction` Arc. Slice 1 (#57) built the device resolver and Slice 2 (#59)
threaded `device=` through the `train_*`/`evaluate_*` entry points, but neither could confirm that
an accelerator is ever actually used: every existing device test monkeypatches
`torch.cuda.is_available()`, so it passes identically on a machine with no GPU. This Slice closes
that gap on real hardware, and answers the question that follows immediately after — whether
running on the GPU is *better*.

No new paper is cited, so `../LITERATURE.md` is unchanged.

## The blocker was packaging, not hardware

The machine has a GTX 1660 Ti and a working driver (591.86, CUDA 13.1), yet
`torch.cuda.is_available()` returned `False`. `uv.lock` resolved `torch` from plain PyPI, and the
PyPI **Windows** wheel ships no CUDA runtime at all — every one of its CUDA dependencies
(`cuda-toolkit`, `nvidia-cudnn-cu13`, `triton`, …) carries a `sys_platform == 'linux'` marker. The
GPU was unusable for packaging reasons alone.

Fix, in the root `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
```

Three properties this deliberately has:

- **`explicit = true`** — the index is consulted only for packages that name it, never as a
  general fallback. Same reasoning that makes this repo SHA-pin third-party Actions: an unpinned
  second package index is a supply-chain surface across the whole workspace.
- **`marker = "sys_platform == 'win32'"`** — without it, CI's `ubuntu-latest` runner would be
  re-sourced too. Verified on the `uv.lock` diff rather than assumed: the *only* lines removed
  from the PyPI `torch` block are its four `win_amd64` wheels. Every Linux/macOS wheel URL, hash,
  and dependency line is byte-identical, so CI resolves exactly what it resolved before.
- **cu126, not a newer variant** — it is the CUDA build that ships a `cp313` Windows wheel for the
  torch version already locked (2.13.0), so this changes the *build variant* without changing the
  version. The driver's CUDA 13.1 runs 12.6-built binaries, and Turing/`sm_75` is supported.

Result: `torch 2.13.0+cu126`, `torch.cuda.is_available() == True`, matmul executing on `cuda:0`.

**Supply-chain delta is one wheel.** On Windows the CUDA runtime ships *inside* the torch wheel
(`cudart64_12.dll`, `cublas64_12.dll`, `cudnn64_9.dll`, …) rather than as separate `nvidia-*`
packages, so the lock gained exactly one `[[package]]` entry and no new transitive dependencies.
On Linux it gained nothing. License files ship in the wheel, not just metadata
(`torch-2.13.0+cu126.dist-info/licenses/`: a top-level `LICENSE` plus 35 `third_party/` entries;
NVIDIA-authored source components such as cutlass and cudnn_frontend are BSD-3-Clause). The
bundled CUDA/cuDNN runtime binaries are redistributables under NVIDIA's CUDA EULA / cuDNN SLA.

## Result: placement works — and the GPU is *slower* for this project's workloads

| variant | dtype | optimizer | CPU | GPU | speedup |
|---|---|---|---|---|---|
| `train_cavity_long_horizon` (4000 steps, 5 periods) | FP32 | Adam | **27.55s** | 61.13s | **0.45×** |
| `train_fourier_cavity_lbfgs_fp64` (50×50) | FP64 | L-BFGS | 102.59s | **99.46s** | 1.03× |

CUDA context init: 0.214s, paid once per process, measured and reported separately so a fixed
startup charge isn't misattributed to per-step throughput. All GPU timings are taken after
`torch.cuda.synchronize()` — without it the timer would stop when work was *submitted*, not
finished, and the GPU would look arbitrarily fast.

Both devices genuinely ran: relative L2 error is 0.9255 (CPU) vs 0.9359 (GPU) for the FP32
long-horizon variant — both sitting on this project's documented ~0.93 long-horizon collapse
plateau — and 0.0254 vs 0.0394 for FP64. The numbers differ rather than match because CUDA and
CPU RNG streams differ for the same seed; `train_fourier_cavity_lbfgs` already documents that the
`points_seed` reproducibility guarantee holds *within* a device, not across devices. That is a
property to design experiments around, not a bug: **a CPU result and a GPU result are not
seed-comparable**, so a variant swept across devices would confound device with RNG stream.

**Why the GPU loses, and why FP64 doesn't lose worse.** The obvious prediction for a Turing
consumer card — no tensor cores, FP64 at ~1/32 of FP32 — is that FP64 should be catastrophically
worse. It isn't; it's a wash. Both facts have the same cause: these models are *tiny*
(`hidden=32`, 3 layers, 200 collocation points for the long-horizon variant). The workload is
**kernel-launch-bound, not FLOP-bound**. Each Adam step issues many small kernels that the GPU
finishes almost instantly and then waits for the next launch, so wall-clock is dominated by
per-launch overhead that the CPU simply doesn't pay. The FP32 run, with 4000 such steps, pays that
overhead most often and loses worst (0.45×). The FP64 L-BFGS run does far more arithmetic per
launch inside each line-search iteration, which is enough to hide the launch overhead — and
because the GPU is never FLOP-saturated, the 1/32 FP64 throughput penalty never becomes the
binding constraint either. The two effects roughly cancel to 1.03×.

`tests/test_device_placement_gpu.py` locks in the placement half as this repo's first
`@pytest.mark.gpu` tests (the marker was registered in Slice 1 and had no users until now): that a
resolved CUDA device puts tensors on the GPU *and* runs a real kernel on them, that a model
trained through `train_cavity_long_horizon(device="cuda")` has all parameters on CUDA, and that
GPU training/eval yields a finite error rather than silent NaNs. They are excluded from the
default run and from CI (`ubuntu-latest` has no GPU) and run explicitly with
`uv run pytest -m gpu`. The timing comparison itself is not a test — it lives in
`src/em_piml/device_timing.py`, run offline.

**Leads for whoever picks this up next:**

1. **Don't reach for the GPU on this project's current problem sizes.** Both measured variants are
   launch-bound; the CPU is the faster device and the simpler one (no cross-device RNG caveat).
   The GPU becomes worth revisiting when a problem here gets big enough per step to saturate it —
   a much wider network, a far denser collocation set, or batched multi-seed training sharing one
   kernel launch. That last one is the most promising shape: this project runs many small seeds,
   and batching seeds into single kernels is exactly the transformation that converts a
   launch-bound workload into a FLOP-bound one.
2. **The 1/32-FP64 concern was the wrong thing to worry about at this scale.** Worth re-testing
   only once a workload here is actually FLOP-bound; until then FP64 costs essentially nothing on
   this GPU relative to FP32, which is the opposite of the naive expectation.
3. The standing deferred item about installing torch from the **CPU-only** wheel index (to shrink
   install size) is **still open** — see the Known deferred items note. Issue #60's success
   criteria asked for it in the same change, but that contradicts its own stronger requirement
   that CI's Linux resolution be unchanged, so it was deliberately not done here.

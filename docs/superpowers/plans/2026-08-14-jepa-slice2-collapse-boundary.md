# jepa Arc 1 Slice 2: Collapse Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map where representation collapse appears on the EMA-momentum axis and how that boundary moves with training length.

**Architecture:** Add a checkpoint-evaluation hook to `train_jepa` so one 6000-step run yields metrics at four step counts; promote Slice 1's evaluation harness out of its test file so both the test and a new offline sweep script share it; run a 6-momentum × 3-seed sweep plus controls; lock the finding with a regression test and write it up.

**Tech Stack:** Python 3.12+, `uv` workspace, PyTorch (CPU-only), pytest, ruff.

## Global Constraints

- Repository is `D:\Dev\Projects\Model-Experiments`. Work on a feature branch off `main`.
- Branch naming must match the `feature-branches` ruleset prefixes: use `feat/jepa-momentum-collapse-boundary`.
- ruff: `line-length = 100`, `select = ["E", "F", "I", "UP"]`. Run `uvx ruff check .` before every commit.
- CPU-only. No new runtime dependencies — `torch` and `numpy` only.
- Sweep scripts set `torch.set_num_threads(1)` — measured faster here, not merely conventional (23.9s vs 39.1s for 300 steps on 12 cores).
- Any test that trains a model gets `@pytest.mark.slow`. The default suite deselects `slow` and `gpu`.
- Determinism is a standing project rule: same seed in → bit-identical result out, verified before any number is trusted.
- **Primary metric is `effective_rank`.** `embedding_std` is recorded for continuity only. **No linear-probe R²** — Slice 1 showed random-init sometimes scores highest, so it cannot support a conclusion here.
- Momentum grid: `(0.0, 0.9, 0.99, 0.996, 0.999, 0.9999)`. Seeds: `(0, 1, 2)`. Checkpoints: `(300, 1000, 3000, 6000)`.
- Commit style matches existing history: `jepa: description`.
- Two plotly tests hang on Windows (kaleido headless Chromium). When running the full suite locally, add:
  `--deselect tools/viz/tests/test_plotly_fields.py::test_export_png_writes_nonempty_file --deselect tools/viz/tests/test_plotly_fields.py::test_render_orbit_gif_writes_nonempty_file`

---

### Task 1: File the Intent issue

`projects/jepa` runs on the plain Intent-Issue → PR loop, so the work needs an issue before it starts.

**Files:** none.

**Interfaces:**
- Consumes: the spec at `docs/superpowers/specs/2026-08-14-jepa-slice2-collapse-boundary-design.md`.
- Produces: an issue number, referenced by the final PR's `Closes #N`.

- [ ] **Step 1: Create the issue**

```bash
cd /d/Dev/Projects/Model-Experiments
gh issue create \
  --title "[Intent]: jepa — EMA-momentum collapse boundary (Arc 1 Slice 2)" \
  --label "intent,autonomy:review,project:jepa" \
  --body "## Desired outcome

Slice 1 (issue #69) established that EMA prevents collapse at one operating
point: \`ema_momentum = 0.996\`, 3000 steps. This slice maps the surface —
where is the collapse boundary on the momentum axis, and how does it move with
training length?

Design: \`docs/superpowers/specs/2026-08-14-jepa-slice2-collapse-boundary-design.md\`.

Momentum \`(0.0, 0.9, 0.99, 0.996, 0.999, 0.9999)\` x seeds \`(0, 1, 2)\`,
each trained once to 6000 steps with metrics recorded at 300/1000/3000/6000,
plus the \`use_ema=False\` and random-init controls.

## Success criteria / verification

- A checkpointed run is bit-identical to an unhooked run at the same seed.
- \`projects/jepa/results.csv\` carries every momentum/seed/checkpoint/metric datapoint.
- \`experiments/002-momentum-collapse-boundary.md\` states where the boundary lies, how it moves with training length, and explicitly addresses the high-momentum confound (random-init effective_rank 2.44-2.93 is *higher* than the trained model's 2.35-2.79, so rank alone cannot distinguish 'learned well' from 'never moved' — loss-curve evidence must settle it).
- A regression test locks the finding and runs in CI.
- \`uv run pytest -q\` and \`uvx ruff check .\` clean.

## Constraints

CPU-only, no new dependencies. The sweep is ~3.1 hours offline and stays out
of the default test suite (precedent: \`point_draw_sweep.py\`).

## Autonomy level

autonomy:review"
```

- [ ] **Step 2: Record the issue number**

Note the number it prints. Every later commit and the final PR reference it.

- [ ] **Step 3: Create the branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/jepa-momentum-collapse-boundary
```

---

### Task 2: Checkpoint-evaluation hook, with determinism proof

The spec's central technical risk. **Outcome, recorded 2026-08-14: the assumption was wrong, and the gate caught it.**

The spec assumed a callback could not perturb training because batches and masks come from explicit generators. In fact `Predictor`'s `nn.TransformerEncoderLayer` stack carries **six `Dropout` modules at the default `p=0.1`**, and dropout draws from the **global** torch RNG every step — so a callback touching the global RNG changes the loss curve immediately. Verified: a no-op callback is bit-identical; a reseeding callback is not.

Rather than fall back to retraining per step count (4× cost), the implementation below saves and restores the global RNG state around the callback, making any callback safe. Steps 3 and 4 reflect that fix.

**Files:**
- Modify: `projects/jepa/src/jepa/train.py`
- Test: `projects/jepa/tests/test_train.py`

**Interfaces:**
- Consumes: existing `train_jepa`.
- Produces:
  `train_jepa(steps=300, seed=0, lr=1e-3, batch_size=32, pool_size=512, ema_momentum=0.996, history=None, use_ema=True, checkpoint_steps=None, on_checkpoint=None)` where
  `checkpoint_steps: Sequence[int] | None` and
  `on_checkpoint: Callable[[int, PatchEncoder], None] | None`.
  The callback fires **after** the optimizer step and EMA update for that step, receiving the 1-based step number and the live online encoder.

- [ ] **Step 1: Write the failing determinism test**

Append to `projects/jepa/tests/test_train.py`:

```python
@pytest.mark.slow
def test_checkpoint_hook_does_not_perturb_training():
    """The sweep in jepa.momentum_steps_sweep evaluates mid-training. That is only sound if the
    callback cannot change the trajectory. train_jepa seeds torch globally once before model
    construction, then draws batches from an explicit torch.Generator and masks from a numpy
    Generator -- never the global RNG -- so a callback that scribbles all over the global RNG
    should still leave training bit-identical. This asserts exactly that, with a deliberately
    hostile callback."""
    seen: list[int] = []

    def hostile(step: int, encoder) -> None:
        seen.append(step)
        torch.manual_seed(999)  # deliberately corrupt the global RNG
        torch.rand(64)

    hooked: list[float] = []
    train_jepa(
        steps=60,
        seed=0,
        history=hooked,
        checkpoint_steps=(20, 40, 60),
        on_checkpoint=hostile,
    )

    plain: list[float] = []
    train_jepa(steps=60, seed=0, history=plain)

    assert seen == [20, 40, 60]
    assert hooked == plain
```

Ensure `test_train.py` imports `pytest` and `torch` at the top; add them if absent.

- [ ] **Step 2: Run it to verify it fails**

```bash
cd /d/Dev/Projects/Model-Experiments
uv run pytest projects/jepa/tests/test_train.py::test_checkpoint_hook_does_not_perturb_training -o addopts="" -v
```

Expected: FAIL — `train_jepa() got an unexpected keyword argument 'checkpoint_steps'`.

- [ ] **Step 3: Implement the hook**

In `projects/jepa/src/jepa/train.py`, add to the imports at the top:

```python
from collections.abc import Callable, Sequence
```

Add these two parameters to `train_jepa`'s signature, after `use_ema: bool = True`:

```python
    checkpoint_steps: Sequence[int] | None = None,
    on_checkpoint: Callable[[int, PatchEncoder], None] | None = None,
```

Immediately before the training loop, add:

```python
    checkpoint_set = set(checkpoint_steps or ())
```

Change the loop header from `for _ in range(steps):` to:

```python
    for step in range(steps):
```

At the very end of the loop body, after the `if history is not None:` block, add:

```python
        if on_checkpoint is not None and (step + 1) in checkpoint_set:
            # Fired after the optimizer step and EMA update, so the encoder reflects a completed
            # step.
            #
            # The global torch RNG is saved and restored around the callback because training
            # *does* consume it: Predictor's nn.TransformerEncoderLayer stack carries six Dropout
            # modules at the PyTorch default p=0.1, and dropout samples from the global RNG every
            # step. (Batches and masks come from the explicit batch_rng/mask_rng generators, which
            # is what makes this the *only* global-RNG dependency -- but it is enough.) Without
            # this guard a callback that touches the global RNG silently changes the trajectory it
            # is meant to observe, which would corrupt every downstream metric.
            # Asserted by test_checkpoint_hook_does_not_perturb_training.
            rng_state = torch.get_rng_state()
            try:
                on_checkpoint(step + 1, encoder)
            finally:
                torch.set_rng_state(rng_state)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest projects/jepa/tests/test_train.py::test_checkpoint_hook_does_not_perturb_training -o addopts="" -v
```

Expected: PASS.

**If it FAILS, stop and report.** Do not weaken the test to make it pass — its whole purpose is to catch a silent trajectory change that would corrupt every downstream metric. When this gate first ran it *did* fail (see the outcome note at the top of this task); the correct response was to find the real global-RNG consumer and guard against it, not to relax the assertion.

- [ ] **Step 5: Confirm no regression in the existing jepa tests**

```bash
uv run pytest projects/jepa -o addopts="" -q
```

Expected: all pass, including Slice 1's three slow tests.

- [ ] **Step 6: Lint and commit**

```bash
uvx ruff check .
git add projects/jepa/src/jepa/train.py projects/jepa/tests/test_train.py
git commit -m "jepa: add checkpoint-evaluation hook to train_jepa (Arc 1 Slice 2, issue #N)

Lets one long run yield metrics at several step counts instead of retraining
per cell. Includes a determinism test with a callback that deliberately
corrupts the global torch RNG, proving training is unaffected."
```

Replace `#N` with Task 1's issue number.

---

### Task 3: Promote the evaluation harness into `src/`

Slice 1's harness (`collapse_metrics`, `build_encoder`, the pooling helpers) lives inside `tests/test_baseline_collapse_avoidance.py`, so a sweep script in `src/` cannot reuse it. This is a **pure refactor** — no behaviour changes, no number changes.

**Files:**
- Create: `projects/jepa/src/jepa/harness.py`
- Modify: `projects/jepa/tests/test_baseline_collapse_avoidance.py`

**Interfaces:**
- Consumes: `jepa.eval.effective_rank`, `jepa.eval.embedding_std`, `jepa.train.train_jepa`, `jepa.models.PatchEncoder`, `jepa.bouncing_ball.generate_dataset`.
- Produces, in `jepa.harness`:
  - `PROBE_TRAIN_SEED_OFFSET: int = 10_000`, `PROBE_TEST_SEED_OFFSET: int = 20_000`
  - `probe_frames_and_targets(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]`
  - `per_patch_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor`
  - `flattened_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor`
  - `collapse_metrics(encoder: PatchEncoder, seed: int, n_test: int = 300) -> dict[str, float]`
  - `probe_r2(encoder: PatchEncoder, seed: int, n_train: int = 4000, n_test: int = 300) -> float`
  - `build_encoder(variant: str, seed: int, steps: int = 3000) -> PatchEncoder`

- [ ] **Step 1: Create `harness.py`**

Create `projects/jepa/src/jepa/harness.py` with this exact content:

```python
"""Shared evaluation harness for this project's experiments: frame/target generation, the two
pooling strategies Slice 1 settled on, collapse metrics, the linear probe, and encoder
construction per variant.

Promoted verbatim out of tests/test_baseline_collapse_avoidance.py (Arc 1 Slice 1, issue #69) so
that offline sweep scripts under src/ can reuse it rather than duplicating it. Behaviour is
unchanged -- the numbers in experiments/001-baseline-collapse-avoidance.md still reproduce.
"""

from __future__ import annotations

import numpy as np
import torch

from jepa.bouncing_ball import CANVAS_SIZE, generate_dataset
from jepa.eval import effective_rank, embedding_std, linear_probe_r2
from jepa.models import PatchEncoder
from jepa.train import EMBED_DIM, HIDDEN_DIM, PATCH_SIZE, train_jepa

# Held-out probe splits use master seeds offset well clear of any training-pool seed, so a probe
# frame is never one the encoder was trained on.
PROBE_TRAIN_SEED_OFFSET = 10_000
PROBE_TEST_SEED_OFFSET = 20_000


def probe_frames_and_targets(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n single-frame samples normalized to [0, 1], plus their [x, y, vx, vy] ground truth."""
    dataset = generate_dataset(n_sequences=n, n_frames=1, master_seed=seed)
    frames = torch.from_numpy(dataset["frames"][:, 0]).float().unsqueeze(1) / 255.0
    positions = dataset["positions"][:, 0]
    velocities = dataset["velocities"][:, 0]
    targets = torch.from_numpy(np.concatenate([positions, velocities], axis=-1)).float()
    return frames, targets


def per_patch_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Every (frame, patch) pair as its own sample, (N * num_patches, embed_dim) -- the direct
    "do patch representations vary meaningfully" collapse question, not diluted by averaging
    content-bearing patches against the ~55/64 pure-background patches this task's frames have."""
    with torch.no_grad():
        tokens = encoder(frames)
    return tokens.reshape(-1, tokens.shape[-1])


def flattened_embeddings(encoder: PatchEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Every patch embedding concatenated per frame, (N, num_patches * embed_dim) -- preserves
    spatial position (unlike mean-pooling) for the linear probe."""
    with torch.no_grad():
        tokens = encoder(frames)
    return tokens.reshape(tokens.shape[0], -1)


def collapse_metrics(encoder: PatchEncoder, seed: int, n_test: int = 300) -> dict[str, float]:
    """embedding_std and effective_rank over per-patch embeddings of a held-out frame set."""
    frames, _ = probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    embeddings = per_patch_embeddings(encoder, frames)
    return {
        "embedding_std": embedding_std(embeddings),
        "effective_rank": effective_rank(embeddings),
    }


def probe_r2(encoder: PatchEncoder, seed: int, n_train: int = 4000, n_test: int = 300) -> float:
    """Held-out linear-probe R^2 for position+velocity. Retained for Slice 1's regression test;
    Slice 2 deliberately does not use it (see that slice's design -- random-init sometimes scores
    highest, so it cannot support a conclusion at this scale)."""
    train_frames, train_targets = probe_frames_and_targets(n_train, seed + PROBE_TRAIN_SEED_OFFSET)
    test_frames, test_targets = probe_frames_and_targets(n_test, seed + PROBE_TEST_SEED_OFFSET)
    train_embeddings = flattened_embeddings(encoder, train_frames)
    test_embeddings = flattened_embeddings(encoder, test_frames)
    return linear_probe_r2(train_embeddings, train_targets, test_embeddings, test_targets)


def build_encoder(variant: str, seed: int, steps: int = 3000) -> PatchEncoder:
    """variant: "full" (EMA target), "no_ema" (gradient-trained target), or "random_init"
    (untrained encoder, no train_jepa call at all)."""
    if variant == "full":
        encoder, _, _ = train_jepa(steps=steps, seed=seed, use_ema=True)
    elif variant == "no_ema":
        encoder, _, _ = train_jepa(steps=steps, seed=seed, use_ema=False)
    elif variant == "random_init":
        torch.manual_seed(seed)
        encoder = PatchEncoder(
            image_size=CANVAS_SIZE,
            patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return encoder
```

- [ ] **Step 2: Point Slice 1's test at the new module**

In `projects/jepa/tests/test_baseline_collapse_avoidance.py`, **delete** these now-duplicated definitions: `PROBE_TRAIN_SEED_OFFSET`, `PROBE_TEST_SEED_OFFSET`, `_probe_frames_and_targets`, `_per_patch_embeddings`, `_flattened_embeddings`, `collapse_metrics`, `probe_r2`, `build_encoder`.

Then replace the import block

```python
import numpy as np
import pytest
import torch
from jepa.bouncing_ball import CANVAS_SIZE, generate_dataset
from jepa.eval import effective_rank, embedding_std, linear_probe_r2
from jepa.models import PatchEncoder
from jepa.train import EMBED_DIM, HIDDEN_DIM, PATCH_SIZE, train_jepa
```

with

```python
import pytest
from jepa.harness import build_encoder, collapse_metrics, probe_r2
from jepa.train import train_jepa
```

Keep `COLLAPSE_STEPS`, `COLLAPSE_RANK_THRESHOLD`, and `PROBE_R2_FLOOR` in the test file — they are Slice 1's thresholds, not shared harness.

`build_encoder`'s default `steps=3000` already equals `COLLAPSE_STEPS`, so behaviour is unchanged either way — but pass it **explicitly** at the three call sites so the value stays visibly bound to the thresholds it was chosen against, rather than silently depending on a default in another module:

```python
        full_encoder = build_encoder("full", seed, steps=COLLAPSE_STEPS)
        no_ema_encoder = build_encoder("no_ema", seed, steps=COLLAPSE_STEPS)
        random_encoder = build_encoder("random_init", seed, steps=COLLAPSE_STEPS)
```

- [ ] **Step 3: Verify the refactor changed nothing**

```bash
uv run pytest projects/jepa -o addopts="" -q
```

Expected: every test passes, including all three Slice 1 slow tests. **This is the acceptance criterion for the refactor** — if any threshold assertion now fails, the move was not behaviour-preserving; revert and redo.

- [ ] **Step 4: Lint and commit**

```bash
uvx ruff check .
git add projects/jepa/src/jepa/harness.py projects/jepa/tests/test_baseline_collapse_avoidance.py
git commit -m "jepa: promote the evaluation harness from the Slice 1 test into src/jepa/harness.py

Pure refactor so offline sweep scripts can reuse it. Slice 1's thresholds stay
in its own test; all its assertions still pass unchanged."
```

---

### Task 4: The sweep script

**Files:**
- Create: `projects/jepa/src/jepa/momentum_steps_sweep.py`

**Interfaces:**
- Consumes: `train_jepa(..., checkpoint_steps=, on_checkpoint=)` (Task 2), `jepa.harness.collapse_metrics` and `build_encoder` (Task 3).
- Produces: an executable module printing one CSV-shaped line per datapoint, plus a human-readable summary.

- [ ] **Step 1: Create the script**

Create `projects/jepa/src/jepa/momentum_steps_sweep.py`:

```python
"""Arc 1 Slice 2: where does representation collapse appear on the EMA-momentum axis, and how
does that boundary move with training length?

Slice 1 (issue #69) measured one point -- momentum 0.996, 3000 steps. It also found the gap is
invisible at 300 steps, so momentum and training length cannot be studied independently.

Offline exploratory script, deliberately out of the default test suite (precedent:
em-piml's point_draw_sweep.py). ~3.1 hours: 21 runs at ~8.8 min each.

Run: uv run python -m jepa.momentum_steps_sweep
"""

from __future__ import annotations

import statistics

import torch

from jepa.harness import build_encoder, collapse_metrics
from jepa.train import train_jepa

# Single-threaded is measurably faster here, not just conventional: 23.9s vs 39.1s for 300 steps
# on 12 cores (torch's intra-op threading loses to overhead at this model size).
torch.set_num_threads(1)

# Spans both degenerate ends deliberately. At 0.0 the target is the online encoder copied every
# step (stop-gradient, zero smoothing) -- distinct from use_ema=False, where the target is a
# separate network trained by gradient descent. At 0.9999 the target barely moves from its random
# initialization. 0.996 is Slice 1's operating point, included for direct comparability.
MOMENTA = (0.0, 0.9, 0.99, 0.996, 0.999, 0.9999)
SEEDS = (0, 1, 2)
CHECKPOINTS = (300, 1000, 3000, 6000)
MAX_STEPS = max(CHECKPOINTS)

# Fraction of history used for the loss-slope statistic. The high-momentum confound this exists
# for: Slice 1 measured random-init effective_rank at 2.44-2.93, ABOVE the trained model's
# 2.35-2.79, so a healthy-looking rank at momentum 0.9999 could equally mean "never moved".
# Rank alone cannot tell those apart; the loss trend can.
SLOPE_WINDOW = 0.1


def _loss_slope(history: list[float], upto: int) -> float:
    """Mean loss over the last SLOPE_WINDOW of the first `upto` steps minus the mean over the
    window immediately before it. Negative means the loss is still falling."""
    window = max(1, int(upto * SLOPE_WINDOW))
    recent = history[upto - window : upto]
    earlier = history[max(0, upto - 2 * window) : upto - window]
    if not earlier:
        return 0.0
    return statistics.mean(recent) - statistics.mean(earlier)


def _run(variant: str, momentum: float | None, seed: int, rows: list[dict]) -> None:
    """One training run to MAX_STEPS, recording metrics at every checkpoint."""
    history: list[float] = []

    def record(step: int, encoder) -> None:
        metrics = collapse_metrics(encoder, seed)
        row = {
            "variant": variant,
            "momentum": momentum,
            "seed": seed,
            "steps": step,
            "effective_rank": metrics["effective_rank"],
            "embedding_std": metrics["embedding_std"],
            "final_loss": history[step - 1],
            "loss_slope": _loss_slope(history, step),
        }
        rows.append(row)
        print(
            f"  {variant:>9} m={momentum if momentum is not None else 'n/a':>7} seed={seed} "
            f"steps={step:>4}: rank={row['effective_rank']:.3f} "
            f"std={row['embedding_std']:.4f} loss={row['final_loss']:.5f} "
            f"slope={row['loss_slope']:+.6f}",
            flush=True,
        )

    train_jepa(
        steps=MAX_STEPS,
        seed=seed,
        history=history,
        use_ema=(variant == "full"),
        ema_momentum=momentum if momentum is not None else 0.996,
        checkpoint_steps=CHECKPOINTS,
        on_checkpoint=record,
    )


def main() -> None:
    rows: list[dict] = []

    print("--- momentum sweep (EMA) ---", flush=True)
    for momentum in MOMENTA:
        for seed in SEEDS:
            _run("full", momentum, seed, rows)

    print("--- no-EMA control ---", flush=True)
    for seed in SEEDS:
        _run("no_ema", None, seed, rows)

    print("--- random-init control (untrained) ---", flush=True)
    for seed in SEEDS:
        encoder = build_encoder("random_init", seed)
        metrics = collapse_metrics(encoder, seed)
        rows.append(
            {
                "variant": "random_init",
                "momentum": None,
                "seed": seed,
                "steps": 0,
                "effective_rank": metrics["effective_rank"],
                "embedding_std": metrics["embedding_std"],
                "final_loss": float("nan"),
                "loss_slope": float("nan"),
            }
        )
        print(
            f"  random_init seed={seed}: rank={metrics['effective_rank']:.3f} "
            f"std={metrics['embedding_std']:.4f}",
            flush=True,
        )

    print("\n--- effective_rank summary (mean over seeds) ---", flush=True)
    print(f"{'variant/momentum':>18} " + " ".join(f"{c:>8}" for c in CHECKPOINTS), flush=True)
    for momentum in MOMENTA:
        cells = []
        for checkpoint in CHECKPOINTS:
            values = [
                r["effective_rank"]
                for r in rows
                if r["variant"] == "full"
                and r["momentum"] == momentum
                and r["steps"] == checkpoint
            ]
            cells.append(f"{statistics.mean(values):>8.3f}")
        print(f"{momentum:>18} " + " ".join(cells), flush=True)
    for checkpoint in CHECKPOINTS:
        values = [
            r["effective_rank"]
            for r in rows
            if r["variant"] == "no_ema" and r["steps"] == checkpoint
        ]
        print(f"  no_ema @{checkpoint:>5}: {statistics.mean(values):.3f}", flush=True)
    random_values = [r["effective_rank"] for r in rows if r["variant"] == "random_init"]
    print(f"  random_init:    {statistics.mean(random_values):.3f}", flush=True)

    print("\n--- results.csv rows ---", flush=True)
    for row in rows:
        params = (
            f'"{{""momentum"":{row["momentum"]},""steps"":{row["steps"]}}}"'
            if row["momentum"] is not None
            else f'"{{""steps"":{row["steps"]}}}"'
        )
        for metric in ("effective_rank", "embedding_std"):
            print(
                f"ISSUE,002-momentum-collapse-boundary,{row['variant']}"
                f"{'' if row['momentum'] is None else '_m' + str(row['momentum'])},"
                f"{row['seed']},{metric},{row[metric]:.4f},{params},2026-08-14",
                flush=True,
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script's wiring cheaply**

Do **not** run the full sweep yet. Verify the plumbing with a tiny override:

```bash
cd /d/Dev/Projects/Model-Experiments
uv run python -c "
import jepa.momentum_steps_sweep as s
s.MOMENTA = (0.0, 0.996)
s.SEEDS = (0,)
s.CHECKPOINTS = (10, 20)
s.MAX_STEPS = 20
s.main()
"
```

Expected: prints checkpoint lines for both momenta, the no-EMA control, the random-init control, a summary table, and `results.csv` rows — all without error. This exercises every code path in about a minute.

- [ ] **Step 3: Lint and commit**

```bash
uvx ruff check .
git add projects/jepa/src/jepa/momentum_steps_sweep.py
git commit -m "jepa: add the momentum x steps collapse-boundary sweep (Arc 1 Slice 2, issue #N)

Offline script, out of the default suite. Records effective_rank, embedding_std,
final loss and loss slope at four checkpoints per run -- the loss statistics
exist to disambiguate the high-momentum confound, where a healthy-looking rank
may just mean the target never moved."
```

---

### Task 5: Run the sweep

**Files:** none — produces a log.

**Interfaces:**
- Consumes: Task 4's script.
- Produces: `$CLAUDE_JOB_DIR/tmp/momentum-sweep.log` with every datapoint and the summary table.

- [ ] **Step 1: Launch it in the background**

~3.1 hours. Do not run it in the foreground.

```bash
cd /d/Dev/Projects/Model-Experiments
uv run python -m jepa.momentum_steps_sweep 2>&1 | tee "$CLAUDE_JOB_DIR/tmp/momentum-sweep.log"
```

Run with `run_in_background: true`.

- [ ] **Step 2: While it runs, sanity-check the first lines**

Read the log once a few checkpoint lines have appeared. Verify `momentum=0.996` at `steps=3000` lands near Slice 1's published 2.35–2.79 for `effective_rank`. If it does not, stop — either the harness refactor changed behaviour or the checkpoint hook is perturbing training, and the remaining ~3 hours would be wasted.

- [ ] **Step 3: On completion, capture the summary**

Save the summary table and the `results.csv` rows section. These feed Tasks 6 and 7.

---

### Task 6: Regression test for the finding

**Files:**
- Create: `projects/jepa/tests/test_momentum_collapse_boundary.py`

**Interfaces:**
- Consumes: `jepa.train.train_jepa`, `jepa.harness.collapse_metrics`.
- Produces: a test locking in the boundary.

- [ ] **Step 1: Write the test**

The directional assertion below needs no magic number and is writable before the sweep finishes; the boundary constant is filled in from Task 5's measured data.

Create `projects/jepa/tests/test_momentum_collapse_boundary.py`:

```python
"""Locks in Arc 1 Slice 2's finding (issue #N). See
../experiments/002-momentum-collapse-boundary.md for the full write-up and the measured surface.

Deliberately reduced step budget relative to the exploratory sweep's 6000: this reproduces the
*direction* of the finding as a regression check, not the full momentum x steps surface.
"""

from __future__ import annotations

import pytest
from jepa.harness import collapse_metrics
from jepa.train import train_jepa

# Shortest budget at which the momentum effect was reliably observed across seeds in the sweep.
# Slice 1 established the gap is invisible at 300 steps, so this cannot go much lower.
BOUNDARY_STEPS = 3000

# Slice 1's operating point -- the known-healthy reference.
HEALTHY_MOMENTUM = 0.996

# The lowest momentum in the sweep grid: target copied from the online encoder every step, with
# no smoothing. Set COLLAPSED_MOMENTUM from Task 5's data to whichever grid value sits clearly on
# the collapsed side of the measured boundary at BOUNDARY_STEPS.
COLLAPSED_MOMENTUM = 0.0


@pytest.mark.slow
def test_low_momentum_collapses_relative_to_the_slice1_operating_point():
    """Headline finding: below the boundary, EMA smoothing is too weak to prevent the target from
    tracking the online encoder, and effective_rank drops relative to Slice 1's 0.996 operating
    point. Same seed and step budget for both -- momentum is the only variable, so this is a
    direct comparison with no threshold constant to drift."""
    healthy, _, _ = train_jepa(
        steps=BOUNDARY_STEPS, seed=0, use_ema=True, ema_momentum=HEALTHY_MOMENTUM
    )
    collapsed, _, _ = train_jepa(
        steps=BOUNDARY_STEPS, seed=0, use_ema=True, ema_momentum=COLLAPSED_MOMENTUM
    )
    healthy_rank = collapse_metrics(healthy, seed=0)["effective_rank"]
    collapsed_rank = collapse_metrics(collapsed, seed=0)["effective_rank"]
    assert healthy_rank > collapsed_rank, (
        f"expected momentum={HEALTHY_MOMENTUM} (effective_rank={healthy_rank:.3f}) to beat "
        f"momentum={COLLAPSED_MOMENTUM} (effective_rank={collapsed_rank:.3f}) -- if this now "
        f"fails, the boundary in experiments/002-momentum-collapse-boundary.md needs revisiting"
    )
```

**If the sweep shows momentum 0.0 does NOT collapse** — i.e. the boundary sits elsewhere, or `effective_rank` is flat across the whole grid — then this directional test asserts something false. In that case replace its body with the finding that *is* true, keeping the same shape: a same-seed, same-budget comparison between two grid points, with the failure message naming the write-up. Do not assert a relationship the data does not show.

- [ ] **Step 2: Run it**

```bash
uv run pytest projects/jepa/tests/test_momentum_collapse_boundary.py -o addopts="" -v
```

Expected: PASS. Takes roughly 9 minutes (two 3000-step runs).

- [ ] **Step 3: Lint and commit**

```bash
uvx ruff check .
git add projects/jepa/tests/test_momentum_collapse_boundary.py
git commit -m "jepa: regression test for the momentum collapse boundary (issue #N)"
```

---

### Task 7: Write-up, records, and PR

**Files:**
- Create: `projects/jepa/experiments/002-momentum-collapse-boundary.md`
- Modify: `projects/jepa/results.csv`
- Modify: `projects/jepa/CLAUDE.md`

**Interfaces:**
- Consumes: Task 5's log, Task 6's test.
- Produces: issue #N closed.

- [ ] **Step 1: Append `results.csv` rows**

Take the `--- results.csv rows ---` block from Task 5's log, replace the literal `ISSUE` prefix with Task 1's issue number, and append to `projects/jepa/results.csv`. Header is already `issue,experiment_slug,variant,seed,metric,value,params,date`. One row per datapoint — do not average into a single row.

- [ ] **Step 2: Write `experiments/002-momentum-collapse-boundary.md`**

Follow `projects/jepa/experiments/TEMPLATE.md`. It must contain:

1. The one-line question as the title, with `(issue #N)`.
2. Motivation — Slice 1 measured one point; the gap is invisible at 300 steps, so momentum and training length are not separable. Cite Slice 1's write-up and Assran et al. (arXiv:2301.08243).
3. Implementation — the checkpoint hook, the harness promotion, and the momentum grid, including **why 0.0 and 0.9999 are in it** (both degenerate ends) and why momentum 0.0 differs from `use_ema=False`.
4. **`**Result: <one-line verdict>.**`** on its own line.
5. The momentum × steps `effective_rank` table (mean over seeds), with the no-EMA and random-init reference rows.
6. Prose interpretation — where the boundary is and how it moves with training length.
7. **The high-momentum confound, addressed explicitly with the loss evidence.** State for each high-momentum cell whether its rank reflects a learned representation or an untrained one, citing `final_loss`/`loss_slope`. Slice 1's random-init rank (2.44–2.93) is the reference line.
8. A line naming `tests/test_momentum_collapse_boundary.py` as what locks the finding in.
9. **Leads for whoever picks this up next** — at minimum: Slice 3 (masking ratio, predictor depth), and momentum *schedules*, which this slice deliberately deferred until the static boundary was known.

- [ ] **Step 3: Update `projects/jepa/CLAUDE.md`**

Two edits:

- Add a row to the experiment index (currently reading "No experiments recorded yet" — replace that line if Slice 1's merge did not already):

```markdown
| #N | Where is the EMA-momentum collapse boundary, and how does it move with training length? | <verdict> | `experiments/002-momentum-collapse-boundary.md` |
```

- In the Arc 1 roadmap, record the Slice 2/3 split: Slice 2 is momentum × steps (this work); **Slice 3** is the masking-ratio and predictor-depth sweep, still to be filed.

- [ ] **Step 4: Full verification**

```bash
uvx ruff check .
uv run pytest -q --deselect tools/viz/tests/test_plotly_fields.py::test_export_png_writes_nonempty_file --deselect tools/viz/tests/test_plotly_fields.py::test_render_orbit_gif_writes_nonempty_file
uv run pytest projects/jepa -o addopts="" -q
```

Expected: ruff clean, fast suite green, full jepa suite (slow included) green.

- [ ] **Step 5: Commit and open the PR**

```bash
git add projects/jepa/experiments/002-momentum-collapse-boundary.md \
        projects/jepa/results.csv projects/jepa/CLAUDE.md
git commit -m "jepa: where is the EMA-momentum collapse boundary? (Arc 1 Slice 2, issue #N)

<one-line verdict>."
git push -u origin feat/jepa-momentum-collapse-boundary
gh pr create --base main --title "jepa: EMA-momentum collapse boundary (Arc 1 Slice 2, issue #N)" --body "$(cat <<'EOF'
## Summary

Closes #N.

<Three bullets: where the boundary is, how it moves with training length, and how the high-momentum confound was resolved.>

## Test plan

- [ ] `uvx ruff check .` — clean
- [ ] `uv run pytest -q` (fast suite) — passes
- [ ] `uv run pytest projects/jepa -o addopts="" -q` (including slow) — passes
- [ ] Checkpoint-hook determinism verified: a hooked run is bit-identical to an unhooked one

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr checks --watch
```

- [ ] **Step 6: Hand off for review**

The issue is `autonomy:review`. **Do not self-merge.** Report the PR number and the finding.

---

## Final verification

```bash
cd /d/Dev/Projects/Model-Experiments
git checkout main && git pull --ff-only origin main
gh issue view <N> --json state --jq .state     # expect: CLOSED
uv run pytest -q                                # expect: green
git log --oneline -3
```

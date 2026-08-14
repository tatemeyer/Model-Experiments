# Runway Clearing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock the two stranded PRs, fix the Dependabot resolution failure, prune 18 worktree directories and ~92 branches, and finish issue #43 from work salvaged out of a worktree scheduled for deletion.

**Architecture:** Three sequential phases — unblock (GitHub operations only), hygiene (dependency fix, salvage, pruning), then research (issue #43). Phases are ordered by hard dependencies, not preference: PR #87 and the `requires-python` bump both rewrite `uv.lock`, and the Helmholtz salvage must complete before its worktree is deleted.

**Tech Stack:** Python 3.12+, `uv` workspace, PyTorch (CPU-only), pytest, ruff, `gh` CLI, git.

## Global Constraints

- Repository is `D:\Dev\Projects\Model-Experiments`. Work on feature branches in the main checkout — do NOT create a git worktree; Phase 2 deletes worktrees.
- Python floor becomes `>=3.12` after Task 4. Before Task 4 it is `>=3.11`.
- ruff: `line-length = 100`, `select = ["E", "F", "I", "UP"]`. Run `uvx ruff check .` before every commit.
- CPU-only. No GPU. No new runtime dependencies for issue #43 — `torch` only.
- Individual training runs stay well under a minute; do not raise `steps` or network size without re-checking runtime.
- Any test that trains a model gets `@pytest.mark.slow` (20 of 23 existing em-piml test files do). The default suite deselects `slow` and `gpu`.
- Determinism is a standing project rule: same seed in → bit-identical result out, verified before any number is trusted.
- Commit style matches existing history: `scope: description` (e.g. `em-piml: ...`, `ci: ...`, `docs: ...`).
- Never force-push anything except PR #87's own branch (Task 2), which is explicitly authorized.
- Squash-merge PRs, matching commits #82–#90.

---

## Amendment 2026-08-14: Tasks 1 and 2 are superseded

**Tasks 1 and 2 below rest on a false premise and must not be executed as
written.** Both PRs targeted stale intermediate branches, not `main`:

| PR | Issue | Base branch | GitHub state | Content on `main`? |
|---|---|---|---|---|
| #86 | #62 | `feat/em-piml-field-array-persistence` | MERGED | **No** |
| #87 | #63 | `feat/em-piml-field-array-persistence` | OPEN | No |
| #88 | #64 | `feat/em-piml-field-render-core` | MERGED | **No** |

Only #85 had `main` as its base. #86 and #88 display as MERGED but merged
into branches that were themselves never fast-forwarded into `main`, so
three finished slices are stranded. This — not a stacked-PR auto-close
quirk — is why issues #62/#63/#64 stayed open.

Executing Task 1 as written merged #88 into the dead
`feat/em-piml-field-render-core` branch, with **zero net effect on `main`**.
Its squash commit `25faa10` survives there; nothing was lost.

Verified by dry run: `git rebase --onto main d161c04 25faa10` replays issues
#62 and #64 onto `main` with **no conflicts**. Cherry-picking #63's
`670efe9` conflicts in exactly two files — `tools/viz/pyproject.toml` and
`uv.lock` — resolved by unioning the dependency lists and running `uv lock`.
The plotly source and test files apply cleanly.

**Replace Tasks 1 and 2 with Tasks 1A, 1B, and 1C**: three sequential PRs
based on `main`, squash-merged in issue order, so `main` gets one clean
squash commit per issue and each issue auto-closes on its own PR.

### Task 1A: Restack issue #62 (field rendering + PyVista 3D surface)

- [ ] **Step 1:** `git checkout main && git pull --ff-only origin main`
- [ ] **Step 2:** `git checkout -b fix/restack-field-render-core`
- [ ] **Step 3:** `git cherry-pick 2dd3ff3` — expected clean.
- [ ] **Step 4:** `uv sync --all-packages && uvx ruff check . && uv run pytest -q`
- [ ] **Step 5:** `git push -u origin fix/restack-field-render-core`
- [ ] **Step 6:** `gh pr create --base main --title "mx-viz: new per-frame field rendering + PyVista 3D surface (issue #62)" --body "..."` with `Closes #62` and a note that this restacks PR #86's work, which merged into a stale base and never reached `main`.
- [ ] **Step 7:** `gh pr checks --watch` then `gh pr merge --squash --delete-branch`

### Task 1B: Restack issue #63 (Plotly interactive wrapper)

- [ ] **Step 1:** `git checkout main && git pull --ff-only origin main`
- [ ] **Step 2:** `git checkout -b fix/restack-plotly-interactive`
- [ ] **Step 3:** `git cherry-pick 670efe9` — expect conflicts in `tools/viz/pyproject.toml` and `uv.lock` only.
- [ ] **Step 4:** Resolve `tools/viz/pyproject.toml` by keeping **every** entry from both sides — the `[3d]` extra's `pyvista`/`trame`/`trame-vtk`/`trame-vuetify`/`nest-asyncio2`/`imageio` *and* the plotly requirement. PyVista's `export_html` imports the `trame*` packages at call time; dropping them breaks Task 1A's work.
- [ ] **Step 5:** `git checkout --ours uv.lock && uv lock` — never hand-merge lockfile conflict markers.
- [ ] **Step 6:** `git add -A && git cherry-pick --continue`
- [ ] **Step 7:** `uv sync --all-packages && uvx ruff check . && uv run pytest -q`
- [ ] **Step 8:** Push, `gh pr create --base main` with `Closes #63`, watch checks, squash-merge.
- [ ] **Step 9:** Close PR #87 with a comment pointing at the replacement PR, since its branch is now obsolete.

### Task 1C: Restack issue #64 (end-to-end PoC rerender)

- [ ] **Step 1:** `git checkout main && git pull --ff-only origin main`
- [ ] **Step 2:** `git checkout -b fix/restack-poc-rerender`
- [ ] **Step 3:** `git cherry-pick 25faa10` — expected clean once #62 is on `main`.
- [ ] **Step 4:** `uv sync --all-packages && uvx ruff check . && uv run pytest -q`
- [ ] **Step 5:** Push, `gh pr create --base main` with `Closes #64`, watch checks, squash-merge.

After 1A–1C, Task 3's issue-closing steps are mostly redundant — verify
rather than re-close. Task 3's branch-protection issue is still required,
and should additionally note that PRs merging into stale non-`main` bases
went unnoticed for ten days.

---

### Task 1: Merge PR #88 — SUPERSEDED, DO NOT EXECUTE

**Files:** none — GitHub operation only.

**Interfaces:**
- Consumes: nothing.
- Produces: `main` containing field-visualization Slice 5; `origin/main` advanced.

- [ ] **Step 1: Confirm the PR is still green and mergeable**

```bash
cd /d/Dev/Projects/Model-Experiments
gh pr view 88 --json state,mergeable,mergeStateStatus,statusCheckRollup \
  --jq '{state,mergeable,mergeStateStatus,checks:[.statusCheckRollup[]?|{name,conclusion}]}'
```

Expected: `state: OPEN`, `mergeable: MERGEABLE`, `mergeStateStatus: CLEAN`, the `verify` check `SUCCESS`.

If it is no longer `CLEAN`, stop and rebase it the same way Task 2 rebases #87.

- [ ] **Step 2: Squash-merge**

```bash
gh pr merge 88 --squash --delete-branch
```

- [ ] **Step 3: Update the local checkout**

```bash
git checkout main
git pull --ff-only origin main
git log -1 --oneline
```

Expected: the top commit is PR #88's squash commit, ending in `(#88)`.

---

### Task 2: Rebase and merge PR #87 — SUPERSEDED, DO NOT EXECUTE

Superseded by Task 1B above. PR #87's base is not `main`, so rebasing it in
place would not ship it. Retained for the conflict-resolution detail, which
Task 1B reuses.

PR #87 is `CONFLICTING`. Conflicts are in `tools/viz/pyproject.toml`, `tools/viz/src/mx_viz/plotly_fields.py`, `tools/viz/tests/test_plotly_fields.py`, and `uv.lock`, caused by PR #86 landing the `[3d]` extra into the same files.

**Files:**
- Modify: `tools/viz/pyproject.toml`
- Modify: `tools/viz/src/mx_viz/plotly_fields.py`
- Modify: `tools/viz/tests/test_plotly_fields.py`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: `main` from Task 1.
- Produces: `main` containing field-visualization Slice 4.

- [ ] **Step 1: Check out the PR branch and start the rebase**

```bash
git fetch origin
git checkout feat/em-piml-plotly-interactive
git rebase origin/main
```

Expected: rebase stops with conflicts.

- [ ] **Step 2: Resolve `tools/viz/pyproject.toml` by unioning the dependency lists**

Open the file. The `[3d]` extra from PR #86 contains `pyvista`, `plotly`, `trame`, `trame-vtk`, `trame-vuetify`, `nest-asyncio2`, and `imageio`. PR #87 adds its own Plotly requirement. Keep **every** entry from both sides — do not drop `trame*`/`nest-asyncio2`/`imageio`, which PyVista's `export_html` imports at call time and which PR #86 added deliberately.

Remove all `<<<<<<<`, `=======`, `>>>>>>>` markers.

- [ ] **Step 3: Resolve the two `plotly_fields` conflicts**

```bash
git diff --diff-filter=U --name-only
```

For `tools/viz/src/mx_viz/plotly_fields.py` and `tools/viz/tests/test_plotly_fields.py`, keep PR #87's version of its own new code, and keep any import or helper that PR #86 introduced alongside it. Resolve by reading both sides — these are small files.

- [ ] **Step 4: Regenerate the lockfile rather than hand-merging it**

```bash
git checkout --ours uv.lock
uv lock
```

`uv lock` recomputes the lockfile from the (now unioned) `pyproject.toml` files. Never hand-merge `uv.lock` conflict markers.

- [ ] **Step 5: Complete the rebase**

```bash
git add tools/viz/pyproject.toml tools/viz/src/mx_viz/plotly_fields.py tools/viz/tests/test_plotly_fields.py uv.lock
git rebase --continue
```

- [ ] **Step 6: Verify locally before pushing**

```bash
uv sync --all-packages
uvx ruff check .
uv run pytest -q
```

Expected: ruff clean; the fast suite passes (baseline is 92 passed, 27 deselected — the count may rise with #87's new tests).

- [ ] **Step 7: Force-push and wait for CI**

```bash
git push --force-with-lease origin feat/em-piml-plotly-interactive
gh pr checks 87 --watch
```

`--force-with-lease`, not `--force`: it refuses if someone else pushed in the meantime.

- [ ] **Step 8: Squash-merge once green**

```bash
gh pr view 87 --json mergeable,mergeStateStatus --jq '{mergeable,mergeStateStatus}'
gh pr merge 87 --squash --delete-branch
git checkout main && git pull --ff-only origin main
```

---

### Task 3: Close resolved issues and file the branch-protection finding

**Files:** none — GitHub operations only.

**Interfaces:**
- Consumes: merged PRs from Tasks 1 and 2.
- Produces: an issue number for the branch-protection finding, referenced nowhere else.

- [ ] **Step 1: Check which issues auto-closed**

```bash
gh issue view 62 --json state --jq .state
gh issue view 63 --json state --jq .state
gh issue view 64 --json state --jq .state
```

#62's work shipped in PR #86, but the stacked-PR base swap prevented its `Closes #62` from firing. #63 and #64 should have auto-closed with PRs #87 and #88.

- [ ] **Step 2: Close whichever are still open, with a reason**

```bash
gh issue close 62 --comment "Delivered by PR #86 (mx-viz per-frame field rendering + PyVista 3D surface). The PR's \"Closes #62\" did not fire because the PR was stacked on #85 and its base changed at merge time. Closing manually."
```

Repeat for 63 and 64 only if Step 1 showed them `OPEN`, adjusting the PR number (#87 for 63, #88 for 64).

- [ ] **Step 3: File the branch-protection finding**

```bash
gh issue create \
  --title "main has no branch protection, contradicting the documented CI-gated merge policy" \
  --label "enhancement" \
  --body "\`README.md\` and \`CLAUDE.md\` describe required status checks and autonomy-label-gated auto-merge, but \`GET /repos/tatemeyer/Model-Experiments/branches/main/protection\` returns 404 — \`main\` is entirely unprotected.

Found during the 2026-08-14 runway-clearing pass (see \`docs/superpowers/specs/2026-08-14-runway-clearing-design.md\`). Deliberately not fixed there: configuring protection interacts with \`auto-merge.yml\` and could block in-flight PRs, so it deserves its own deliberate pass rather than being a footnote in cleanup.

Decide: which checks are required, whether \`enforce_admins\` is on, and how that interacts with the \`autonomy:safe\` self-merge path."
```

- [ ] **Step 4: Confirm no unintended PRs remain open**

```bash
gh pr list --state open
```

Expected: empty.

---

### Task 4: Fix the Dependabot resolution failure

Root cause: `requires-python = ">=3.11"` across the workspace, but `numpy==2.5.1` requires `>=3.12`. uv cannot resolve, so **every** Dependabot update fails — not only numpy's.

**Files:**
- Modify: `projects/em-piml/pyproject.toml:5`
- Modify: `projects/jepa/pyproject.toml:5`
- Modify: `tools/datasets/pyproject.toml:5`
- Modify: `tools/viz/pyproject.toml:5`
- Modify: `pyproject.toml` (root, `[tool.ruff] target-version`)
- Modify: `CONVENTIONS.md` (new dated entry)
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: `main` from Task 2 (both change `uv.lock`; this must come second).
- Produces: a Python floor of 3.12 that all later tasks build on.

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b chore/require-python-312
```

- [ ] **Step 2: Bump the four workspace packages**

In each of `projects/em-piml/pyproject.toml`, `projects/jepa/pyproject.toml`, `tools/datasets/pyproject.toml`, and `tools/viz/pyproject.toml`, change line 5:

```toml
requires-python = ">=3.12"
```

(from `requires-python = ">=3.11"`). The root `pyproject.toml` has no `requires-python` key — do not add one.

- [ ] **Step 3: Bump ruff's target version in the root `pyproject.toml`**

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 4: Record the decision in `CONVENTIONS.md`**

Append a new dated entry in the same style as the existing ones:

```markdown
## 2026-08-14 — Minimum Python is 3.12

Every workspace package declares `requires-python = ">=3.12"`, and ruff
targets `py312`.

Why: `numpy` 2.5+ requires Python >= 3.12. While the workspace still
declared `>=3.11`, uv could not produce a resolution covering the declared
range, so *every* Dependabot update failed with
`dependency_file_not_resolvable` — not just numpy's. Nothing in this repo
was actually exercising 3.11 (the development venv runs 3.14 and CI pins no
version matrix), so raising the floor costs nothing and unblocks dependency
updates permanently.
```

- [ ] **Step 5: Regenerate the lockfile and re-sync**

```bash
uv lock
uv sync --all-packages
```

- [ ] **Step 6: Verify nothing broke**

```bash
uvx ruff check .
uv run pytest -q
```

Expected: ruff clean, fast suite passes. If the resolution pulled a newer transitive package that breaks a test, fix it here — that is exactly the signal this task exists to surface.

- [ ] **Step 7: Commit and open the PR**

```bash
git add -A
git commit -m "chore: require Python >= 3.12 across the workspace

Unblocks Dependabot, which failed every update (not just numpy's) because
requires-python >=3.11 had no valid resolution against numpy 2.5+."
git push -u origin chore/require-python-312
gh pr create --fill
```

- [ ] **Step 8: Merge once green**

```bash
gh pr checks --watch
gh pr merge --squash --delete-branch
git checkout main && git pull --ff-only origin main
```

---

### Task 5: Salvage the Helmholtz work onto a branch

The only copy of issue #43's implementation is uncommitted inside worktree `.claude/worktrees/agent-a81c78384db7b1953`, whose base is from 2026-07-28. Re-apply it onto current `main` rather than cherry-picking the stale base. **This task must complete before Task 6 deletes that worktree.**

**Files:**
- Create: `projects/em-piml/src/em_piml/helmholtz.py`
- Create: `projects/em-piml/src/em_piml/helmholtz_capacity_sweep.py`
- Modify: `projects/em-piml/src/em_piml/model.py` (add `HelmholtzModePINN` after `CavityPINN`)
- Modify: `projects/em-piml/src/em_piml/train.py` (add imports, `_helmholtz_pinn_loss`, `_train_pinn_adam_helmholtz`, `train_helmholtz_mode`, `evaluate_relative_l2_error_helmholtz`)

**Interfaces:**
- Consumes: `main` from Task 4.
- Produces, for Tasks 8–10:
  - `em_piml.helmholtz.L: float`
  - `em_piml.helmholtz.eigenvalue(mode_order: int) -> float`
  - `em_piml.helmholtz.anchor_x(mode_order: int) -> float`
  - `em_piml.helmholtz.analytical_mode(x: torch.Tensor, mode_order: int) -> torch.Tensor`
  - `em_piml.helmholtz.helmholtz_residual(model: nn.Module, x: torch.Tensor, mode_order: int) -> torch.Tensor`
  - `em_piml.model.HelmholtzModePINN(hidden: int = 64, num_layers: int = 3)`
  - `em_piml.train.train_helmholtz_mode(mode_order: int, hidden: int = 64, num_layers: int = 3, steps: int = 2000, seed: int = 0, n_collocation: int = 200, lr: float = 3e-3) -> HelmholtzModePINN`
  - `em_piml.train.evaluate_relative_l2_error_helmholtz(model: nn.Module, mode_order: int, seed: int = 123, n_points: int = 500) -> float`

- [ ] **Step 1: Branch off the updated main**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/em-piml-helmholtz-capacity
```

- [ ] **Step 2: Copy the two new modules verbatim**

```bash
SRC=.claude/worktrees/agent-a81c78384db7b1953/projects/em-piml/src/em_piml
cp "$SRC/helmholtz.py" projects/em-piml/src/em_piml/helmholtz.py
cp "$SRC/helmholtz_capacity_sweep.py" projects/em-piml/src/em_piml/helmholtz_capacity_sweep.py
```

- [ ] **Step 3: Extract the `model.py` and `train.py` additions as a patch**

```bash
git -C .claude/worktrees/agent-a81c78384db7b1953 diff \
  projects/em-piml/src/em_piml/model.py \
  projects/em-piml/src/em_piml/train.py \
  > "$CLAUDE_JOB_DIR/tmp/helmholtz-additions.patch"
```

- [ ] **Step 4: Apply the patch to the current tree**

```bash
git apply --3way "$CLAUDE_JOB_DIR/tmp/helmholtz-additions.patch"
```

If it conflicts (both files changed on `main` since 2026-07-28), apply the additions by hand instead. They are purely additive — no existing function is modified:
- `model.py`: insert the `HelmholtzModePINN` class after `CavityPINN` and before `FourierCavityPINN`.
- `train.py`: add the `em_piml.helmholtz` imports, add `HelmholtzModePINN` to the existing `em_piml.model` import block, and append the four new functions before the existing `evaluate_relative_l2_error`.

- [ ] **Step 5: Verify it imports and lints**

```bash
uv run python -c "from em_piml.train import train_helmholtz_mode, evaluate_relative_l2_error_helmholtz; from em_piml.helmholtz import analytical_mode; print('ok')"
uvx ruff check .
```

Expected: prints `ok`, ruff clean.

- [ ] **Step 6: Confirm the existing suite still passes**

```bash
uv run pytest -q
```

Expected: no regressions (the new code is additive and untested so far).

- [ ] **Step 7: Commit and push immediately**

Pushing now is the point of this task — it is what makes Task 6 safe.

```bash
git add projects/em-piml/src/em_piml/helmholtz.py \
        projects/em-piml/src/em_piml/helmholtz_capacity_sweep.py \
        projects/em-piml/src/em_piml/model.py \
        projects/em-piml/src/em_piml/train.py
git commit -m "em-piml: salvage Helmholtz eigenvalue implementation (issue #43)

Recovered from an uncommitted worktree scheduled for deletion and rebased
onto current main. Implementation only -- no results, tests, or write-up yet."
git push -u origin feat/em-piml-helmholtz-capacity
```

- [ ] **Step 8: Verify the salvage survived**

```bash
git ls-tree -r origin/feat/em-piml-helmholtz-capacity --name-only | grep helmholtz
```

Expected: both `helmholtz.py` and `helmholtz_capacity_sweep.py` listed. **Do not start Task 6 until this passes.**

---

### Task 6: Prune worktrees

18 directories exist under `.claude/worktrees/`; only 7 are registered with git.

**Files:** none in-tree — filesystem and git metadata only.

**Interfaces:**
- Consumes: the pushed branch from Task 5.
- Produces: `git worktree list` showing only the main checkout.

- [ ] **Step 1: Re-confirm the salvage is safely on the remote**

```bash
git ls-tree -r origin/feat/em-piml-helmholtz-capacity --name-only | grep helmholtz
```

Expected: two files. If not, return to Task 5.

- [ ] **Step 2: Re-check every registered worktree for unexamined work**

```bash
git worktree list --porcelain | grep '^worktree' | sed 's|^worktree ||' | while read -r d; do
  echo "=== $d"
  git -C "$d" status --short 2>&1 | head -20
done
```

Known and accounted for: `agent-a81c78…` (Helmholtz — salvaged in Task 5) and `agent-a4ba15…` (a 142-line earlier draft of the Sobol write-up that already shipped as 106 lines in PR #80 — superseded, discard).

If any *other* worktree shows modified or untracked files that are not obviously superseded, stop and report before deleting.

- [ ] **Step 3: Remove the six readable registered worktrees**

```bash
for w in agent-a4ba1521e2b6c2b66 agent-a7e4a9faf7d0fe973 agent-a81c78384db7b1953 \
         agent-acbd465916b8fc4b9 agent-af1e4190d254f36d5 issue-46-dielectric; do
  git worktree remove --force ".claude/worktrees/$w" && echo "removed $w"
done
```

- [ ] **Step 4: Handle the Administrator-owned worktree**

`agent-ad160d26eb8896bd1` is owned by `BUILTIN\Administrators`; git refuses with "dubious ownership".

```bash
git config --global --add safe.directory "D:/Dev/Projects/Model-Experiments/.claude/worktrees/agent-ad160d26eb8896bd1"
git worktree remove --force .claude/worktrees/agent-ad160d26eb8896bd1
```

If removal still fails on file permissions, **stop and ask the repository owner** to run an elevated delete of that one directory. Do not attempt to escalate privileges.

- [ ] **Step 5: Delete the 11 orphaned directories**

These are not registered with git — plain directory removal.

```bash
for w in ci-fix-no-slow-tests device-abstraction-slice2 em-piml-merge-conflicts \
         em-piml-piratenets em-piml-rwf-two-mode em-piml-sobol-sampling \
         feat+jepa-research-scaffold feat-em-piml-neusa \
         field-viz-pyvista-headless-ci jepa-bouncing-ball jepa-training-loop; do
  rm -rf ".claude/worktrees/$w" && echo "deleted $w"
done
```

- [ ] **Step 6: Prune stale metadata and verify**

```bash
git worktree prune
git worktree list
ls .claude/worktrees/ 2>/dev/null || echo "(worktrees dir empty or gone)"
```

Expected: `git worktree list` shows only `D:/Dev/Projects/Model-Experiments`, and the directory listing is empty.

---

### Task 7: Prune branches and stop the recurrence

43 local and 49 remote branches, nearly all already squash-merged.

**Files:** none in-tree.

**Interfaces:**
- Consumes: merged PRs from Tasks 1, 2, 4.
- Produces: branch lists reduced to `main`, `feat/em-piml-helmholtz-capacity`, and anything with an open PR.

- [ ] **Step 1: Stop the recurrence first**

```bash
gh api -X PATCH repos/tatemeyer/Model-Experiments -f delete_branch_on_merge=true \
  --jq '.delete_branch_on_merge'
```

Expected: `true`.

- [ ] **Step 2: Build the delete list using merged-PR status, not git ancestry**

Squash-merged branches do **not** appear under `git branch --merged` — they read as "1 ahead". Ancestry is the wrong signal here.

```bash
git fetch --prune origin
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/ | grep -v '^main$'); do
  state=$(gh pr list --head "$b" --state all --json state --jq '.[0].state // "NONE"' 2>/dev/null)
  echo "$state $b"
done | sort | tee "$CLAUDE_JOB_DIR/tmp/branch-states.txt"
```

- [ ] **Step 3: Review the list before deleting anything**

```bash
grep -c '^MERGED' "$CLAUDE_JOB_DIR/tmp/branch-states.txt"
grep -v '^MERGED' "$CLAUDE_JOB_DIR/tmp/branch-states.txt"
```

The second command prints everything that will be **kept**. Confirm it contains `feat/em-piml-helmholtz-capacity` and nothing surprising. Anything marked `NONE` (a branch that never had a PR — e.g. `tmp-cleanup`, `worktree-agent-*`) is kept by default; delete those only after confirming they hold nothing unique:

```bash
git log origin/main..<branch> --oneline
```

An empty result means the branch has no unique commits and is safe to delete.

- [ ] **Step 4: Delete merged local branches**

```bash
grep '^MERGED ' "$CLAUDE_JOB_DIR/tmp/branch-states.txt" | awk '{print $2}' | while read -r b; do
  git branch -D "$b" && echo "deleted local $b"
done
```

`-D` rather than `-d` is correct and necessary: squash-merged branches are not recognized as merged by `-d`. This is safe **only** because the delete list came from merged-PR status.

- [ ] **Step 5: Delete the corresponding remote branches**

```bash
grep '^MERGED ' "$CLAUDE_JOB_DIR/tmp/branch-states.txt" | awk '{print $2}' | while read -r b; do
  git push origin --delete "$b" 2>&1 | tail -1
done
```

Some will already be gone (Tasks 1/2/4 used `--delete-branch`); "remote ref does not exist" is a benign result.

- [ ] **Step 6: Verify**

```bash
git fetch --prune origin
echo "local:  $(git branch | wc -l)"
echo "remote: $(git branch -r | grep -v HEAD | wc -l)"
git branch -a | grep -v HEAD
```

Expected: a short list — `main`, `feat/em-piml-helmholtz-capacity`, and any deliberately-kept branch.

---

### Task 8: Verify determinism and probe the sweep's cost and direction

No number from this implementation is trustworthy until determinism is confirmed — a standing project rule that has caught real bugs twice (issues #19 and #32).

This task also decides two things Task 9 and Task 10 depend on: whether the full sweep fits the runtime budget, and **which direction the capacity finding goes**, which determines what the regression test asserts.

**Files:**
- Create: `$CLAUDE_JOB_DIR/tmp/probe_helmholtz.py` (scratch, not committed)

**Interfaces:**
- Consumes: `train_helmholtz_mode`, `evaluate_relative_l2_error_helmholtz` from Task 5.
- Produces: a recorded per-run wall time, a go/no-go on the full 64-run sweep, and the observed direction of the capacity effect at the hard mode order.

- [ ] **Step 1: Check out the branch**

```bash
git checkout feat/em-piml-helmholtz-capacity
git pull --ff-only origin feat/em-piml-helmholtz-capacity
```

- [ ] **Step 2: Write the probe script**

Create `$CLAUDE_JOB_DIR/tmp/probe_helmholtz.py`:

```python
import time

import torch
from em_piml.train import evaluate_relative_l2_error_helmholtz, train_helmholtz_mode

torch.set_num_threads(1)

# 1. Determinism: identical seed must give a bit-identical result.
a = train_helmholtz_mode(mode_order=4, hidden=32, seed=0, steps=200)
b = train_helmholtz_mode(mode_order=4, hidden=32, seed=0, steps=200)
err_a = evaluate_relative_l2_error_helmholtz(a, mode_order=4)
err_b = evaluate_relative_l2_error_helmholtz(b, mode_order=4)
print(f"determinism: {err_a!r} vs {err_b!r} -> {'PASS' if err_a == err_b else 'FAIL'}")

# 2. Worst-case timing: the widest net at the hard mode order.
start = time.perf_counter()
model = train_helmholtz_mode(mode_order=16, hidden=256, seed=0, steps=2000)
worst = time.perf_counter() - start
worst_err = evaluate_relative_l2_error_helmholtz(model, mode_order=16)
print(f"worst-case run: {worst:.1f}s, relative_l2={worst_err:.4f}")
print(f"projected 64-run sweep upper bound: {worst * 64 / 60:.1f} min")

# 3. Direction: does capacity help at the hard mode order?
start = time.perf_counter()
small = train_helmholtz_mode(mode_order=16, hidden=16, seed=0, steps=2000)
small_time = time.perf_counter() - start
small_err = evaluate_relative_l2_error_helmholtz(small, mode_order=16)
print(f"narrow run: {small_time:.1f}s, hidden=16 relative_l2={small_err:.4f}")
print(f"DIRECTION: hidden=256 ({worst_err:.4f}) vs hidden=16 ({small_err:.4f}) -> "
      f"{'CAPACITY HELPS' if worst_err < small_err else 'CAPACITY DOES NOT HELP'}")

# 4. Easy mode sanity: the fundamental should be comfortably learnable.
easy = train_helmholtz_mode(mode_order=1, hidden=64, seed=0, steps=2000)
print(f"easy mode (n=1, hidden=64) relative_l2={evaluate_relative_l2_error_helmholtz(easy, mode_order=1):.4f}")
```

- [ ] **Step 3: Run the probe**

```bash
uv run python "$CLAUDE_JOB_DIR/tmp/probe_helmholtz.py"
```

- [ ] **Step 4: Act on each result**

- **`determinism: FAIL`** — stop. Check that `torch.manual_seed(seed)` runs *before* `HelmholtzModePINN(...)` is constructed in `train_helmholtz_mode`, which is the exact bug from issue #19. Fix, commit, re-run before continuing.
- **Projected sweep > ~40 min** — reduce scope for Task 10 and record the reduction. Reduce in this order: drop the depth sub-sweep (24 of 64 runs) first, then reduce seeds from `(0, 1, 2, 7)` to `(0, 1)`. Follow issue #46's precedent: state the reduction and its reason explicitly in the write-up. Do **not** silently shrink it.
- **Easy mode `relative_l2` > 0.2** — the fundamental should be easy. Suspect the anchor term or `lr`; investigate before running the full sweep, or the whole sweep measures a broken setup.
- **Record the DIRECTION line.** Task 9 selects its test body from it.

- [ ] **Step 5: Record the probe output**

Paste the probe's stdout into the task notes. Task 10's write-up cites these numbers as the runtime justification for whatever sweep scope was chosen.

---

### Task 9: Write the regression tests

Mirrors the structure of `tests/test_dielectric_interface_capacity.py` (issue #46): one fast structural test that runs in the default suite, plus one `@pytest.mark.slow` test locking the headline directional finding without magic numbers.

This satisfies issue #43's "the regression test must still run by default" — the structural half runs by default, and the training half carries `@pytest.mark.slow` like 20 of the 23 existing em-piml test files.

**Files:**
- Create: `projects/em-piml/tests/test_helmholtz_capacity.py`

**Interfaces:**
- Consumes: everything in Task 5's Produces block, plus Task 8's DIRECTION result.
- Produces: a green test file for Task 10's PR.

- [ ] **Step 1: Write the two fast tests**

Create `projects/em-piml/tests/test_helmholtz_capacity.py`:

```python
from __future__ import annotations

import math

import pytest
import torch
from em_piml.helmholtz import L, analytical_mode, eigenvalue, helmholtz_residual
from em_piml.train import evaluate_relative_l2_error_helmholtz, train_helmholtz_mode


class _AnalyticalMode(torch.nn.Module):
    """Wraps the closed-form eigenfunction as a Module so helmholtz_residual can be applied to
    it -- the residual of the exact solution must be zero by construction."""

    def __init__(self, mode_order: int):
        super().__init__()
        self.mode_order = mode_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return analytical_mode(x, self.mode_order)


def test_analytical_mode_satisfies_helmholtz_equation_and_bcs():
    """Locks in the closed-form reference: E_n(x) = sin(k_n x) has zero Helmholtz residual and
    vanishes at both PEC walls. This is the exactness guarantee every relative-L2 number in
    experiments/043-helmholtz-eigenvalue-capacity.md is measured against."""
    mode_order = 3
    x = torch.linspace(0.05, L - 0.05, 64).reshape(-1, 1)
    residual = helmholtz_residual(_AnalyticalMode(mode_order), x, mode_order)
    assert residual.abs().max().item() == pytest.approx(0.0, abs=1e-3)

    walls = torch.tensor([[0.0], [L]])
    assert analytical_mode(walls, mode_order).abs().max().item() == pytest.approx(0.0, abs=1e-5)

    assert eigenvalue(mode_order) == pytest.approx(mode_order * math.pi / L)


def test_training_is_deterministic_for_a_fixed_seed():
    """This project's standing rule (issues #19 and #32): same seed in, bit-identical result out.
    Cheap enough to run in the default suite at 50 steps -- the point is the seeding contract,
    not accuracy."""
    first = train_helmholtz_mode(mode_order=4, hidden=32, seed=0, steps=50)
    second = train_helmholtz_mode(mode_order=4, hidden=32, seed=0, steps=50)
    assert evaluate_relative_l2_error_helmholtz(
        first, mode_order=4
    ) == evaluate_relative_l2_error_helmholtz(second, mode_order=4)
```

- [ ] **Step 2: Run them and confirm they pass**

```bash
uv run pytest projects/em-piml/tests/test_helmholtz_capacity.py -v
```

Expected: 2 passed. If `test_analytical_mode_satisfies_helmholtz_equation_and_bcs` fails, the physics module is wrong — fix `helmholtz.py` before going further, because every later number depends on it.

- [ ] **Step 3: Append the slow directional test — use the variant matching Task 8's DIRECTION line**

**If Task 8 printed `CAPACITY HELPS`**, append:

```python
# Reduced but real step budget, mirroring tests/test_dielectric_interface_capacity.py: enough to
# reproduce the direction of the finding without re-running the exploratory sweep.
STEPS = 600
HARD_MODE = 16


@pytest.mark.slow
def test_capacity_reduces_relative_l2_error_on_the_hard_mode():
    """Headline finding: on the time-independent Helmholtz eigenvalue target -- which has no time
    dimension, no causality, and none of the long-horizon collapse mechanism -- widening the
    network gives a real drop in relative L2 error at a high mode order. Same seed and step budget
    for both; capacity is the only variable."""
    narrow = train_helmholtz_mode(mode_order=HARD_MODE, hidden=16, seed=0, steps=STEPS)
    wide = train_helmholtz_mode(mode_order=HARD_MODE, hidden=256, seed=0, steps=STEPS)
    narrow_err = evaluate_relative_l2_error_helmholtz(narrow, mode_order=HARD_MODE)
    wide_err = evaluate_relative_l2_error_helmholtz(wide, mode_order=HARD_MODE)
    assert wide_err < narrow_err, (
        f"expected hidden=256 (relative_l2={wide_err:.4f}) to beat hidden=16 "
        f"(relative_l2={narrow_err:.4f}) -- if this now fails, the capacity-helps finding in "
        f"experiments/043-helmholtz-eigenvalue-capacity.md needs revisiting"
    )
```

**If Task 8 printed `CAPACITY DOES NOT HELP`**, append this instead:

```python
# Reduced but real step budget, mirroring tests/test_dielectric_interface_capacity.py: enough to
# reproduce the direction of the finding without re-running the exploratory sweep.
STEPS = 600
HARD_MODE = 16
# Floor the widest network still fails to clear, set with margin below the observed value so the
# test asserts "the gap persists" without flaking on ordinary run-to-run variance. Replace 0.5
# with a value comfortably below the hidden=256 relative_l2 that Task 8's probe actually measured.
PERSISTENT_GAP_FLOOR = 0.5


@pytest.mark.slow
def test_capacity_does_not_close_the_hard_mode_gap():
    """Headline finding: on the time-independent Helmholtz eigenvalue target, widening the network
    does NOT close the high-mode-order gap -- matching issue #25's negative capacity result on the
    time-domain two-mode target, and consistent with Chaudhry's near-zero width-scaling exponents
    (arXiv:2603.12556). Locks in the failure signature, since there is no accuracy bar to clear."""
    wide = train_helmholtz_mode(mode_order=HARD_MODE, hidden=256, seed=0, steps=STEPS)
    wide_err = evaluate_relative_l2_error_helmholtz(wide, mode_order=HARD_MODE)
    assert wide_err > PERSISTENT_GAP_FLOOR, (
        f"hidden=256 reached relative_l2={wide_err:.4f}, below the {PERSISTENT_GAP_FLOOR} floor "
        f"this test locks in -- capacity now appears to help, so the negative finding in "
        f"experiments/043-helmholtz-eigenvalue-capacity.md needs revisiting"
    )
```

Set `PERSISTENT_GAP_FLOOR` from Task 8's measured `hidden=256` error, with the same 2–4x-style margin convention the baseline test uses — comfortably below the observed value so ordinary variance cannot flake it.

- [ ] **Step 4: Run the slow test**

```bash
uv run pytest projects/em-piml/tests/test_helmholtz_capacity.py -v -o addopts=""
```

Expected: 3 passed. `-o addopts=""` clears the default `-m 'not slow and not gpu'` deselection.

- [ ] **Step 5: Lint and commit**

```bash
uvx ruff check .
git add projects/em-piml/tests/test_helmholtz_capacity.py
git commit -m "em-piml: regression tests for the Helmholtz eigenvalue capacity finding (issue #43)"
git push
```

---

### Task 10: Run the sweep, write it up, and open the PR

**Files:**
- Create: `projects/em-piml/experiments/043-helmholtz-eigenvalue-capacity.md`
- Modify: `projects/em-piml/results.csv` (append rows)
- Modify: `projects/em-piml/CLAUDE.md` (experiment index + open leads)
- Modify: `projects/em-piml/LITERATURE.md` (two paper rows)

**Interfaces:**
- Consumes: everything from Tasks 5, 8, 9.
- Produces: issue #43 closed.

- [ ] **Step 1: Start the sweep in the background**

Use whatever scope Task 8 determined. Full scope:

```bash
uv run python -m em_piml.helmholtz_capacity_sweep 2>&1 | tee "$CLAUDE_JOB_DIR/tmp/helmholtz-sweep.log"
```

Run it in the background and continue with Step 2 while it works.

- [ ] **Step 2: While it runs, add the two `LITERATURE.md` rows**

Match the existing table's column shape and verdict vocabulary (tried/worked, tried/didn't, tried/actively-worse, ruled-out, theory-only). Both papers are cited by issue #43:

- Chaudhry, *PINN width-scaling laws*, arXiv:2603.12556 — the direct motivation: width-scaling exponents near zero or negative across several PDEs. Verdict follows Task 8/10's actual result.
- Wang, Yu, Perdikaris, *When and why PINNs fail to train: a neural tangent kernel perspective*, arXiv:2007.14527 — theory-only here; motivates why width mostly stabilizes NTK convergence rather than buying frequency content.

- [ ] **Step 3: Collect the sweep results**

```bash
grep -E 'hidden=|num_layers=' "$CLAUDE_JOB_DIR/tmp/helmholtz-sweep.log" | tail -40
```

- [ ] **Step 4: Append `results.csv` rows**

One row per `(issue, variant, seed, metric)` datapoint, matching the existing header exactly:

```
issue,experiment_slug,variant,seed,metric,value,params,date
```

For example, for the width sweep at the hard mode order:

```
43,043-helmholtz-eigenvalue-capacity,width_hard_n16_hidden256,0,relative_l2,<value>,"{""mode_order"":16,""hidden"":256,""num_layers"":3,""steps"":2000}",2026-08-14
```

Use the real measured values. Every seed gets its own row — do not average into a single row.

- [ ] **Step 5: Write `experiments/043-helmholtz-eigenvalue-capacity.md`**

Follow `experiments/TEMPLATE.md` exactly. It must contain:

1. The one-line question as the title, with `(issue #43)`.
2. Motivation — the five long-horizon experiments that never varied capacity, and issue #25's negative capacity finding on the time-domain two-mode target. Cite both arXiv papers.
3. Implementation — `helmholtz.py`, `HelmholtzModePINN`, `train_helmholtz_mode`; the three-term loss (PDE + BC + anchor) and **why the anchor term exists**: `E=0` trivially satisfies both the residual and the Dirichlet BCs, the same degenerate escape hatch the long-horizon-collapse thread documents.
4. **`**Result: <one-line verdict>.**`** on its own line.
5. Results tables — relative L2 vs. width per mode order, and the depth sub-sweep if it ran.
6. Prose interpretation.
7. **The comparison issue #43 explicitly requires:** does capacity help here where it didn't in issue #25, and if the answers differ, a stated hypothesis for why (the absence of the collapse/causality mechanism is the obvious candidate).
8. A pointwise or per-mode diagnosis if the result is negative or surprising — the template is explicit that a bare aggregate number is not enough.
9. A line naming `tests/test_helmholtz_capacity.py` as what locks the finding in.
10. **Leads for whoever picks this up next.**
11. **If Task 8 forced a scope reduction, a "Scope reduction" section** stating what was cut and why, with the measured runtime that justified it — issue #46's precedent.

- [ ] **Step 6: Update `projects/em-piml/CLAUDE.md`**

Two edits:

- Add a row to the **Standalone** table in the experiment index, using the verdict key (✅ helped / ⚠️ partial / ❌ no effect / 🔻 actively worse):

```markdown
| #43 | Does a Helmholtz eigenvalue waveguide problem show capacity effects the time-domain long-horizon problem doesn't? | <verdict> | `experiments/043-helmholtz-eigenvalue-capacity.md` |
```

- In **Open leads → long-horizon-collapse**, strike through the `issue #43` clause in the "Network capacity has never been varied…" bullet and replace it with a pointer to the result, matching how #40, #38, #39, #41, #35, and #37 were struck through. Leave the #44 and #45 clauses intact — those are still open.

- [ ] **Step 7: Full verification before the PR**

```bash
uvx ruff check .
uv run pytest -q
uv run pytest projects/em-piml -o addopts="" -q
```

Expected: ruff clean, fast suite green, and the full em-piml suite (slow tests included) green.

- [ ] **Step 8: Commit**

```bash
git add projects/em-piml/experiments/043-helmholtz-eigenvalue-capacity.md \
        projects/em-piml/results.csv \
        projects/em-piml/CLAUDE.md \
        projects/em-piml/LITERATURE.md
git commit -m "em-piml: does a Helmholtz eigenvalue problem show capacity effects the time-domain problem doesn't? (issue #43)

<one-line verdict>. Isolates spectral bias from the long-horizon collapse
mechanism by removing the time dimension entirely."
git push
```

- [ ] **Step 9: Open the PR**

```bash
gh pr create --title "em-piml: Helmholtz eigenvalue capacity sweep (issue #43)" --body "$(cat <<'EOF'
## Summary

Closes #43.

<Two or three bullets: the finding, the comparison against issue #25, and the sweep scope actually run.>

## Test plan

- [ ] `uv run pytest -q` (fast suite) — passes
- [ ] `uv run pytest projects/em-piml -o addopts="" -q` (including slow) — passes
- [ ] `uvx ruff check .` — clean
- [ ] Determinism verified: same seed produces a bit-identical result

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr checks --watch
```

- [ ] **Step 10: Hand off for review**

Issue #43 is labeled `autonomy:review`. **Do not self-merge.** Report the PR number and the finding to the repository owner for approval.

---

## Final verification

Run after Task 10 is approved and merged:

```bash
cd /d/Dev/Projects/Model-Experiments
git checkout main && git pull --ff-only origin main
gh pr list --state open                          # expect: empty
gh issue list --state open                       # expect: #43 gone; #62/#63/#64 gone
git worktree list                                # expect: only the main checkout
git branch -a | grep -v HEAD                     # expect: a short list
uv run pytest -q                                 # expect: green
gh run list --branch main --limit 3              # expect: no failures
```

The Dependabot fix cannot be confirmed until its next scheduled run; check with `gh run list --workflow "Dependabot Updates" --limit 3` a day later, or trigger it from the repository's Insights → Dependency graph → Dependabot tab.

---

## Outcome (2026-08-14)

**Phases 1 and 2 complete. Phase 3 deliberately stopped at diagnosis.**

### Shipped

- Issues **#62, #63, #64** restacked onto `main` and closed via PRs
  **#91/#92/#93** — the field-visualization Arc is now actually on `main`.
  PR #87 closed as obsolete.
- **Dependabot fixed** (PR **#95**): `requires-python >= 3.12` across the
  workspace; `uv.lock` collapsed from a split `numpy v2.4.6, v2.5.1` marker
  resolution to a single `v2.5.1`. Confirmed working — the `Dependency Graph`
  "Configured Graph Update: uv" run on `main` succeeded afterwards.
- **All 18 worktree directories removed.** `git worktree remove` reported
  "Filename too long" (Windows MAX_PATH, deep `.venv` paths) and deregistered
  without deleting; the fix was `Remove-Item -LiteralPath '\?\<abs path>'`.
  The Administrator-owned worktree needed no elevated delete after all.
- **Local branches pruned 43 -> 3** (`main` plus the two salvage branches).
  `delete_branch_on_merge` enabled.

### Two salvages, one unplanned

- `feat/em-piml-helmholtz-capacity` — issue #43's implementation.
- `feat/jepa-baseline-collapse-avoidance-salvage` — **issue #69's near-complete
  deliverable**, found uncommitted inside the Administrator-owned worktree that
  git could not even read, one delete away from destruction. Write-up, 27
  `results.csv` rows, three regression tests, and a `use_ema` flag. Reported on
  issue #69; no PR opened (it is `autonomy:review`).

### Corrections to this plan's own premises

1. **Tasks 1-2 were wrong** — see the Amendment above. Both PRs targeted stale
   non-`main` bases; `mergeStateStatus: CLEAN` was true relative to a dead base.
2. **`main` is NOT unprotected.** The initial finding came from
   `GET /branches/main/protection` returning 404, which is simply what the
   classic endpoint returns for a **ruleset**-protected branch. Three rulesets
   are active; the `main` one enforces PRs, the `verify` check (strict), linear
   history, and blocks deletion/non-fast-forward. Issue #94 has been corrected
   and retitled.
3. **Remote branch pruning is impossible as configured.** The
   `feature-branches` ruleset has a `deletion` rule covering `feat/**`,
   `fix/**`, `docs/**`, `chore/**`, so all 50 merged remote branches are
   undeletable — this is also why `--delete-branch` and
   `delete_branch_on_merge` never worked. Left in place by decision; recorded
   on issue #94.

### Why Phase 3 stopped

The salvaged Helmholtz implementation **collapses to the trivial `E = 0`
solution for every mode order above the fundamental** (`relative_l2` 1.0000 at
n>=4), with the amplitude anchor itself ignored. Verified mechanism: `loss_pde`
carries a `k^2 ~ 2530` factor at n=16 against a single-point, weight-1 anchor,
so `E -> 0` is the cheapest minimum. Out-of-tree rebalancing recovers n=2
(0.9445 -> 0.0095) and n=4 (1.0000 -> 0.1551) at identical capacity, but not
n=8.

Running the capacity sweep in this state would measure collapse, not capacity,
and produce a confidently wrong "capacity does not help" result. Choosing a
loss formulation is a design decision that belongs in a
`superpowers:brainstorming` pass, not an ad-hoc mid-implementation fix. Full
diagnosis posted to issue #43; the branch carries the recovered code plus two
passing fast tests (closed-form reference correctness, seed determinism) and
deliberately no capacity regression test.

Budget note for whoever picks it up: the full 64-run sweep measured **~107
min** (100.1s worst case at hidden=256, n=16), over convention — it needs
reducing, with the reduction stated explicitly (issue #46 precedent).

### Known environment caveat

`test_export_png_writes_nonempty_file` and
`test_render_orbit_gif_writes_nonempty_file` drive kaleido's headless-Chromium
subprocess and **hang indefinitely on Windows**. Deselect them locally; Linux
CI runs them normally. Pre-existing, documented in `670efe9`'s commit message.

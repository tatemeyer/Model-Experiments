# Clear the runway: unblock, prune, and finish issue #43

**Date:** 2026-08-14
**Status:** approved

## Why this exists

Model-Experiments went idle on 2026-08-04 mid-stream. Two PRs were left
open, several issues stayed open despite their work having shipped, 18
worktree directories accumulated on disk, ~90 local and remote branches
were left behind, and Dependabot's `uv` job started failing on 2026-08-11.
None of this is broken code — the fast suite passes (92 passed, 27
deselected) and CodeQL is green — but all of it is friction on restarting
work.

This design covers clearing that friction, plus finishing one piece of
research (issue #43) whose only surviving implementation lives as
uncommitted files inside a worktree that is scheduled for deletion.

**Where this document lives, and why not `docs/design/specs/`.** That tree's
own `README.md` scopes it as a bounded, owner-directed exception for the
em-piml modernization Design, explicitly *not* a repo-wide default. This
work is not part of that Design, so it does not get a Design/Arc/Slice
folder or a Rev-A→Rev-0 lifecycle.

## Phase 1 — Unblock

GitHub operations only; no repository code changes.

1. **Merge PR #88** (`feat/em-piml-poc-experiment-rerender`,
   field-visualization Slice 5). Already green and `MERGEABLE`.
   Squash-merge, matching the current pattern on `main` — commits #82
   through #90 are all squash commits carrying a `(#NN)` suffix.

2. **Rebase PR #87** (`feat/em-piml-plotly-interactive`, Slice 4) onto the
   updated `main`. It is `CONFLICTING`, with conflicts in
   `tools/viz/pyproject.toml`, `tools/viz/src/mx_viz/plotly_fields.py`,
   `tools/viz/tests/test_plotly_fields.py`, and `uv.lock`. The cause is PR
   #86 landing the `[3d]` extra (`trame`, `trame-vtk`, `trame-vuetify`,
   `nest-asyncio2`, `imageio`) into the same files.

   Resolution: union the dependency lists by hand, and **regenerate
   `uv.lock` with `uv lock` rather than hand-merging it**. Then run the
   fast suite, force-push, wait for CI, squash-merge.

3. **Close issues #62, #63, #64.** #62's work shipped in PR #86, but the
   stacked-PR base swap prevented the `Closes #62` line from firing.
   #63/#64 should auto-close when #87/#88 merge; verify and close manually
   if they do not.

**Ordering constraint:** #88 merges first because it is already clean; #87
then rebases onto a `main` that contains it.

## Phase 2 — Hygiene

4. **Fix Dependabot, as its own PR.** Root cause: `requires-python` is
   `>=3.11` across the workspace, but `numpy==2.5.1` requires `>=3.12`, so
   uv cannot resolve and *every* Dependabot update fails, not just numpy.

   Bump `requires-python` to `>=3.12` in all four workspace packages
   (`projects/em-piml`, `projects/jepa`, `tools/datasets`, `tools/viz`),
   bump root `[tool.ruff] target-version` to `py312`, regenerate `uv.lock`,
   and add a dated `CONVENTIONS.md` entry recording the minimum-Python
   decision and its motivation. `CONVENTIONS.md` currently records no
   Python-version decision, so this contradicts nothing.

   This must land **after** #87, because both change `uv.lock`.

5. **Salvage the Helmholtz work.** Create branch
   `feat/em-piml-helmholtz-capacity` off the updated `main` and re-apply the
   four salvaged pieces — `helmholtz.py`, `helmholtz_capacity_sweep.py`, and
   the additions to `model.py` and `train.py` — onto current `main` rather
   than cherry-picking their stale 2026-07-28 base. Commit and push
   immediately. This branch is also Phase 3's working branch.

   This step must complete before step 6, or the work is destroyed with its
   worktree.

6. **Prune worktrees.** Eighteen directories exist under
   `.claude/worktrees/`; only seven are registered with git.

   - Seven registered: `git worktree remove`, using `--force` where dirty
     (the only work worth keeping is preserved by step 5).
   - Eleven orphaned directories: delete outright, then `git worktree prune`.

   Re-check each registered worktree's `git status` immediately before
   removal so nothing unexamined is destroyed.

7. **Set `delete_branch_on_merge` to `true`** so merged branches stop
   accumulating.

8. **Prune branches, merged-only.** 43 local and 49 remote.

   The deletion criterion is **"this branch has a MERGED PR on GitHub"**
   (`gh pr list --head <branch> --state merged`), *not* git ancestry:
   squash-merged branches do not appear under `git branch --merged` and
   instead read as "1 ahead," so an ancestry-based check would either spare
   everything or require force-deleting blind.

   Never deleted: `main`, `feat/em-piml-helmholtz-capacity`, and any branch
   with an open PR.

9. **File an issue for the branch-protection drift.** `main` has no branch
   protection (the API returns 404) while `README.md` and `CLAUDE.md`
   describe required status checks and autonomy-label-gated auto-merge.
   Deliberately not fixed here — it is a sensitive change that deserves its
   own pass.

## Phase 3 — Finish issue #43 (Helmholtz eigenvalue capacity)

The salvaged code already implements the problem: `helmholtz.py` (eigenvalue,
anchor point, closed-form mode, residual), `HelmholtzModePINN`, a training
and evaluation path in `train.py`, and a sweep script covering widths
{16, 32, 64, 128, 256} × two mode orders (n=1 easy, n=16 hard) × 4 seeds,
plus a depth sub-sweep. What is missing is verification, numbers, and every
written record.

10. **Verify determinism first.** Same seed in, bit-identical result out.
    This is the project's standing rule and has caught real bugs twice
    (issues #19 and #32). No number is trusted before this passes.

11. **Timing probe before committing to the full sweep.** Time one run at
    the worst case (`hidden=256`, `mode_order=16`) and extrapolate over all
    64 runs (40 width + 24 depth). If it exceeds the project's runtime
    budget, use the issue's own stated escape hatch — keep the expensive
    sweep out of the default suite, as `point_draw_sweep.py` and
    `num_bands_sweep.py` already do. If a scope reduction is still needed,
    follow issue #46's precedent and **state the reduction explicitly in the
    write-up** rather than quietly shrinking the experiment.

12. **Run the sweep in the background; write the regression test while it
    runs.** `projects/em-piml/tests/test_helmholtz_capacity.py` locks in the
    headline finding and must run in the default fast suite — so it asserts
    the *direction* of the result on a small, fast subset, not the full
    sweep.

13. **Write up `experiments/043-helmholtz-eigenvalue-capacity.md`** per
    `experiments/TEMPLATE.md`: motivation, implementation, results table,
    prose interpretation, and — required explicitly by the issue — a written
    comparison against issue #25's negative capacity finding on the
    time-domain two-mode target, including a stated hypothesis if the two
    answers differ. A standalone top-level file, per the template's rule
    that a genuinely new question starts standalone.

14. **Update the surrounding records.** `results.csv` rows in tidy long
    format (`issue,experiment_slug,variant,seed,metric,value,params,date`);
    a `CLAUDE.md` experiment-index row with a verdict marker; the `#43` lead
    struck through under the `long-horizon-collapse` thread's open leads;
    and `LITERATURE.md` rows for Chaudhry (arXiv:2603.12556) and
    Wang/Yu/Perdikaris (arXiv:2007.14527).

15. **Open a PR with `Closes #43`** and get CI green. The issue is
    `autonomy:review`, so the repository owner approves the merge.

## Verification

Work is done when all of the following hold:

- The fast suite passes locally and in CI on every PR opened here.
- `gh pr list --state open` returns nothing unintended.
- `git worktree list` shows only the main checkout.
- Local and remote branches are reduced to `main` plus genuinely active work.
- The Dependabot `uv` job resolves instead of failing.
- Issue #43 is closed and its write-up is on `main`.

## Risks

- **The sweep exceeds the CI runtime budget.** Fallback: keep it out of the
  default suite, and if scope must shrink, disclose the reduction explicitly
  in the write-up (issue #46 precedent).
- **Worktree `agent-ad160d26eb8896bd1` is owned by `BUILTIN\Administrators`**
  and git refuses to read it ("dubious ownership"). Try `safe.directory`
  first; if the files are ACL-locked, ask the owner to run one elevated
  delete rather than escalating privileges automatically.
- **The `requires-python` bump surfaces transitive resolution changes.** The
  fast suite must pass before that PR merges.
- **Rebasing PR #87 requires a force-push** to its branch. Authorized by the
  repository owner; recorded here because it is normally a sensitive
  operation.

## Operational notes

- Work happens on feature branches in the Model-Experiments checkout, not in
  a git worktree — Phase 2 deletes worktrees, so creating one would be
  self-defeating.
- Phases 1 and 2 steps 6–9 are mechanical/`git-adjacent` and need no review
  gate. Step 4 and all of Phase 3 change code and go through PRs with CI.

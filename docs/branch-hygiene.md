# Branch hygiene: the stale-base / stranded-merge failure

Reference implementation for the failure described in `Model-Experiments` issue
#94. Built and proven here; `parallax` and `SESH` adopt it by copying two files.
Written so a session in another repo can act on it without re-deriving anything.

## The failure

A PR merges, displays as **MERGED**, CI is green — and the work never reaches
the default branch, because the PR was never pointed at it. Four occurrences:

| repo | PRs | how it was caught |
|---|---|---|
| Model-Experiments | #26, #86, #88 | audit, ten days later |
| SESH | #39–#41 (Arc 3 Phases 1–3) | grepping `master` for `ATTENTION_MS` / `Via::read` |
| Parallax | #44 | went `CONFLICTING` after its parent squashed |

There is **no GitHub signal** for this. No notification, no failing check, no
state change. That is what makes it dangerous, and why the guard is a clock
rather than a check on the diff.

## There are two independent problems

**1. Retargeting.** A child PR points at its parent's branch. When the parent
merges and its branch is deleted, GitHub retargets the child to the parent's
base. Fixed by **`delete_branch_on_merge = true`** — if the branch is never
deleted, the retarget never happens and the child merges into a dead branch.

**2. The squash collision.** Fixed by **no setting at all.** A squash-merged
parent rewrites its commits into one new commit on the default branch. A child
branched from the parent still carries the parent's *original* commits, so
rebasing it replays commits that add files the squash already added — `add/add`
conflict on every one. Parallax #44 carried 32 such commits.

`delete_branch_on_merge` is therefore **necessary and not sufficient.** The
ancestry guard is the part that catches problem 2.

### The recovery rule that follows

> When a parent squash-merges, do **not** rebase the child. Re-create the child
> from the default branch and cherry-pick only the child's own commits.

Parallax #44 was recovered exactly this way — one cherry-pick, diff matched.

## What to copy

Two files, repo-agnostic — they resolve the default branch at runtime and take
the repo from `GITHUB_REPOSITORY`. No edits needed:

- `.github/scripts/branch_hygiene.sh`
- `.github/workflows/branch-hygiene.yml`

Optionally `.github/stale-base-allowlist.txt` (create it empty; it only needs
entries once Check B finds a benign case).

### What the guard checks

**Check A — open PRs not targeting the default branch.** Zero false positives,
and it fires *before* the stranding, while retargeting is a one-click fix. Runs
per-PR (scoped to that PR only, so another PR's finding never red-Xes yours).

**Check B — merged PRs whose merge commit is not an ancestor of the default
branch.** Runs daily. Lower confidence *by design*: a stacked child whose parent
squash-merged legitimately trips this even though its content shipped — all
three Model-Experiments hits are that case, verified by hand. A Check B hit means
*"prove the content landed"*, not *"work was lost"*. Confirmed-benign PRs go in
the allowlist so the guard converges to silence rather than becoming noise that
gets ignored — which is the failure mode it exists to prevent.

Verified both directions before shipping: exit 1 with findings, exit 0 once
explained.

## Blocker to solve *before* enabling required checks

**A docs-only PR that produces no check run will wait forever** on a ruleset
requiring that check. Hit in Parallax #43.

It does **not** affect Model-Experiments: `ci.yml` is `on: pull_request:` with no
`paths:` filter and its `verify` job has no `if:`, so every PR gets a `verify`
run regardless of content. Confirmed against merged PRs.

It **will** affect any repo whose CI is path-filtered. The fix is a required
check that always runs and aggregates the optional ones:

```yaml
jobs:
  test:
    if: <heavy work only when relevant paths changed>
    # ...

  verify:            # <- the ONLY name listed as a required status check
    needs: [test]
    if: always()     # runs even when `test` is skipped
    runs-on: ubuntu-latest
    steps:
      - name: Gate
        run: |
          # A skipped dependency is success here; only a real failure fails.
          [ "${{ needs.test.result }}" = "failure" ] && exit 1
          [ "${{ needs.test.result }}" = "cancelled" ] && exit 1
          echo "ok"
```

Require `verify`, never the heavy jobs directly. Adding a path filter to a job
whose name is a required check is how a repo silently deadlocks its own merges.

## Ruleset config for a repo that has none

`parallax` and `SESH` currently have **no rulesets and no branch protection** —
`GET /repos/{owner}/{repo}/rulesets` returns `[]` for both.

> **Do not diagnose this with `GET /branches/{branch}/protection`.** That
> endpoint returns **404 for a ruleset-protected branch**, so a 404 means
> "no *classic* protection", not "unprotected". This exact inference has now
> produced a wrong conclusion twice in issue #94's own history. Use the
> `/rulesets` endpoint.

Model-Experiments' working configuration, documented in full in
`.github/SETUP.md` — mirror it:

- **`main` ruleset** (target `~DEFAULT_BRANCH`): require a PR (**0** approvals),
  require status check `verify` (strict / branches up to date), require linear
  history, block force pushes, restrict deletions, **bypass list empty**.
- **`tags` ruleset**: restrict deletions and updates.

### The interaction most likely to break something quietly

Rulesets have **no `enforce_admins` field** — that is classic branch protection.
The equivalent is `bypass_actors`, and empty means *nobody* bypasses, admins
included. Model-Experiments already runs at that strictest setting, deliberately.

The dangerous knob is not admin enforcement, it is
**`required_approving_review_count`**. It is `0`, and it must stay `0` while
`autonomy:safe` self-merge exists:

- `auto-merge.yml` enables GitHub auto-merge using `GITHUB_TOKEN`.
- A token cannot approve a PR, and Settings → Actions has *"Allow GitHub Actions
  to create and approve pull requests"* **unchecked**.
- The sole human is the PR author, and an author cannot self-approve.

So raising it to 1 leaves every `autonomy:safe` PR sitting in auto-merge-enabled
limbo — permanently, and **silently**, which is the same class of failure as #94
itself. If human approval is ever wanted, gate it by *label* in a workflow;
a ruleset cannot distinguish `autonomy:safe` from `autonomy:review`.

Watch `require_extra_approval_for_unattributed_changes` (currently `true`) for
the same reason: with 0 required approvals it is mostly inert, but a commit with
unrecognized authorship can demand an approval nobody is able to give.

## Stacked PRs

**Recommended: don't.** Require every PR to target the default branch, which is
what Check A enforces. With `required_linear_history` and squash merges, stacks
are structurally hostile — a squashed parent *always* strands its child.

If a stack is genuinely needed, the convention must state the recovery rule
above. A convention that permits stacks without saying what happens when the
parent squashes is the convention that produced all four incidents.

## Related repo setting

`delete_branch_on_merge = true` is enabled on all three repos. On
Model-Experiments it is currently **defeated** by the `feature-branches` ruleset,
whose only rule is `deletion` — so merged branches cannot be pruned. Leftover
merged branches are what stacked PRs accidentally target, so pruning them is
causally preventive here, not just tidiness. See issue #94 for the decision.

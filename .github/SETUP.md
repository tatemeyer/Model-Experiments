# Manual GitHub setup

Repo-level settings and objects that no GitHub API tool available to
Claude Code sessions can create or change — label creation, rulesets,
most of Settings. Do these once, by hand. Written for a **public**,
solo-maintained repo (it was made public to unlock "Allow auto-merge"
and, as a side effect, several paid Advanced Security features become
free — see below).

## Labels

Settings → Labels → New label, for each:

- `intent` — applied automatically by the Intent issue template
- `autonomy:safe` — implement, open PR, auto-merge on green CI
- `autonomy:review` — implement and open PR, human approves before merge
- `autonomy:human` — do not implement autonomously
- `needs-intent` — issue lacks a verifiable success criterion
- `project:<name>` (e.g. `project:em-piml`, `project:jepa`,
  `project:shared` for cross-cutting work) — which project an Issue/PR
  is scoped to, applied manually (see `CONVENTIONS.md`'s 2026-07-30
  "Multi-project gitops" entry). Note: despite this section's own
  framing above, these three were in fact created via `gh label create`
  from a Claude Code session — the `gh` CLI can create labels; this
  doc's "no GitHub API tool available" claim is stale for label
  creation specifically (rulesets/Settings pages still need doing by
  hand).

## Branch naming convention

Referenced by the rulesets below, so establishing it first:

- `main` — trunk. Always releasable, always green.
- `feat/<slug>` — new functionality.
- `fix/<slug>` — bug fixes.
- `docs/<slug>` — docs/meta-only changes (like this one).
- `chore/<slug>` — tooling, deps, CI.
- `experiment/<slug>` — research spikes that may never merge.
- Any of the above gets a `<project>-` prefix on the slug
  (`feat/jepa-research-scaffold`) when the change is scoped to one
  `projects/<name>/` or `tools/<name>/`; omit it for cross-cutting
  changes. Matches the existing `feature-branches` ruleset glob
  unchanged — see `CONVENTIONS.md`'s 2026-07-30 entry.

## Branches

Settings → General → Default branch → switch to **`main`**. The repo
was created without an initial `main`; until this switch, new clones
and PRs default to whichever branch this setting still points at.

Beyond the default-branch pointer, don't use the legacy "Branch
protection rules" page — GitHub is deprecating it in favor of
**Rulesets** (below), which cover both branches and tags, layer
correctly when multiple rulesets target the same ref, and support
delegated bypass. Skip straight to Rules → Rulesets.

## Tags

No releases are cut yet, so no tags exist. When a project starts
tagging snapshots (e.g. `em-piml` release checkpoints), use
`v<major>.<minor>.<patch>`. The tag ruleset below is worth creating now,
pre-emptively — it costs nothing idle and means nobody can silently
force-push or delete a tag once one exists (see the March 2026
`trivy-action` incident, where a compromised maintainer force-pushed 75
of 76 version tags to redirect them at malicious commits — this is
exactly what tag immutability rules prevent).

## Rules → Rulesets

Repo → Settings → Rules → Rulesets → New ruleset, four of them:

### 1. `main` (Target: branch, pattern `~DEFAULT_BRANCH`)

- Require a pull request before merging — 0 required approvals for now
  (solo maintainer); raise to 1+ if collaborators join
- Require status checks to pass: select **`gate`** (the `CI` workflow) —
  see "The single required check" below. Historically this was `verify`;
  do not switch back to naming a work job directly
- Require branches to be up to date before merging — on (CI is cheap
  here; keeps `main` from absorbing a stale PR)
- Require linear history — on (`auto-merge.yml` always squash-merges)
- Block force pushes — on
- Restrict deletions — on
- Bypass list: none. Leave this empty, even for yourself as admin —
  the point of `autonomy:safe` auto-merge is that CI is the only gate;
  an admin bypass quietly defeats that

### 2. `tags` (Target: tag, pattern `**`)

- Restrict deletions — on
- Restrict updates (no force-push to move an existing tag) — on

### 3. `feature-branches` (Target: branch, patterns `feat/**`, `fix/**`,
   `docs/**`, `chore/**`, `experiment/**`)

- Restrict deletions — on (protects in-progress work from an accidental
  delete while a PR is still open)
> **Open decision (issue #94): this deletion rule defeats
> `delete_branch_on_merge`.** That repo setting is enabled, but this rule
> forbids deleting `feat/**`/`fix/**`/`docs/**`/`chore/**`/`experiment/**`,
> so merged branches cannot be pruned (~50 outstanding) and
> `gh pr merge --delete-branch` reports "Cannot delete this branch".
> The rule's stated intent below — protecting an in-progress branch from an
> accidental delete — is real, but GitHub already offers one-click branch
> restore from the PR page, and leftover *merged* branches are precisely what
> stacked PRs accidentally target. Recommendation: drop this ruleset. It has
> exactly one rule, so removing the rule leaves nothing behind:
>
> ```
> gh api -X DELETE repos/tatemeyer/Model-Experiments/rulesets/18951955
> ```
>
> To restore it later, recreate a branch ruleset named `feature-branches`
> targeting those five patterns with a single `deletion` rule and no bypass
> actors. The conservative alternative is to keep the rule and add a narrow
> bypass actor for cleanup only — but the bypass list is deliberately empty
> repo-wide, so that trades one documented invariant for another.

- Nothing else — these are short-lived, single-author branches;
  over-constraining them (blocking force-push, requiring checks) adds
  friction with no real safety benefit here. Claude Code sessions in
  this repo already default to new commits over amends, so force-push
  isn't part of the normal workflow anyway, but no need to hard-block it
  for edge cases (e.g. you personally cleaning up a branch by hand)

### 4. Enforcement status

Set all of the above to **Active**, not "Evaluate" — Evaluate mode only
logs what *would* have been blocked, it doesn't actually block anything.

## The single required check (issue #94)

`main` requires exactly one status check, named **`gate`**, defined in
`.github/workflows/ci.yml` next to the jobs it gates rather than in this
settings page. It is an `if: always()` job that `needs:` the real work
jobs and fails if any of them reports anything other than `success`.

**Why a name that does no work is the right name to require.** A required
context is matched by string. Job names are not stable: the moment
`verify` becomes a matrix — which this repo's own scaling principle
anticipates, one project per `projects/<name>/`, on a job already taking
8m26s — its contexts become `verify (jepa)`, `verify (em-piml)`, and the
required `verify` never reports again. Every PR then blocks forever on a
check that cannot run, with the fix living here rather than in the diff
that caused it. `gate` decouples the two: adding or splitting a job is a
reviewable change to `ci.yml`, and the ruleset never has to be touched.

`if: always()` is load-bearing. Without it, a failed dependency makes
`gate` **skipped** rather than failed, and GitHub counts a skipped
required check as satisfied — the gate would go green by not running.

### Never require a check name before the job exists on `main`

A required context that no workflow produces blocks **every** PR
permanently, and it does so silently: the PR shows "Expected — waiting for
status to be reported," not a failure. Order matters:

1. Land the workflow change that adds the job to the **default branch**.
2. Confirm it actually ran there (`gh run list --branch main`) and that
   the context name is exactly what the ruleset will name.
3. Only then add the name to the ruleset.
4. **Any PR opened before the job existed keeps a stale merge ref and
   never produces the check.** Reopening does not reliably refresh it;
   `gh pr update-branch <n>` does. Learned the hard way in Parallax.

Doing the switch while **zero PRs are open** avoids step 4 entirely, and
is worth waiting for.

### How this interacts with the `autonomy:safe` self-merge path

`auto-merge.yml` enables GitHub auto-merge on `autonomy:safe` PRs, so the
required check *is* the gate for the autonomous path — nothing else stands
between a labelled PR and `main`. Three interactions are worth knowing
before touching either side, and all three fail **silently**, which is the
signature this whole issue is about:

- **Required approvals must stay at 0.** Raising
  `required_approving_review_count` to 1 deadlocks self-merge
  permanently: `auto-merge.yml` uses `GITHUB_TOKEN`, a token cannot
  approve a PR, "Allow GitHub Actions to create and approve pull
  requests" is deliberately unchecked, and the only human is the PR
  author, who cannot self-approve. Every `autonomy:safe` PR would sit in
  auto-merge-enabled limbo with no error. If human approval is ever
  wanted, gate it **by label in a workflow** — a ruleset cannot tell
  `autonomy:safe` from `autonomy:review`.
- **`strict` (require branches up to date) + auto-merge is a stall
  vector.** With `strict: true`, an auto-merge-enabled PR that falls
  behind `main` cannot merge until its branch is updated, and if that
  update would conflict, auto-merge is turned off with no notification.
  This has already happened here: PR #106 was green and mergeable, #105
  landed, and #106 silently became `CONFLICTING` — two PRs appending to
  the same Markdown file. This repo's conventions *manufacture* that
  conflict, since experiments append to `results.csv`, `LITERATURE.md`
  and `CONVENTIONS.md` by design. Keep `strict: true` — absorbing a stale
  PR is worse — but treat `gh pr update-branch` as routine rather than
  exceptional whenever more than one PR is open.
- **`require_extra_approval_for_unattributed_changes: true`** is mostly
  inert at 0 approvals, but a commit with unrecognised authorship can
  demand an approval nobody is able to give — the same silent stall from
  a third direction.

`branch-hygiene.yml` stays **not required**, deliberately: `audit`
reports on repository state rather than on the diff, so a finding about
another PR must never block an unrelated merge. `base-is-default` is
diff-scoped and could reasonably become blocking, but `needs:` cannot
cross workflow files — making it blocking means either moving that one
job into `ci.yml` so it joins `gate`, or adding a second required name,
which is exactly what `gate` exists to avoid. Left as-is.

## Actions

Settings → Actions → General:

- **Actions permissions**: "Allow `tatemeyer`, and select non-`tatemeyer`,
  actions and reusable workflows" → allow-list `astral-sh/setup-uv`
  (everything else we use, `actions/checkout`, is a GitHub-authored
  action and always allowed). Narrows what a future accepted PR could
  introduce into CI.
- **Fork pull request workflows**: now that the repo is public, anyone
  can open a PR from a fork. Set "Require approval for all outside
  collaborators" — safest default for a solo repo with no expected
  external contributors; a fork's first workflow run always needs your
  explicit approval before it executes.
- **Workflow permissions**: leave the default ("Read repository contents
  permission") — both of our workflows already declare their own
  `permissions:` block (`ci.yml` needs none beyond default read;
  `auto-merge.yml` explicitly requests `contents: write` and
  `pull-requests: write`), so there's no need for a repo-wide write
  default. Leave "Allow GitHub Actions to create and approve pull
  requests" **unchecked** — nothing in this repo needs Actions itself to
  open or approve PRs.

Both third-party actions we use are now pinned by commit SHA (not a
moving version tag) in `ci.yml`, with `dependabot.yml` set up to keep
those pins current via PRs — see Advanced Security below.

### Troubleshooting: CI shows `startup_failure` with zero jobs created

Already happened once — cost several hours and let a handful of PRs
merge without their gating CI actually running. Signature: `CI`
workflow runs show `status: completed, conclusion: startup_failure`,
`total_jobs: 0` (no job ever started, not even a failed one), and the
run **cannot be retried** via the API (`403`) — a policy-level block,
not a transient blip. Other workflow types (CodeQL) keep working fine,
which is what makes it easy to miss.

Root cause that one time: the "Allow or block specified actions and
reusable workflows" allow-list entry was `astral-sh/setup-uv` with no
`@ref` suffix. `ci.yml` references the action pinned by full commit SHA
(`astral-sh/setup-uv@<sha>`); GitHub's allow-list matcher needs the
pattern to include a ref part (`astral-sh/setup-uv@*` to match any ref).
Without it, the action doesn't match the allow-list, so GitHub refuses
to start the workflow at all.

If you see this again: Settings → Actions → General → the allow-list
box → confirm every entry has an `@ref`/`@*` suffix matching how the
action is actually referenced in the workflow files. After fixing it,
the already-failed runs are dead ends — you need a fresh commit/push to
get a new, retriable run.

## Web hooks

Not applicable right now. Claude Code sessions subscribed to a PR
already receive comments/CI/review events through a managed integration
(`subscribe_pr_activity`) — no manual webhook needed for that. Only add
one under Settings → Webhooks if you want a *third-party* integration
(Slack/Discord notifications, etc.) later.

## Environments

Not needed yet — Environments gate *deployments* (required reviewers,
wait timers, environment-scoped secrets), and nothing in this repo
deploys anywhere. Revisit if/when a project publishes something (e.g. a
Pages site, a hosted demo) and that publish step should be gated.

## Codespaces

Optional — only relevant if you personally open a browser/VS Code
Codespace against this repo instead of (or alongside) Claude Code web
sessions; nothing here requires it.

- If you do want it: add a `.devcontainer/devcontainer.json` that runs
  `uv sync --all-packages` post-create, so a Codespace matches the `uv`
  workspace exactly. Ask and I'll scaffold it.
- Either way, set **Settings (personal, not repo) → Billing → Codespaces
  spending limit** to `$0` unless you're actively using them — Codespaces
  compute is billed regardless of repo visibility, unlike Actions
  minutes (see below).

## Pages

Not needed yet — nothing in this repo produces a static site. When a
project wants to publish results (plots, a write-up), prefer Pages
source = **GitHub Actions** (a workflow that builds and deploys) over
the legacy "deploy from a branch" option, so publishing goes through the
same CI path as everything else.

## Advanced Security

Public repos get most of this **free** (it's normally paid for private
repos) — worth turning all of it on. Settings → Code security:

- **Dependabot alerts** — on
- **Dependabot security updates** — on (auto-opens PRs patching
  vulnerable dependencies; triage these like any other PR — they won't
  carry an `autonomy:*` label automatically, so add one by hand before
  expecting auto-merge to touch them)
- **Dependabot version updates** — already configured via
  `.github/dependabot.yml` in this PR (`uv` ecosystem for the workspace,
  `github-actions` for the SHA-pinned workflow actions). Note: Dependabot's
  `uv` support is still rough as of mid-2026 — it sometimes updates
  `uv.lock` without touching `pyproject.toml`'s version constraint, or
  skips a bump entirely if `pyproject.toml` has no constraint at all.
  Spot-check its PRs rather than trusting them blindly.
- **Secret scanning** — on
- **Secret scanning push protection** — on (blocks a push containing a
  recognizable secret pattern before it lands, not just after — the
  important one now that the repo is public)
- **Code scanning (CodeQL)** — enable "Default setup" (auto-detects
  Python, runs on push/PR plus a weekly schedule)
- **Private vulnerability reporting** — on (lets someone report a
  security issue privately instead of filing a public issue)

## Secrets and variables

Nothing needed yet — `mx-data`'s registered sources are all public URLs
or in-repo generators, no auth required. When a future project needs one
(e.g. a HuggingFace token, a W&B API key):

- Add it under Settings → Secrets and variables → Actions, scoped to the
  narrowest token permissions the provider allows — never commit it to
  a file (`.env`, `*.pem`, `*.key` are now gitignored specifically to
  make that harder to do by accident).
- Prefer non-secret config (a default project name, a dataset URL) as
  plain committed config in the project's own files over a repo
  "Variables" entry — an agent reading the repo should be able to see
  it without needing Settings access.
- Codespaces secrets / Dependabot secrets: same guidance, if/when
  needed; nothing to set up now.

## Projects board (optional)

Projects → New project. Suggested columns: `Needs intent`, `Ready`, `In
progress`, `In review`, `Done`. No API tool here can create this — it's
a 1-minute manual step if you want the Insights/Projects view described
in `CLAUDE.md`.

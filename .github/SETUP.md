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

## Branch naming convention

Referenced by the rulesets below, so establishing it first:

- `main` — trunk. Always releasable, always green.
- `feat/<slug>` — new functionality.
- `fix/<slug>` — bug fixes.
- `docs/<slug>` — docs/meta-only changes (like this one).
- `chore/<slug>` — tooling, deps, CI.
- `experiment/<slug>` — research spikes that may never merge.

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
- Require status checks to pass: select **`verify`** (the `CI` workflow)
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
- Nothing else — these are short-lived, single-author branches;
  over-constraining them (blocking force-push, requiring checks) adds
  friction with no real safety benefit here. Claude Code sessions in
  this repo already default to new commits over amends, so force-push
  isn't part of the normal workflow anyway, but no need to hard-block it
  for edge cases (e.g. you personally cleaning up a branch by hand)

### 4. Enforcement status

Set all of the above to **Active**, not "Evaluate" — Evaluate mode only
logs what *would* have been blocked, it doesn't actually block anything.

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

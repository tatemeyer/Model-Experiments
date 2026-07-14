# Manual GitHub setup

These steps configure repo-level settings and objects that no GitHub API
tool available to Claude Code sessions can create (label creation,
branch protection, and repo settings all require scopes/endpoints this
session's GitHub MCP tools don't expose). Do these once, by hand.

## 1. Labels

Settings → Labels → New label, for each:

- `intent` — applied automatically by the Intent issue template
- `autonomy:safe` — implement, open PR, auto-merge on green CI
- `autonomy:review` — implement and open PR, human approves before merge
- `autonomy:human` — do not implement autonomously
- `needs-intent` — issue lacks a verifiable success criterion

## 2. Allow auto-merge

Settings → General → Pull Requests → check **Allow auto-merge**.

Required for `.github/workflows/auto-merge.yml` — without this, `gh pr
merge --auto` in that workflow fails outright.

## 3. Branch protection on `main`

Settings → Branches → Add branch protection rule → branch name pattern
`main`:

- Require a pull request before merging
- Require status checks to pass before merging → select the `verify`
  job from the `CI` workflow

Without this, "auto-merge" just merges immediately — it doesn't actually
wait for CI to go green.

## 4. Default branch

Settings → General → Default branch → switch to `main`.

The repo was created without an initial `main`; until this switch, new
clones and PRs will default to whatever branch this setting still points
at.

## 5. (Optional) Projects board

Projects → New project. Suggested columns: `Needs intent`, `Ready`, `In
progress`, `In review`, `Done`. No API tool here can create this — it's
a 1-minute manual step if you want the Insights/Projects view described
in `CLAUDE.md`.

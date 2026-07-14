# Model-Experiments

This repo is a testbed for **Bitter Lesson Engineering (BLE)**: instead of
handing Claude a methodology to follow, we specify *outcomes* and
*verification*, and let the model figure out *how*. GitHub itself
(Issues, Projects, PRs, Actions, Insights) is the harness — the structured
environment where intent turns into verified, shipped code.

Background: Richard Sutton's "The Bitter Lesson" + Daniel Miessler's posts
on Bitter Lesson Engineering and Intent Engineering
(https://danielmiessler.com/blog/bitter-lesson-engineering,
https://danielmiessler.com/blog/intent-engineering).

## The loop

1. **Intent → Issue.** A human (or Claude, when triaging) files an Issue
   using the Intent template. It states the desired end state and how to
   verify it — never implementation steps. An issue that only describes
   steps is under-specified and should be sent back for clarification
   (`needs-intent` label), not implemented as literally written.
2. **Issue → PR.** Claude Code implements against the stated intent,
   choosing its own approach, and opens a PR that links the issue and
   states how it verified the outcome.
3. **PR → CI.** `.github/workflows/ci.yml` is the source of truth for
   "done." If it can be checked by a machine, it belongs in CI, not in a
   human's head.
4. **Merge.** Issues/PRs labeled `autonomy:safe` auto-merge once CI is
   green (see `.github/workflows/auto-merge.yml`). Anything riskier
   (`autonomy:review`) waits for explicit human approval. Nothing merges
   with red CI, regardless of label.
5. **Insights.** GitHub's Insights/Pulse and Projects views are the
   feedback signal for the experiment itself: are well-specified intents
   (clear outcome + verification) actually completing autonomously more
   often than vague ones? That correlation is the thing being measured
   here, not just the shipped code.

## Autonomy labels

- `autonomy:safe` — Claude may implement, open a PR, and it auto-merges
  on green CI. No human review required.
- `autonomy:review` — Claude implements and opens a PR, but a human must
  approve before merge.
- `autonomy:human` — intentionally left for a human; Claude should not
  implement this one.
- `needs-intent` — the issue lacks a verifiable success criterion; ask
  clarifying questions instead of guessing at implementation.

## Scaling principles

This repo starts as one project and is expected to grow to many
(`projects/<name>/`), and to tens of thousands of lines. Four principles
govern how it scales:

1. **Context is a forest, not a monolith.** Don't centralize
   documentation in one giant doc. Each `projects/<name>/` (and any
   `tools/<name>/`) gets its own scoped `CLAUDE.md` covering only that
   subtree. This root `CLAUDE.md` stays a short router plus the
   cross-cutting rules below — it should not grow project-specific
   detail. Claude Code loads `CLAUDE.md` from the cwd up to the repo
   root, so a session working inside a project only pulls in that
   project's context plus this file, not every other project's.
2. **Internal tooling over agent labor.** Repetitive work (fetching
   data, scaffolding a new project, running sweeps) should become a
   script/CLI under `tools/`, not something an agent re-derives freeform
   each time it comes up. Check `tools/README.md` before writing a
   one-off script — if a tool already does it, use it; if it's a
   recurring need without one, build the tool, don't repeat the labor.
3. **Conventions grow with SOTA, not habit.** `CONVENTIONS.md` is a
   dated, living record of repo-wide technical decisions (package
   manager, linter, ML framework default, etc.). Revisit and update it
   as better practice emerges — don't let it fossilize into "how we've
   always done it."
4. **Docs are agent-first.** Don't write human-oriented comments,
   docstrings, or doc trees for code a human doesn't plan to touch by
   hand. Write only what a future agent needs to avoid re-deriving
   context; skip prose that exists just to explain what the code already
   makes obvious.

## Working rules for Claude Code sessions here

- Read the linked Issue's "Success Criteria" section before writing any
  code — that is the spec. If it's missing or unverifiable, stop and ask
  rather than inventing your own definition of done.
- Prefer letting CI encode verification over describing it in prose.
- Do not add scaffolding, abstractions, or process beyond what the
  current Issue's intent requires — this repo is deliberately minimal
  until real experiments give it code to run.
- Check `tools/README.md` before scripting something ad hoc; check
  `CONVENTIONS.md` before picking a library/tool/pattern a convention
  already covers.

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

## Working rules for Claude Code sessions here

- Read the linked Issue's "Success Criteria" section before writing any
  code — that is the spec. If it's missing or unverifiable, stop and ask
  rather than inventing your own definition of done.
- Prefer letting CI encode verification over describing it in prose.
- Do not add scaffolding, abstractions, or process beyond what the
  current Issue's intent requires — this repo is deliberately minimal
  until real experiments give it code to run.

# Model-Experiments

A testbed for using GitHub (Issues, Projects, Pull Requests, Actions,
Insights) as the harness layer between a human and Claude Code — an
experiment in **Bitter Lesson Engineering** / **Intent Engineering**:
specify the outcome and how to verify it, let the AI decide how.

See [`CLAUDE.md`](./CLAUDE.md) for the full loop and the rules Claude Code
sessions follow in this repo.

## Quick start

1. File an Issue using the **Intent** template — describe the desired
   end state and how it can be verified, not the implementation steps.
2. Label it with an autonomy level (`autonomy:safe`, `autonomy:review`,
   or `autonomy:human`).
3. Claude Code (or a human) implements it and opens a PR.
4. CI verifies it. `autonomy:safe` PRs merge themselves on green CI;
   everything else waits for review.

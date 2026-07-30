# Design specs: process and glossary

This document defines how the `docs/design/specs/` tree works — the vocabulary,
the revision lifecycle, and the review gates. It is repo-wide, reusable
infrastructure: it applies to any Design that ever lives under
`docs/design/specs/`, not only the one that prompted writing it
(`2026-07-28-em-piml-modernization`).

**This process is a bounded, owner-directed exception, not a new repo-wide
default.** Root `CLAUDE.md`'s "Working rules" section says: "Do not add
scaffolding, abstractions, or process beyond what the current Issue's intent
requires — this repo is deliberately minimal until real experiments give it
code to run." Everything outside a specific Design's own tree keeps working
exactly as it does today: a human or Claude files an Intent Issue, Claude
implements against it and opens a PR, CI is the source of truth for "done,"
autonomy labels gate merge. This spec tree does not replace that loop or
introduce epics into the Issue tracker — see "Relationship to the Issue/PR
loop" below.

## The hierarchy: Design → Arc → Slice

```
docs/design/specs/<date>-<design-slug>/<date>-<arc-slug>/<date>-<slice-slug>/
```

- **Design** — the outcome-level umbrella. Describes a domain of change (e.g.
  "modernize em-piml's visualization/ML-framework/compute stack"), not a
  single technique. Stays stable even as the arcs underneath it are added,
  renamed, or abandoned.
- **Arc** — one coherent initiative within a Design (e.g. "migrate to JAX,"
  "adopt Rerun.io for visualization"). An Arc's own first document is itself
  a narrower charter — same Rev-A → Rev-0 lifecycle, scoped to that one
  initiative — and that document's creation date is what mints the Arc
  folder's real date prefix. Don't pre-date an Arc folder before its first
  document actually exists.
- **Slice** — a concrete, implementable unit of work within an Arc, sized
  roughly like a single Issue/PR. Slice documents are where implementation
  detail (file paths, function signatures, test plans) actually belongs —
  Design- and Arc-level documents stay at vision/scope level.

Every document's date prefix is the date *that specific document* was
created, not the date of the Design/Arc it lives under.

## Revision lifecycle: Rev-A → Rev-B → ... → Rev-0

A spec document starts at **Rev-A**. Each round of review produces a new
lettered revision (Rev-B, Rev-C, ...) — the same file, mutated in place; git
history is the audit trail for what changed between revisions, and the
document's own `## Revision History` table (see template below) makes that
trail readable without spelunking `git log`.

Once a revision clears every review gate (see "Review lenses" below), it is
retitled **Rev-0** — frozen, final. This is the one deliberate exception to
"mutate in place": once a document is Rev-0, it becomes read-only by
convention.

## Change Orders

A Rev-0 document does not get edited again. Any further change to it is a
**Change Order** — a new document under that slice's `change-orders/`
subfolder, which itself goes through its own Rev-A → Rev-0 cycle before it
takes effect. The original Rev-0 document's frontmatter gets a
`superseded-by:` pointer once a Change Order is accepted; the Change Order's
frontmatter gets a `supersedes:` pointer back.

Abandoning an Arc or Slice that never reached Rev-0 is **not** a Change
Order — it's a lightweight status change (see frontmatter below). Change
Orders exist to amend frozen content, not to close out work that was never
finished.

## Review lenses ("departments")

In a solo/small-research-repo context, "different departments for QA" cashes
out as distinct reviewer *lenses* applied to the same document, not literal
org units. Every revision carries a `## Gates — Rev-<letter>` checklist
against these six:

1. **Technical feasibility** — do the named prerequisites actually exist, or
   have a credible path to existing, before work that depends on them
   starts?
2. **License/compliance** — does everything proposed avoid depending on
   anything whose license doesn't actually permit this repo's use?
3. **Cost/compute-budget** — does everything stay inside this repo's stated
   compute assumptions, or is a deviation explicitly flagged and justified?
4. **Convention-alignment** — does this revision correctly identify every
   existing `CONVENTIONS.md`/`CLAUDE.md` entry it touches, and propose the
   right mechanism (a new dated entry, gated on a stated trigger) rather than
   silently drifting from them?
5. **Goal-delivery** — does the scope, as written, actually get to the
   stated outcome, or does it quietly narrow/drift?
6. **Security** — credentials/secrets handling, supply-chain risk from new
   dependencies, and any surface a cloud-compute or live-visualization
   component opens up — reviewed by back-tracing from the intended finished
   product to what the spec currently says, not just checked against a
   generic checklist.

## Frontmatter every spec document carries

```yaml
---
title: "<document title>"
design: <date>-<design-slug>
arc: <date>-<arc-slug>
slice: <date>-<slice-slug>
revision: A            # A, B, C, ... then "0" once frozen
status: draft           # draft | in-review | rev-0 | abandoned | superseded
date: <date of this revision>
related-arcs: [<bare-slug>, ...]   # arcs referenced but not yet created
supersedes: null
superseded-by: null
---
```

At the Arc level, `status` additionally tracks `proposed | active | completed
| abandoned` — abandoning an Arc is a legitimate, expected outcome (e.g. a
technical prerequisite turns out unbridgeable), not a failure requiring a
Change Order.

## Relationship to the Issue/PR loop

This spec tree is **upstream design authority**, not a parallel tracking
system. When a Slice document is ready to be built, the work still goes
through this repo's normal loop: an Intent Issue linking back to the Slice
document, a PR implementing it, CI as the source of truth for done,
autonomy-label-gated merge. No epics are introduced into the Issue tracker —
the Design-level index (one per Design, listing its Arcs and their status)
is the closest thing to an epic-tracking surface this process has, and it
lives entirely inside `docs/design/specs/`, not in Issues.

## Glossary

| Term | Meaning |
|---|---|
| Design | Outcome-level umbrella; decomposes into Arcs. |
| Arc | One coherent initiative within a Design; decomposes into Slices. |
| Slice | A concrete, Issue/PR-sized unit of implementable work. |
| Revision | A lettered draft (Rev-A, Rev-B, ...) of a document, in review. |
| Rev-0 | The frozen, final revision — read-only by convention after this point. |
| Gate | One of the six review lenses a revision must clear before advancing. |
| Change Order | A new document amending a Rev-0 document; goes through its own Rev-A → Rev-0 cycle. |

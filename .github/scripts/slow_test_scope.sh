#!/usr/bin/env bash
# Computes which project/tool directories' slow tests ci.yml's "Test (slow)" step should run,
# scoped to what the current PR/push actually touched (issue #71). Writes a space-separated list
# of pytest path args to $GITHUB_OUTPUT's `paths` key; an empty value means "run everything"
# (pytest falls back to root pyproject.toml's testpaths), used both as the safe default and as the
# explicit fallback whenever a root-level/shared file changed.
#
# Inputs (env): EVENT_NAME (github.event_name), BASE_SHA, HEAD_SHA -- for pull_request these are
# the real PR base/head commits (not the synthetic merge commit checkout defaults to); for push
# they're github.event.before/after. Requires a full-history checkout (fetch-depth: 0) so both
# SHAs are resolvable locally.
set -euo pipefail

run_everything() {
    echo "$1"
    echo "paths=" >>"$GITHUB_OUTPUT"
    exit 0
}

if [ -z "${BASE_SHA:-}" ] || [ -z "${HEAD_SHA:-}" ] || [[ "$BASE_SHA" =~ ^0+$ ]]; then
    run_everything "no usable base/head SHA for event '$EVENT_NAME' -- running slow tests for every project"
fi

CHANGED=$(git diff --name-only "$BASE_SHA" "$HEAD_SHA")
echo "changed files (${BASE_SHA:0:8}..${HEAD_SHA:0:8}):"
echo "$CHANGED"

if [ -z "$CHANGED" ]; then
    run_everything "no changed files detected -- running slow tests for every project"
fi

if echo "$CHANGED" | grep -qvE '^(projects|tools)/'; then
    run_everything "a root-level/shared file changed -- running slow tests for every project"
fi

SCOPED=$(echo "$CHANGED" | grep -E '^(projects|tools)/[^/]+/' | cut -d/ -f1-2 | sort -u | tr '\n' ' ')
echo "scoped to: ${SCOPED:-<none>}"
echo "paths=$SCOPED" >>"$GITHUB_OUTPUT"

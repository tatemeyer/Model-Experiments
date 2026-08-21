"""The published metrics feed must still match the record it is derived from (issue #112).

`results.csv` is the record of truth; `results.jsonl` is a checked-in projection of it in the
shape the Parallax cockpit reads. Two copies of the same facts drift -- that is what a second
hand-maintained file always does -- so this test makes divergence a red CI run instead of a
cockpit quietly showing last month's numbers.

Regenerate with:

    uv run mx-viz feed projects/jepa/results.csv --out projects/jepa/results.jsonl

Line endings are deliberately not special-cased: `write_feed` emits LF, git checks the file out
CRLF on Windows, and `Path.read_text` translates CRLF back to LF under universal newlines — so
the comparison below holds on both platforms. Verified, not assumed. Do not "fix" this by
passing `newline=""` to the read; that is what would actually break it.
"""

from __future__ import annotations

from pathlib import Path

from mx_viz import feed

_PROJECT = Path(__file__).resolve().parents[1]
_CSV = _PROJECT / "results.csv"
_JSONL = _PROJECT / "results.jsonl"


def test_published_feed_matches_results_csv():
    assert _JSONL.exists(), "results.jsonl is missing; run `mx-viz feed` (see module docstring)"
    expected = feed.render_feed(_CSV)
    actual = _JSONL.read_text(encoding="utf-8")
    assert actual == expected, (
        "results.jsonl is out of date with results.csv -- regenerate it with "
        "`uv run mx-viz feed projects/jepa/results.csv --out projects/jepa/results.jsonl`"
    )


def test_the_feed_the_manifest_declares_actually_resolves():
    """parallax.yaml's metrics glob is `projects/*/results.jsonl`; this is the file it must find.

    Issue #112's first success criterion, asserted rather than eyeballed: the manifest declared a
    feed nothing backed for as long as nothing checked.
    """
    matches = sorted((_PROJECT.parent).glob("*/results.jsonl"))
    assert _JSONL in matches

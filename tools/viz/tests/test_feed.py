"""Tests for mx_viz.feed (issue #112).

The interesting assertions here are not about JSON formatting -- they are about the *shape* the
consumer reconstructs from these records. `_group_like_consumer` below mirrors Parallax's
`observation`/`parse_metrics` (baseline/src/adapters/artifact.rs): a record with a string
`metric` and a numeric `value` is one observation, its **string** fields are its dimensions, and
its numeric fields are dropped. Series are grouped by (metric, dimensions).

That mirror is what makes these tests worth having. A change to `feed.py` that emits `seed` as a
string, or `issue` as a number, still produces perfectly valid JSONL -- and silently destroys
either the spread a null result lives in or the dimension that separates one experiment from
another. Only a test that groups the way the consumer groups can catch it.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest
from mx_viz import feed

_HEADER = "issue,experiment_slug,variant,seed,metric,value,params,date\n"


def _row(variant: str, seed: int, value: float, *, metric: str = "effective_rank") -> str:
    params = '"{""steps"":3000}"'
    return f"69,001-baseline,{variant},{seed},{metric},{value},{params},2026-08-03\n"


@pytest.fixture
def results_csv(tmp_path):
    """Three variants x three seeds -- Arc 1 Slice 1's shape, with its real values."""
    path = tmp_path / "results.csv"
    path.write_text(
        _HEADER
        + _row("full", 0, 2.779)
        + _row("full", 1, 2.352)
        + _row("full", 2, 2.791)
        + _row("no_ema", 0, 1.336)
        + _row("no_ema", 1, 1.387)
        + _row("no_ema", 2, 1.250)
        + _row("random_init", 0, 2.934)
        + _row("random_init", 1, 2.461)
        + _row("random_init", 2, 2.437),
        encoding="utf-8",
    )
    return path


def _group_like_consumer(jsonl_text: str) -> dict[tuple[str, tuple], list[float]]:
    """Mirror of Parallax's parse_metrics long-format branch. See module docstring."""
    groups: dict[tuple[str, tuple], list[float]] = defaultdict(list)
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name, value = record.get("metric"), record.get("value")
        assert isinstance(name, str), "every record must be long-format"
        assert isinstance(value, float), "value must be numeric, or it is not a measurement"
        dimensions = tuple(
            sorted((k, v) for k, v in record.items() if k != "metric" and isinstance(v, str))
        )
        groups[(name, dimensions)].append(float(value))
    return groups


def test_seeds_collapse_into_one_series_carrying_their_spread(results_csv):
    """The property a null result depends on.

    `seed` is emitted numerically so the consumer drops it as a dimension and the three runs
    become three points of one series. Emitting it as a string would give each run its own
    one-point series -- valid JSONL, and the finding would be gone.
    """
    groups = _group_like_consumer(feed.render_feed(results_csv))

    by_variant = {
        dict(dimensions)["variant"]: points
        for (name, dimensions), points in groups.items()
        if name == "effective_rank"
    }

    assert sorted(by_variant) == ["full", "no_ema", "random_init"]
    assert all(len(points) == 3 for points in by_variant.values())

    # Arc 1 Slice 1's actual conclusion, reconstructed from the feed: `full` sits almost
    # entirely inside `random_init` (not distinguishable from untrained on this metric at this
    # budget), while `no_ema` separates cleanly below both.
    assert (min(by_variant["full"]), max(by_variant["full"])) == (2.352, 2.791)
    assert (min(by_variant["random_init"]), max(by_variant["random_init"])) == (2.437, 2.934)
    assert max(by_variant["no_ema"]) < min(by_variant["random_init"])


def test_no_identifier_is_published_as_a_measurement(results_csv):
    """The defect issue #112 reported against the wide reading of this data.

    An identifier surfacing as a series name means something is charting issue numbers or seed
    indices as if they were measurements.
    """
    groups = _group_like_consumer(feed.render_feed(results_csv))
    names = {name for name, _ in groups}
    assert names == {"effective_rank"}
    assert not names & {"issue", "seed", "steps", "value"}


def test_identifiers_are_strings_and_survive_as_dimensions(results_csv):
    """`issue` is numeric in the CSV; emitted as a number the consumer would drop it."""
    record = json.loads(feed.render_feed(results_csv).splitlines()[0])

    assert record["issue"] == "69"
    assert record["experiment_slug"] == "001-baseline"
    assert record["variant"] == "full"
    # `steps` comes out of params and is emitted as a string for the same reason: it is the axis
    # a duration study groups by, and a numeric field never becomes a dimension.
    assert record["steps"] == "3000"
    assert isinstance(record["seed"], int)
    assert isinstance(record["value"], float)


def test_a_param_colliding_with_a_reserved_field_is_refused(tmp_path):
    """Silently publishing a config parameter as the measurement is the worst available outcome."""
    path = tmp_path / "results.csv"
    path.write_text(
        _HEADER + '69,001-baseline,full,0,effective_rank,2.779,"{""value"":1}",2026-08-03\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collides with the feed's reserved"):
        feed.render_feed(path)


def test_render_is_deterministic_and_newline_terminated(results_csv):
    """The feed is checked in, so an unstable render would produce diff noise every run."""
    first = feed.render_feed(results_csv)
    assert first == feed.render_feed(results_csv)
    assert first.endswith("\n")
    assert len(first.splitlines()) == 9


def test_write_feed_reports_what_it_wrote(results_csv, tmp_path):
    out = tmp_path / "nested" / "results.jsonl"
    assert feed.write_feed(results_csv, out) == 9
    assert out.read_text(encoding="utf-8") == feed.render_feed(results_csv)

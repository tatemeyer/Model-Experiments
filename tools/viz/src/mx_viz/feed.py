"""Publish a project's tidy long-format `results.csv` as the JSONL metrics feed
`parallax.yaml` declares (issue #112).

`results.csv` stays the record of truth. This module writes a *derived projection* of it in
the one shape the Parallax cockpit can currently read, and `test_results_feed.py` in each
project asserts the checked-in `.jsonl` still matches its `.csv`, so the two cannot drift
silently -- the failure mode a second hand-maintained copy would otherwise guarantee.

**The record schema is not a style choice; it is dictated by what the consumer does with each
field type.** Parallax's `parse_metrics` (baseline/src/adapters/artifact.rs) treats a record
carrying a string `metric` and a numeric `value` as one *observation*, and then keeps only the
record's **string** fields as that observation's dimensions -- numeric fields are dropped. Series
are grouped by (metric name, dimensions). Three consequences, all deliberate below:

* **Identifiers are emitted as strings** (`issue`, and every config parameter), so they survive
  as dimensions and partition the series. `issue` emitted as a number would silently vanish.
  This is also the fix for the defect Parallax recorded against this very file: "issue and seed
  are numeric, and nothing marks them as identifiers."
* **`seed` is emitted as a number, on purpose**, so it is dropped as a dimension and its runs
  collapse *into* one series as repeated points. That is what makes a null result legible:
  grouped by (metric, variant), `full` spans 2.352..2.791 against `random_init`'s 2.437..2.934,
  and a reader sees the overlap. Emitting `seed` as a string would give every run its own
  one-point series and destroy exactly the spread the finding lives in.
* **Nothing is emitted that is neither an identifier nor the measurement.** `date` is
  provenance, not a dimension, and including it would partition series by write date for no
  analytical gain. `results.csv` remains the archive.

A long-format feed is also how the ordering question answers itself. The consumer marks any
feed carrying `metric`/`value` records as `Unordered`, because these are measurements of
separate configurations whose record order is the writing loop's nesting -- re-order the loop
and a line chart changes shape without a single measurement changing. That property is stated
by *being* long-format, which matters because a manifest cannot state it: Parallax's
`ArtifactEntry` is `deny_unknown_fields` over exactly `kind`, `adapter` and `watch`, so there is
nowhere in `parallax.yaml` to put a `shape:` or `ordered:` key. The feed has to carry its own
shape, and this one does.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

# Reserved by the consumer: `metric` names the series, `value` carries the measurement.
# A config parameter colliding with either would be read as the measurement itself, so the
# collision is refused loudly rather than silently mis-published.
METRIC_FIELD = "metric"
VALUE_FIELD = "value"
RESERVED_FIELDS = (METRIC_FIELD, VALUE_FIELD)

# Columns of `results.csv` that are not themselves a dimension: `value`/`metric` are the
# measurement, `seed` is the replicate axis (see module docstring), and `date` is provenance.
_DATE_COLUMN = "date"
_PARAMS_COLUMN = "params"
_SEED_COLUMN = "seed"

# Emitted as strings so the consumer keeps them as dimensions. `experiment_slug` and `variant`
# are already strings in the CSV; `issue` is not, and that is the point.
_IDENTIFIER_COLUMNS = ("issue", "experiment_slug", "variant")


def _flatten_params(raw: str) -> dict[str, str]:
    """Config parameters, stringified so each survives as a dimension.

    Every value becomes a string, including numeric ones like `steps`: a numeric field is
    dropped by the consumer's dimension filter, and `steps` is precisely the axis Arc 2's
    duration curve lives on. Nested values are JSON-encoded rather than flattened further --
    no parameter in this repo is nested today, and inventing a flattening convention for a
    case that does not exist would be scaffolding.
    """
    if not raw:
        return {}
    params = json.loads(raw)
    flattened: dict[str, str] = {}
    for key, value in params.items():
        if key in RESERVED_FIELDS:
            raise ValueError(
                f"parameter {key!r} collides with the feed's reserved {key!r} field; "
                "rename it in results.csv before publishing"
            )
        flattened[key] = value if isinstance(value, str) else json.dumps(value)
    return flattened


def records_from_results_csv(csv_path: str | Path) -> Iterator[dict[str, object]]:
    """One JSON-ready record per measurement, in the CSV's own row order.

    Row order is preserved for reviewability of the generated file (a stable diff when a slice
    appends rows), not because it means anything -- the consumer reads this shape as unordered.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record: dict[str, object] = {
                METRIC_FIELD: row[METRIC_FIELD],
                VALUE_FIELD: float(row[VALUE_FIELD]),
                _SEED_COLUMN: int(row[_SEED_COLUMN]),
            }
            for column in _IDENTIFIER_COLUMNS:
                record[column] = str(row[column])
            record.update(_flatten_params(row.get(_PARAMS_COLUMN, "")))
            yield record


def render_feed(csv_path: str | Path) -> str:
    """The full JSONL text for a results CSV, newline-terminated.

    Returned as text rather than written directly so a test can compare it against the
    checked-in file without touching the filesystem.
    """
    lines = [
        json.dumps(record, sort_keys=True) for record in records_from_results_csv(csv_path)
    ]
    return "".join(f"{line}\n" for line in lines)


def write_feed(csv_path: str | Path, jsonl_path: str | Path) -> int:
    """Write the feed beside its source. Returns the number of records written."""
    text = render_feed(csv_path)
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text.count("\n")

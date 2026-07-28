from __future__ import annotations

import json
from pathlib import Path


def save_results(
    path: str | Path,
    results: dict[str, dict],
    metadata: dict | None = None,
) -> None:
    """Save sweep-style results (variant -> {seed: value}) as JSON.

    Keys are coerced to str (JSON object keys must be strings) so callers can pass
    dict[str, dict[int, float]] directly, matching point_draw_sweep.py's/
    num_bands_sweep.py's existing in-memory result shape without reformatting.
    """
    payload = {
        "metadata": metadata or {},
        "results": {
            variant: {str(seed): value for seed, value in seed_values.items()}
            for variant, seed_values in results.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_results(path: str | Path) -> dict:
    """Load a results JSON file written by save_results."""
    return json.loads(Path(path).read_text())

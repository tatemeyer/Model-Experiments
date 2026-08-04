from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIELD_ARTIFACT_SCHEMA_VERSION = 1
_FIELD_ARTIFACT_KEYS = ("x", "t", "grid_x", "grid_t", "true", "predicted", "schema_version")


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


def save_field_artifact(
    path: str | Path,
    *,
    x: np.ndarray,
    t: np.ndarray,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    true: np.ndarray,
    predicted: np.ndarray,
    schema_version: int = FIELD_ARTIFACT_SCHEMA_VERSION,
) -> None:
    """Persist a target/predicted field evaluation as a plain-array .npz artifact.

    x/t are the 1D grid axes; grid_x/grid_t/true/predicted are the 2D (x, t) arrays a
    caller like em_piml.train.evaluate_field_grid already produces. Written with only
    numeric numpy arrays (no pickled objects), so it always round-trips through
    load_field_artifact's allow_pickle=False load.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=np.asarray(x),
        t=np.asarray(t),
        grid_x=np.asarray(grid_x),
        grid_t=np.asarray(grid_t),
        true=np.asarray(true),
        predicted=np.asarray(predicted),
        schema_version=np.asarray(schema_version),
    )


def load_field_artifact(path: str | Path) -> dict[str, np.ndarray]:
    """Load a field artifact written by save_field_artifact.

    allow_pickle=False is passed explicitly (not just relied on as numpy's default) so this
    is never an arbitrary-code-execution sink for a handed-around .npz file -- an artifact
    that requires pickle to load (e.g. containing an object-dtype array) raises instead of
    silently deserializing.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        missing = [key for key in _FIELD_ARTIFACT_KEYS if key not in data]
        if missing:
            raise ValueError(f"field artifact {path} is missing keys: {missing}")
        return {key: data[key] for key in _FIELD_ARTIFACT_KEYS}


def validate_field_artifact(data: dict[str, np.ndarray]) -> None:
    """Check shape/rank invariants of a loaded field artifact, raising ValueError if violated."""
    for key in ("x", "t"):
        if data[key].ndim != 1:
            raise ValueError(f"field artifact key {key!r} must be 1D, got shape {data[key].shape}")
    grid_shape = data["grid_x"].shape
    for key in ("grid_t", "true", "predicted"):
        if data[key].shape != grid_shape:
            raise ValueError(
                f"field artifact key {key!r} shape {data[key].shape} does not match "
                f"grid_x shape {grid_shape}"
            )

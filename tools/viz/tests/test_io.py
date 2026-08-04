from __future__ import annotations

import json

import numpy as np
import pytest
from mx_viz import io


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "results.json"
    results = {"uniform": {"0": 0.92, "1": 0.93}, "causal": {"0": 0.91, "1": 0.94}}
    io.save_results(path, results, metadata={"title": "long-horizon comparison"})

    loaded = io.load_results(path)
    assert loaded["metadata"]["title"] == "long-horizon comparison"
    assert loaded["results"] == results


def test_save_results_coerces_int_seed_keys(tmp_path):
    path = tmp_path / "results.json"
    io.save_results(path, {"variant": {0: 0.5, 1: 0.6}})
    raw = json.loads(path.read_text())
    assert raw["results"]["variant"] == {"0": 0.5, "1": 0.6}


def test_save_results_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "results.json"
    io.save_results(path, {"variant": {"0": 1.0}})
    assert path.exists()


def test_save_results_defaults_metadata_to_empty_dict(tmp_path):
    path = tmp_path / "results.json"
    io.save_results(path, {"variant": {"0": 1.0}})
    assert io.load_results(path)["metadata"] == {}


def _make_field_grid():
    x = np.linspace(0.0, 1.0, 4)
    t = np.linspace(0.0, 1.0, 3)
    grid_x, grid_t = np.meshgrid(x, t, indexing="xy")
    true = np.sin(grid_x) * np.cos(grid_t)
    predicted = true + 0.01
    return x, t, grid_x, grid_t, true, predicted


def test_save_and_load_field_artifact_roundtrip(tmp_path):
    path = tmp_path / "field.npz"
    x, t, grid_x, grid_t, true, predicted = _make_field_grid()
    io.save_field_artifact(
        path, x=x, t=t, grid_x=grid_x, grid_t=grid_t, true=true, predicted=predicted
    )

    loaded = io.load_field_artifact(path)
    assert np.array_equal(loaded["x"], x)
    assert np.array_equal(loaded["t"], t)
    assert np.array_equal(loaded["grid_x"], grid_x)
    assert np.array_equal(loaded["grid_t"], grid_t)
    assert np.array_equal(loaded["true"], true)
    assert np.array_equal(loaded["predicted"], predicted)
    assert int(loaded["schema_version"]) == io.FIELD_ARTIFACT_SCHEMA_VERSION
    io.validate_field_artifact(loaded)  # doesn't raise


def test_save_field_artifact_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "field.npz"
    x, t, grid_x, grid_t, true, predicted = _make_field_grid()
    io.save_field_artifact(
        path, x=x, t=t, grid_x=grid_x, grid_t=grid_t, true=true, predicted=predicted
    )
    assert path.exists()


def test_load_field_artifact_rejects_pickled_contents(tmp_path):
    # An artifact that requires pickle to load (an object-dtype array) must be rejected, not
    # silently deserialized -- load_field_artifact is a sink for files a user hands around, and
    # allow_pickle=True would make it an arbitrary-code-execution vector.
    path = tmp_path / "malicious.npz"
    np.savez(path, x=np.array([object()], dtype=object))

    with pytest.raises(ValueError):
        io.load_field_artifact(path)


def test_load_field_artifact_rejects_missing_keys(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, x=np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="missing keys"):
        io.load_field_artifact(path)


def test_validate_field_artifact_rejects_shape_mismatch():
    x, t, grid_x, grid_t, true, predicted = _make_field_grid()
    data = {
        "x": x,
        "t": t,
        "grid_x": grid_x,
        "grid_t": grid_t,
        "true": true,
        "predicted": predicted[:-1],  # wrong shape
    }
    with pytest.raises(ValueError, match="predicted"):
        io.validate_field_artifact(data)

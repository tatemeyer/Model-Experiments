from __future__ import annotations

import json

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

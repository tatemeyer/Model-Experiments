from __future__ import annotations

from mx_viz import cli, io


def test_sweep_command_writes_image(tmp_path):
    results_path = tmp_path / "results.json"
    io.save_results(results_path, {"uniform": {"0": 0.9, "1": 0.92}}, metadata={"title": "demo"})
    out_path = tmp_path / "sweep.png"

    exit_code = cli.main(["sweep", str(results_path), "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_sweep_command_accepts_kind_bar(tmp_path):
    results_path = tmp_path / "results.json"
    io.save_results(results_path, {"uniform": {"0": 0.9}})
    out_path = tmp_path / "sweep.png"

    exit_code = cli.main(["sweep", str(results_path), "--out", str(out_path), "--kind", "bar"])

    assert exit_code == 0
    assert out_path.exists()

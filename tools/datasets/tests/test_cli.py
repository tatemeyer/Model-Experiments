from __future__ import annotations

import sys

import pytest
from mx_datasets import cli, registry


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    registry_dir = tmp_path / "tools" / "datasets" / "registry"
    generators_dir = registry_dir / "generators"
    generators_dir.mkdir(parents=True)

    generator_script = generators_dir / "gen.py"
    generator_script.write_text(
        "from pathlib import Path\n"
        "dest = Path('.data/fake-dataset')\n"
        "dest.mkdir(parents=True, exist_ok=True)\n"
        "(dest / 'hello.txt').write_text('hi\\n')\n"
    )

    (registry_dir / "fake-dataset.toml").write_text(
        'name = "fake-dataset"\n'
        'description = "test dataset"\n'
        'license = "n/a"\n'
        'destination = ".data/fake-dataset"\n\n'
        "[source]\n"
        'kind = "generator"\n'
        f"command = [{sys.executable!r}, {str(generator_script)!r}]\n"
    )

    monkeypatch.setattr(registry, "repo_root", lambda: tmp_path)
    return tmp_path


def test_list_shows_registered_dataset(fake_repo, capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "fake-dataset" in out
    assert "not fetched" in out


def test_fetch_then_verify(fake_repo):
    assert cli.main(["fetch", "fake-dataset"]) == 0
    assert (fake_repo / ".data" / "fake-dataset" / "hello.txt").exists()
    assert cli.main(["verify", "fake-dataset"]) == 0


def test_verify_detects_tampering(fake_repo):
    assert cli.main(["fetch", "fake-dataset"]) == 0
    (fake_repo / ".data" / "fake-dataset" / "hello.txt").write_text("tampered\n")
    assert cli.main(["verify", "fake-dataset"]) == 1


def test_verify_unfetched_dataset_fails(fake_repo):
    assert cli.main(["verify", "fake-dataset"]) == 1


def test_fetch_unknown_dataset_fails(fake_repo):
    assert cli.main(["fetch", "does-not-exist"]) == 1

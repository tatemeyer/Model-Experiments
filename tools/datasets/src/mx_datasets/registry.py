from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    license: str
    destination: str  # path relative to repo root
    source_kind: str  # "url" | "generator"
    url: str | None = None
    checksum_sha256: str | None = None
    generator: list[str] | None = None

    @property
    def dest_path(self) -> Path:
        return repo_root() / self.destination


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repo root (no .git directory found above mx_datasets)")


def registry_dir() -> Path:
    return repo_root() / "tools" / "datasets" / "registry"


def load_registry() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    for toml_file in sorted(registry_dir().glob("*.toml")):
        data = tomllib.loads(toml_file.read_text())
        source = data["source"]
        spec = DatasetSpec(
            name=data["name"],
            description=data.get("description", ""),
            license=data.get("license", "unspecified"),
            destination=data["destination"],
            source_kind=source["kind"],
            url=source.get("url"),
            checksum_sha256=source.get("checksum_sha256"),
            generator=source.get("command"),
        )
        if spec.name in specs:
            raise ValueError(f"Duplicate dataset name '{spec.name}' in {toml_file}")
        specs[spec.name] = spec
    return specs

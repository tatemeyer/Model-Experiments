from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from mx_datasets import registry
from mx_datasets.registry import DatasetSpec

LOCK_FILE_REL = Path(".data") / ".mx-lock.json"


def _lock_path() -> Path:
    return registry.repo_root() / LOCK_FILE_REL


def _read_lock() -> dict:
    path = _lock_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_lock(lock: dict) -> None:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _fetch_url(spec: DatasetSpec) -> None:
    if not spec.url:
        raise ValueError(f"Dataset '{spec.name}' has kind=url but no url set")
    if not spec.checksum_sha256:
        print(
            f"warning: '{spec.name}' has no checksum_sha256 set; integrity is unverified",
            file=sys.stderr,
        )

    suffix = "".join(Path(spec.url).suffixes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(spec.url, tmp_path)  # noqa: S310 - registry is repo-controlled, not user input
        if spec.checksum_sha256:
            actual = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
            if actual != spec.checksum_sha256:
                raise ValueError(
                    f"Checksum mismatch for '{spec.name}': "
                    f"expected {spec.checksum_sha256}, got {actual}"
                )
        spec.dest_path.mkdir(parents=True, exist_ok=True)
        try:
            shutil.unpack_archive(str(tmp_path), str(spec.dest_path))
        except shutil.ReadError:
            shutil.move(str(tmp_path), str(spec.dest_path / Path(spec.url).name))
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_generator(spec: DatasetSpec) -> None:
    if not spec.generator:
        raise ValueError(f"Dataset '{spec.name}' has kind=generator but no command set")
    spec.dest_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(spec.generator, cwd=registry.repo_root(), check=True)


def cmd_list(_args: argparse.Namespace) -> int:
    specs = registry.load_registry()
    lock = _read_lock()
    if not specs:
        print("No datasets registered yet. Add a .toml file under tools/datasets/registry/.")
        return 0
    for name, spec in sorted(specs.items()):
        state = "fetched" if name in lock else "not fetched"
        print(f"{name:30s} [{spec.source_kind:9s}] {state:12s} {spec.description}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    specs = registry.load_registry()
    spec = specs.get(args.name)
    if spec is None:
        print(
            f"Unknown dataset '{args.name}'. Run `mx-data list` to see registered datasets.",
            file=sys.stderr,
        )
        return 1

    if spec.source_kind == "url":
        _fetch_url(spec)
    elif spec.source_kind == "generator":
        _run_generator(spec)
    else:
        print(f"Unknown source kind '{spec.source_kind}' for '{spec.name}'", file=sys.stderr)
        return 1

    lock = _read_lock()
    lock[spec.name] = {
        "content_sha256": _hash_directory(spec.dest_path),
        "fetched_at": datetime.now(UTC).isoformat(),
        "source_kind": spec.source_kind,
    }
    _write_lock(lock)
    print(f"Fetched '{spec.name}' -> {spec.destination}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    specs = registry.load_registry()
    names = [args.name] if args.name else sorted(specs)
    lock = _read_lock()
    ok = True
    for name in names:
        spec = specs.get(name)
        if spec is None:
            print(f"Unknown dataset '{name}'", file=sys.stderr)
            ok = False
            continue
        if name not in lock:
            print(f"{name}: not fetched yet")
            ok = False
            continue
        if not spec.dest_path.exists():
            print(f"{name}: MISSING ({spec.destination} does not exist)")
            ok = False
            continue
        actual = _hash_directory(spec.dest_path)
        expected = lock[name]["content_sha256"]
        if actual == expected:
            print(f"{name}: OK")
        else:
            print(f"{name}: MISMATCH (expected {expected}, got {actual})")
            ok = False
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mx-data", description="Repo-wide dataset registry and fetch tool"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="List registered datasets and their fetch status")
    list_p.set_defaults(func=cmd_list)

    fetch_p = sub.add_parser("fetch", help="Fetch (download or generate) a dataset by name")
    fetch_p.add_argument("name")
    fetch_p.set_defaults(func=cmd_fetch)

    verify_p = sub.add_parser(
        "verify", help="Verify a fetched dataset's contents against its recorded checksum"
    )
    verify_p.add_argument(
        "name", nargs="?", default=None, help="Dataset name; omit to verify all fetched datasets"
    )
    verify_p.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

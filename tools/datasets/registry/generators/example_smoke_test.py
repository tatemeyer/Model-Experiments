#!/usr/bin/env python3
"""Deterministic tiny generator exercising the mx-data fetch/verify pipeline. Not real data."""

from pathlib import Path

dest = Path(".data/example-smoke-test")
dest.mkdir(parents=True, exist_ok=True)
(dest / "hello.txt").write_text("mx-data smoke test\n")

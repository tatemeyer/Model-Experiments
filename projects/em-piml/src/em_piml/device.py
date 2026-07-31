from __future__ import annotations

import os
import re
from collections.abc import Mapping

import torch

# cpu | cuda | cuda:<digits> -- the only strings this project's device resolver accepts. Anything
# else (mps, xpu, typos) is rejected loudly rather than silently mis-selecting a device or falling
# through to torch.device()'s own, much more permissive parsing.
_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")

ENV_VAR = "EM_PIML_DEVICE"


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    # device=None (the default) is the plain-CPU path: never errors, regardless of what hardware
    # is present. Every existing training/eval call site keeps this behavior unchanged until it
    # explicitly opts in to a device argument (device-abstraction Arc, Slice 2).
    if device is None:
        return torch.device("cpu")

    if isinstance(device, torch.device):
        resolved = device
    else:
        if not _DEVICE_PATTERN.match(device):
            raise ValueError(
                f"em_piml: unrecognized device string {device!r} -- expected 'cpu', 'cuda', or "
                f"'cuda:<n>'"
            )
        resolved = torch.device(device)

    # An *explicit* request that can't be honored is a hard error, never a silent CPU fallback --
    # a silent downgrade would let results.csv record a CPU run as though it were a GPU one. See
    # the device-abstraction Arc Charter Sec5, "default behavior vs. explicit request are
    # different guarantees, not one".
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"em_piml: device {resolved} was explicitly requested but torch.cuda.is_available() "
            f"is False on this machine -- refusing to silently fall back to CPU. Either drop the "
            f"explicit device argument (defaults to CPU) or fix CUDA availability first."
        )

    # Only the explicit-request path prints -- the plain default (device=None) stays silent so it
    # doesn't spam every existing CPU-only call site once this is threaded through (Slice 2).
    print(f"em_piml: resolved device -> {resolved}")
    return resolved


def resolve_device_from_env(
    cli_value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> torch.device:
    # Thin CLI/env wrapper over resolve_device(): a CLI value always wins over the environment
    # variable. When env isn't passed explicitly, this suppresses EM_PIML_DEVICE while running
    # under pytest (PYTEST_CURRENT_TEST is set by pytest itself) so a GitHub Actions Environment
    # (or any other ambient env var) can't silently change what the CPU-path tests are actually
    # testing. A test that wants to exercise env-var behavior passes env=... explicitly, bypassing
    # the suppression by construction rather than special-casing test code.
    if env is None:
        env = {} if "PYTEST_CURRENT_TEST" in os.environ else os.environ
    value = cli_value if cli_value is not None else env.get(ENV_VAR)
    return resolve_device(value)

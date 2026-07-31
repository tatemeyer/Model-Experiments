from __future__ import annotations

import pytest
import torch
from em_piml.device import ENV_VAR, resolve_device, resolve_device_from_env

# device-abstraction Arc, Slice 1 (device-selection-module): none of these require actual GPU
# hardware -- torch.cuda.is_available() is monkeypatched wherever a "GPU present" branch is
# exercised, so this file runs unmarked (not @pytest.mark.gpu) and stays in the default fast suite.


def test_default_resolves_to_cpu_and_never_errors():
    assert resolve_device() == torch.device("cpu")


def test_explicit_cpu_request_resolves():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_explicit_cuda_without_hardware_raises(monkeypatch):
    # The core research-provenance guarantee: an explicit request that can't be honored is a hard
    # error, never a silent downgrade to CPU that could get recorded as a GPU result.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicitly requested"):
        resolve_device("cuda")


def test_explicit_cuda_with_hardware_succeeds(monkeypatch):
    # Mocked availability -- doesn't require actually running on a machine with a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda")


def test_explicit_cuda_index_succeeds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda:0") == torch.device("cuda:0")


@pytest.mark.parametrize("bad_value", ["mps", "xpu", "gpu", "cuda:", "cuda:x", "", "CPU"])
def test_invalid_device_string_rejected(bad_value):
    with pytest.raises(ValueError, match="unrecognized device string"):
        resolve_device(bad_value)


def test_env_resolver_prefers_cli_over_env(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    resolved = resolve_device_from_env("cuda", env={ENV_VAR: "cpu"})
    assert resolved == torch.device("cuda")


def test_env_resolver_uses_env_when_no_cli_value():
    resolved = resolve_device_from_env(env={ENV_VAR: "cpu"})
    assert resolved == torch.device("cpu")


def test_env_resolver_defaults_to_cpu_with_neither_cli_nor_env():
    assert resolve_device_from_env(env={}) == torch.device("cpu")


def test_env_resolver_rejects_invalid_env_value():
    with pytest.raises(ValueError, match="unrecognized device string"):
        resolve_device_from_env(env={ENV_VAR: "bogus"})


def test_env_resolver_suppresses_ambient_env_under_pytest(monkeypatch):
    # PYTEST_CURRENT_TEST is already set by pytest itself while this test runs. Setting
    # EM_PIML_DEVICE on the real os.environ (as a CI Environment variable might) and calling the
    # resolver with env=None (the real-world default) must NOT pick it up -- otherwise an
    # Environment-level env var could silently change what a CPU-path test is actually testing.
    monkeypatch.setenv(ENV_VAR, "cuda")
    assert resolve_device_from_env() == torch.device("cpu")


def test_env_resolver_explicit_env_bypasses_pytest_suppression(monkeypatch):
    # A test that wants to exercise real env-var behavior passes env=... explicitly (as the other
    # tests above do), sidestepping the pytest suppression by construction.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    resolved = resolve_device_from_env(env={ENV_VAR: "cuda"})
    assert resolved == torch.device("cuda")

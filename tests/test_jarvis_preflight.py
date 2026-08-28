"""Unit tests for the no-download JarvisLabs preflight."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import jarvis_preflight as preflight


def _mock_supported_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = dict(preflight.REAL_DISTRIBUTIONS)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight, "_python_runtime", lambda: ("CPython", 3, 10))
    monkeypatch.setattr(preflight.metadata, "version", lambda name: versions[name])


def test_mock_profile_skips_gpu_and_java_checks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mock_supported_runtime(monkeypatch)
    monkeypatch.setattr(
        preflight,
        "_check_import",
        lambda label, _module: preflight.CheckResult(f"import {label}", True, "ready"),
    )

    def unexpected() -> preflight.CheckResult:
        raise AssertionError("real-only check ran in mock profile")

    monkeypatch.setattr(preflight, "_check_torch_cuda", unexpected)
    monkeypatch.setattr(preflight, "_check_template_torch_origin", unexpected)
    monkeypatch.setattr(preflight, "_check_bitsandbytes_4bit_cuda", unexpected)
    monkeypatch.setattr(preflight, "_check_java21", unexpected)
    monkeypatch.setattr(preflight, "_check_jarvis_uv_config", unexpected)

    assert preflight.main(["--mock"]) == 0
    output = capsys.readouterr().out
    assert "mock profile skips Torch/CUDA" in output
    assert "Preflight passed" in output


def test_mock_profile_rejects_wrong_platform_and_python(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mock_supported_runtime(monkeypatch)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preflight, "_python_runtime", lambda: ("CPython", 3, 11))
    monkeypatch.setattr(
        preflight,
        "_check_import",
        lambda label, _module: preflight.CheckResult(f"import {label}", True, "ready"),
    )

    assert preflight.main(["--mock"]) == 1
    output = capsys.readouterr().out
    assert "expected Linux" in output
    assert "expected CPython 3.10" in output


def test_constraint_failure_is_actionable_and_does_not_echo_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mock_supported_runtime(monkeypatch)
    expected = dict(preflight.REAL_DISTRIBUTIONS)
    monkeypatch.setattr(
        preflight.metadata,
        "version",
        lambda name: "0.0.0" if name == "numpy" else expected[name],
    )
    monkeypatch.setattr(
        preflight,
        "_check_import",
        lambda label, _module: preflight.CheckResult(f"import {label}", True, "ready"),
    )
    secret = "do-not-echo-this-token"
    monkeypatch.setenv("HF_TOKEN", secret)

    assert preflight.main(["--mock"]) == 1
    output = capsys.readouterr().out
    assert preflight.REQUIREMENTS_PATH in output
    assert "expected 2.2.6, found 0.0.0" in output
    assert secret not in output


def test_real_profile_passes_with_mocked_hardware_and_imports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mock_supported_runtime(monkeypatch)
    monkeypatch.setattr(
        preflight,
        "_check_import",
        lambda label, _module: preflight.CheckResult(f"import {label}", True, "ready"),
    )
    monkeypatch.setattr(
        preflight,
        "_check_torch_cuda",
        lambda: preflight.CheckResult("Torch/CUDA", True, "template CUDA ready"),
    )
    monkeypatch.setattr(
        preflight,
        "_check_jarvis_uv_config",
        lambda: preflight.CheckResult("Jarvis uv exclusions", True, "template stack"),
    )
    monkeypatch.setattr(
        preflight,
        "_check_template_torch_origin",
        lambda: preflight.CheckResult("template Torch origin", True, "template inherited"),
    )
    monkeypatch.setattr(
        preflight,
        "_check_bitsandbytes_4bit_cuda",
        lambda: preflight.CheckResult("bitsandbytes 4-bit CUDA", True, "tiny forward ready"),
    )
    monkeypatch.setattr(
        preflight,
        "_check_java21",
        lambda: preflight.CheckResult("Java 21/JAVA_HOME", True, "Java 21"),
    )

    assert preflight.main([]) == 0
    output = capsys.readouterr().out
    assert "Torch/CUDA" in output
    assert "import Pyjnius" in output
    assert "Preflight passed for the real profile" in output


def test_jarvis_uv_config_requires_exact_bundle_only_exclusions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(preflight, "UV_CONFIG_PATH", tmp_path / "uv.toml")
    missing = preflight._check_jarvis_uv_config()
    assert not missing.ok
    assert "rebuild the clean Jarvis upload bundle" in missing.detail

    (tmp_path / "uv.toml").write_text(
        'exclude-dependencies = ["torch"]\n',
        encoding="utf-8",
    )
    incomplete = preflight._check_jarvis_uv_config()
    assert not incomplete.ok
    assert "exact template-stack exclusion contract" in incomplete.detail

    (tmp_path / "uv.toml").write_text(
        "# generated for Jarvis only\n" + preflight.UV_EXCLUDE_DECLARATION + "\n",
        encoding="utf-8",
    )
    exact = preflight._check_jarvis_uv_config()
    assert exact.ok
    assert exact.detail == "torch, torchvision, torchaudio, triton"


def test_committed_jarvis_uv_config_matches_preflight_constants() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    config = (repo_root / "requirements" / "jarvislabs-uv.toml").read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in config.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert active_lines == [preflight.UV_EXCLUDE_DECLARATION]
    assert tuple(re.findall(r'"([^"]+)"', preflight.UV_EXCLUDE_DECLARATION)) == (
        preflight.UV_TEMPLATE_EXCLUDES
    )


def test_real_profile_fails_when_jarvis_uv_config_is_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _mock_supported_runtime(monkeypatch)
    monkeypatch.setattr(
        preflight,
        "_check_import",
        lambda label, _module: preflight.CheckResult(f"import {label}", True, "ready"),
    )
    monkeypatch.setattr(
        preflight,
        "_check_jarvis_uv_config",
        lambda: preflight.CheckResult("Jarvis uv exclusions", False, "invalid contract"),
    )
    for name, label in (
        ("_check_torch_cuda", "Torch/CUDA"),
        ("_check_template_torch_origin", "template Torch origin"),
        ("_check_bitsandbytes_4bit_cuda", "bitsandbytes 4-bit CUDA"),
        ("_check_java21", "Java 21/JAVA_HOME"),
    ):
        monkeypatch.setattr(
            preflight,
            name,
            lambda label=label: preflight.CheckResult(label, True, "ready"),
        )

    assert preflight.main([]) == 1
    output = capsys.readouterr().out
    assert "Jarvis uv exclusions" in output
    assert "invalid contract" in output
    assert "Preflight failed" in output


def test_template_torch_origin_rejects_project_venv_shadow_without_path_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "secret-user-segment"
    virtualenv = tmp_path / secret / ".venv"
    torch_file = virtualenv / "lib" / "python3.10" / "site-packages" / "torch" / "__init__.py"
    monkeypatch.setattr(preflight.sys, "prefix", str(virtualenv))
    monkeypatch.setattr(preflight.sys, "base_prefix", str(tmp_path / "template-python"))

    def import_shadowed_torch(_name: str) -> SimpleNamespace:
        print(secret)
        return SimpleNamespace(__file__=str(torch_file))

    monkeypatch.setattr(preflight.importlib, "import_module", import_shadowed_torch)
    result = preflight._check_template_torch_origin()

    assert not result.ok
    assert "shadowed inside the project venv" in result.detail
    assert secret not in result.detail
    assert secret not in capsys.readouterr().out


def test_template_torch_origin_accepts_template_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    virtualenv = tmp_path / "project" / ".venv"
    torch_file = tmp_path / "template" / "site-packages" / "torch" / "__init__.py"
    monkeypatch.setattr(preflight.sys, "prefix", str(virtualenv))
    monkeypatch.setattr(preflight.sys, "base_prefix", str(tmp_path / "template-python"))
    monkeypatch.setattr(
        preflight.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__file__=str(torch_file)),
    )

    result = preflight._check_template_torch_origin()

    assert result.ok
    assert "inherited from outside" in result.detail


def test_bitsandbytes_probe_executes_tiny_nf4_cuda_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class FakeInferenceMode:
        def __enter__(self) -> None:
            events.append("inference-enter")

        def __exit__(self, *_args: object) -> None:
            events.append("inference-exit")

    class FakeFinite:
        def all(self) -> "FakeFinite":
            return self

        def item(self) -> bool:
            return True

    class FakeLayer:
        def __init__(self, in_features: int, out_features: int, **kwargs: object) -> None:
            events.append(("linear", in_features, out_features, kwargs))
            data = SimpleNamespace(fill_=lambda value: events.append(("fill", value)))
            self.weight = SimpleNamespace(data=data)

        def to(self, device: str) -> "FakeLayer":
            events.append(("to", device))
            return self

        def eval(self) -> None:
            events.append("eval")

        def __call__(self, _sample: object) -> SimpleNamespace:
            events.append("forward")
            return SimpleNamespace(shape=(1, 8))

    def fake_ones(shape: tuple[int, int], *, device: str, dtype: object) -> object:
        events.append(("ones", shape, device, dtype))
        return object()

    fake_torch = SimpleNamespace(
        float16="float16",
        inference_mode=lambda: FakeInferenceMode(),
        ones=fake_ones,
        cuda=SimpleNamespace(synchronize=lambda: events.append("synchronize")),
        isfinite=lambda _output: FakeFinite(),
    )
    fake_bitsandbytes = SimpleNamespace(nn=SimpleNamespace(Linear4bit=FakeLayer))
    modules = {"torch": fake_torch, "bitsandbytes": fake_bitsandbytes}
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: modules[name])

    result = preflight._check_bitsandbytes_4bit_cuda()

    assert result.ok
    assert ("to", "cuda") in events
    assert "forward" in events
    assert "synchronize" in events
    linear_event = next(
        event for event in events if isinstance(event, tuple) and event[0] == "linear"
    )
    assert linear_event[1:3] == (64, 8)
    assert linear_event[3]["quant_type"] == "nf4"


def test_bitsandbytes_probe_sanitizes_import_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "secret-from-bitsandbytes"

    def fake_import(name: str) -> SimpleNamespace:
        if name == "bitsandbytes":
            print(secret)
            raise RuntimeError(secret)
        return SimpleNamespace()

    monkeypatch.setattr(preflight.importlib, "import_module", fake_import)
    result = preflight._check_bitsandbytes_4bit_cuda()

    assert not result.ok
    assert "RuntimeError" in result.detail
    assert secret not in result.detail
    assert secret not in capsys.readouterr().out


def test_java_check_uses_java_home_and_accepts_only_java_21(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    java_binary = tmp_path / "bin" / "java"
    java_binary.parent.mkdir()
    java_binary.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr(preflight.os, "access", lambda _path, _mode: True)

    def java_version_17(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[str(java_binary), "-version"],
            returncode=0,
            stdout="",
            stderr='openjdk version "17.0.12"',
        )

    monkeypatch.setattr(preflight.subprocess, "run", java_version_17)
    result = preflight._check_java21()

    assert not result.ok
    assert "expected Java 21, found Java 17" in result.detail


def test_java_check_does_not_emit_captured_process_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    java_binary = tmp_path / "bin" / "java"
    java_binary.parent.mkdir()
    java_binary.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr(preflight.os, "access", lambda _path, _mode: True)
    secret = "secret-from-java-output"

    def java_version_21(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[str(java_binary), "-version"],
            returncode=0,
            stdout=secret,
            stderr='openjdk version "21.0.8"',
        )

    monkeypatch.setattr(preflight.subprocess, "run", java_version_21)
    result = preflight._check_java21()

    assert result.ok
    assert result.detail == "Java 21"
    assert secret not in capsys.readouterr().out

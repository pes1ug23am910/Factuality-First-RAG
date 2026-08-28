#!/usr/bin/env python3
"""Validate the local JarvisLabs runtime without downloading data or models.

The full check validates the Linux/CPython contract, exact tested Python
constraints, the Torch/CUDA stack inherited from the JarvisLabs template,
real-mode imports, and the external Java 21 runtime required by Pyserini.
``--mock`` checks only the smaller no-model mock runtime.

The script explicitly reads only ``JAVA_HOME`` from the environment. It never
enumerates the environment, prints exception text, or emits captured
third-party output.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import io
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

REQUIREMENTS_PATH: Final = "requirements/jarvislabs-cp310.txt"
PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
UV_CONFIG_PATH: Final = PROJECT_ROOT / "uv.toml"
UV_TEMPLATE_EXCLUDES: Final = ("torch", "torchvision", "torchaudio", "triton")
UV_EXCLUDE_DECLARATION: Final = (
    'exclude-dependencies = ["torch", "torchvision", "torchaudio", "triton"]'
)
JAVA_MAJOR: Final = 21

MOCK_DISTRIBUTIONS: Final = (
    ("factuality_rag", "0.4.0"),
    ("accelerate", "1.14.0"),
    ("datasets", "5.0.0"),
    ("faiss-cpu", "1.14.3"),
    ("numpy", "2.2.6"),
    ("onnxruntime", "1.23.2"),
    ("PyYAML", "6.0.3"),
    ("scikit-learn", "1.7.2"),
    ("sentence-transformers", "5.6.0"),
    ("transformers", "4.57.6"),
)

REAL_DISTRIBUTIONS: Final = MOCK_DISTRIBUTIONS + (
    ("bitsandbytes", "0.49.2"),
    ("pyjnius", "1.7.0"),
    ("pyserini", "1.2.0"),
)

MOCK_IMPORTS: Final = (
    ("project package", "factuality_rag"),
    ("Accelerate", "accelerate"),
    ("Datasets", "datasets"),
    ("FAISS", "faiss"),
    ("NumPy", "numpy"),
    ("ONNX Runtime", "onnxruntime"),
    ("PyYAML", "yaml"),
    ("scikit-learn", "sklearn"),
    ("Sentence Transformers", "sentence_transformers"),
    ("Transformers", "transformers"),
)

REAL_IMPORTS: Final = MOCK_IMPORTS + (
    ("bitsandbytes", "bitsandbytes"),
    ("Pyserini", "pyserini"),
)


@dataclass(frozen=True)
class CheckResult:
    """One preflight result suitable for deterministic, secret-free output."""

    label: str
    ok: bool
    detail: str


def _python_runtime() -> tuple[str, int, int]:
    """Return the implementation and major/minor version for easy unit mocking."""

    return platform.python_implementation(), sys.version_info.major, sys.version_info.minor


def _check_linux() -> CheckResult:
    system = platform.system()
    if system != "Linux":
        return CheckResult(
            "operating system",
            False,
            f"expected Linux, found {system or 'unknown'}; run this on the JarvisLabs instance",
        )
    return CheckResult("operating system", True, "Linux")


def _check_cpython310() -> CheckResult:
    implementation, major, minor = _python_runtime()
    if implementation != "CPython" or (major, minor) != (3, 10):
        return CheckResult(
            "Python runtime",
            False,
            f"expected CPython 3.10, found {implementation} {major}.{minor}; "
            "choose a CPython 3.10 template",
        )
    return CheckResult("Python runtime", True, "CPython 3.10")


def _check_jarvis_uv_config() -> CheckResult:
    """Require the bundle-only uv exclusions before trusting template inheritance."""

    label = "Jarvis uv exclusions"
    try:
        text = UV_CONFIG_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return CheckResult(
            label,
            False,
            "missing or unreadable uv.toml; rebuild the clean Jarvis upload bundle",
        )

    active_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if active_lines != [UV_EXCLUDE_DECLARATION]:
        return CheckResult(
            label,
            False,
            "uv.toml does not contain the exact template-stack exclusion contract",
        )
    return CheckResult(label, True, ", ".join(UV_TEMPLATE_EXCLUDES))


def _check_distribution(distribution: str, expected: str) -> CheckResult:
    label = f"distribution {distribution}"
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return CheckResult(
            label,
            False,
            f"missing; launch with --requirements {REQUIREMENTS_PATH}",
        )
    except Exception as exc:  # metadata backends can fail independently of imports
        return CheckResult(
            label,
            False,
            f"version lookup failed ({type(exc).__name__}); reinstall with {REQUIREMENTS_PATH}",
        )

    if actual != expected:
        return CheckResult(
            label,
            False,
            f"expected {expected}, found {actual}; reinstall with {REQUIREMENTS_PATH}",
        )
    return CheckResult(label, True, actual)


def _check_import(label: str, module_name: str) -> CheckResult:
    """Import a module while discarding potentially noisy third-party output."""

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(
            f"import {label}",
            False,
            f"failed ({type(exc).__name__}); reinstall with {REQUIREMENTS_PATH}",
        )
    return CheckResult(f"import {label}", True, "ready")


def _check_torch_cuda() -> CheckResult:
    """Validate, but deliberately do not version-pin, template-provided Torch."""

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch: Any = importlib.import_module("torch")
    except Exception as exc:
        return CheckResult(
            "Torch/CUDA",
            False,
            f"Torch import failed ({type(exc).__name__}); select a JarvisLabs PyTorch template",
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    version_namespace = getattr(torch, "version", None)
    cuda_runtime = getattr(version_namespace, "cuda", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda_runtime:
        return CheckResult(
            "Torch/CUDA",
            False,
            f"Torch {torch_version} has no CUDA runtime; select a CUDA-enabled PyTorch template",
        )

    try:
        available = bool(cuda.is_available())
        device_count = int(cuda.device_count())
    except Exception as exc:
        return CheckResult(
            "Torch/CUDA",
            False,
            f"CUDA probe failed ({type(exc).__name__}); verify the instance driver with nvidia-smi",
        )

    if not available or device_count < 1:
        return CheckResult(
            "Torch/CUDA",
            False,
            f"Torch {torch_version} cannot access a GPU; verify the instance with nvidia-smi",
        )
    return CheckResult(
        "Torch/CUDA",
        True,
        f"Torch {torch_version}, CUDA {cuda_runtime}, {device_count} device(s)",
    )


def _check_template_torch_origin() -> CheckResult:
    """Reject a Torch copy installed inside the managed project virtualenv."""

    if sys.prefix == sys.base_prefix:
        return CheckResult(
            "template Torch origin",
            False,
            "not running inside a managed project virtual environment; use a jl directory target",
        )

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch: Any = importlib.import_module("torch")
    except Exception as exc:
        return CheckResult(
            "template Torch origin",
            False,
            f"Torch import failed ({type(exc).__name__}); select a JarvisLabs PyTorch template",
        )

    torch_file = getattr(torch, "__file__", None)
    if not isinstance(torch_file, str) or not torch_file:
        return CheckResult(
            "template Torch origin",
            False,
            "Torch has no inspectable module path; the template inheritance cannot be verified",
        )

    try:
        torch_path = Path(torch_file).resolve()
        virtualenv_path = Path(sys.prefix).resolve()
    except OSError as exc:
        return CheckResult(
            "template Torch origin",
            False,
            f"module path inspection failed ({type(exc).__name__}); recreate the managed project venv",
        )

    if torch_path.is_relative_to(virtualenv_path):
        return CheckResult(
            "template Torch origin",
            False,
            "Torch is shadowed inside the project venv; recreate it without installing Torch locally",
        )
    return CheckResult(
        "template Torch origin",
        True,
        "Torch is inherited from outside the managed project virtual environment",
    )


def _check_bitsandbytes_4bit_cuda() -> CheckResult:
    """Exercise one tiny NF4 CUDA linear forward without loading model weights."""

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch: Any = importlib.import_module("torch")
            bitsandbytes: Any = importlib.import_module("bitsandbytes")
            with torch.inference_mode():
                layer = bitsandbytes.nn.Linear4bit(
                    64,
                    8,
                    bias=False,
                    compute_dtype=torch.float16,
                    compress_statistics=False,
                    quant_type="nf4",
                )
                layer.weight.data.fill_(0.125)
                layer = layer.to("cuda")
                layer.eval()
                sample = torch.ones((1, 64), device="cuda", dtype=torch.float16)
                output = layer(sample)
                torch.cuda.synchronize()
                shape_ok = tuple(output.shape) == (1, 8)
                finite = bool(torch.isfinite(output).all().item())
    except Exception as exc:
        return CheckResult(
            "bitsandbytes 4-bit CUDA",
            False,
            f"tiny NF4 forward failed ({type(exc).__name__}); verify Torch/CUDA/bitsandbytes compatibility",
        )

    if not shape_ok or not finite:
        return CheckResult(
            "bitsandbytes 4-bit CUDA",
            False,
            "tiny NF4 forward returned an invalid shape or non-finite values",
        )
    return CheckResult("bitsandbytes 4-bit CUDA", True, "tiny NF4 forward completed")


def _parse_java_major(output: str) -> int | None:
    patterns = (
        r"(?:openjdk|java)\s+version\s+\"?(\d+)",
        r"\bopenjdk\s+(\d+)(?:[.]|\s)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _check_java21() -> CheckResult:
    """Validate the Java binary selected explicitly by JAVA_HOME."""

    java_home = os.environ.get("JAVA_HOME")
    if not java_home or not java_home.strip():
        return CheckResult(
            "Java 21/JAVA_HOME",
            False,
            "JAVA_HOME is unset; configure JDK 21 under the persistent runtime root before launch",
        )

    java_binary = Path(java_home) / "bin" / "java"
    if not java_binary.is_file() or not os.access(java_binary, os.X_OK):
        return CheckResult(
            "Java 21/JAVA_HOME",
            False,
            "JAVA_HOME/bin/java is missing or not executable; install/configure JDK 21",
        )

    try:
        completed = subprocess.run(
            [str(java_binary), "-version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            "Java 21/JAVA_HOME",
            False,
            f"java -version failed ({type(exc).__name__}); verify the persistent JDK installation",
        )

    if completed.returncode != 0:
        return CheckResult(
            "Java 21/JAVA_HOME",
            False,
            f"java -version exited {completed.returncode}; repair JAVA_HOME before indexing",
        )

    major = _parse_java_major(f"{completed.stdout}\n{completed.stderr}")
    if major is None:
        return CheckResult(
            "Java 21/JAVA_HOME",
            False,
            "could not parse java -version; confirm that JAVA_HOME selects JDK 21",
        )
    if major != JAVA_MAJOR:
        return CheckResult(
            "Java 21/JAVA_HOME",
            False,
            f"expected Java 21, found Java {major}; update JAVA_HOME",
        )
    return CheckResult("Java 21/JAVA_HOME", True, "Java 21")


def run_checks(*, mock: bool) -> list[CheckResult]:
    """Run the requested preflight profile without network access."""

    results = [_check_linux(), _check_cpython310()]
    distributions = MOCK_DISTRIBUTIONS if mock else REAL_DISTRIBUTIONS
    results.extend(_check_distribution(name, version) for name, version in distributions)

    imports = MOCK_IMPORTS if mock else REAL_IMPORTS
    results.extend(_check_import(label, module_name) for label, module_name in imports)

    if not mock:
        results.append(_check_jarvis_uv_config())
        torch_result = _check_torch_cuda()
        results.append(torch_result)
        torch_origin_result = _check_template_torch_origin()
        results.append(torch_origin_result)
        if torch_result.ok and torch_origin_result.ok:
            results.append(_check_bitsandbytes_4bit_cuda())
        else:
            results.append(
                CheckResult(
                    "bitsandbytes 4-bit CUDA",
                    False,
                    "skipped until the Torch/CUDA and template Torch origin checks pass",
                )
            )
        java_result = _check_java21()
        results.append(java_result)
        if java_result.ok:
            results.append(_check_import("Pyjnius", "jnius"))
        else:
            results.append(
                CheckResult(
                    "import Pyjnius",
                    False,
                    "skipped until the Java 21/JAVA_HOME check passes",
                )
            )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-download JarvisLabs runtime preflight for Factuality-First RAG."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "check only the lightweight mock runtime; skip Torch/CUDA, bitsandbytes, "
            "and Java/Pyserini/Pyjnius checks"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = "mock" if args.mock else "real"
    print(f"JarvisLabs {profile} preflight (no downloads)")

    results = run_checks(mock=args.mock)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.label}: {result.detail}")

    if args.mock:
        print(
            "[NOTE] mock profile skips Torch/CUDA, bitsandbytes, Pyserini/Pyjnius, and Java checks"
        )

    failures = sum(not result.ok for result in results)
    if failures:
        print(f"Preflight failed: {failures} check(s) need attention; no downloads were attempted.")
        return 1

    print(f"Preflight passed for the {profile} profile; no downloads were attempted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

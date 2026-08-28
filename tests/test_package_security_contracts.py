"""Static publication boundaries for package manifests and target locks."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_LOCK = REPO_ROOT / "requirements" / "locks" / "windows-cp310-cu126.txt"
BUILD_LOCK = REPO_ROOT / "requirements" / "locks" / "windows-cp310-cu126-build.txt"
TORCH_URL = (
    "https://download-r2.pytorch.org/whl/cu126/torch-2.13.0%2Bcu126-cp310-cp310-win_amd64.whl"
)
TORCH_SHA256 = "349ce4cc6d6f6027ce9274fb26bc696572398ec3700e09d8b110dc39ad6a1052"
PYSERINI_SHA256 = "bc1768c49ff1df1edebd010a9492988fea2a8400f0812a53d5acc3086e50d05c"
REQUIREMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:==[^\s\\]+| @ https://[^\s\\]+)")


def _requirement_blocks(text: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    blocks: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not REQUIREMENT_RE.match(line):
            index += 1
            continue
        name = re.split(r"==| @ ", line, maxsplit=1)[0].lower().replace("_", "-")
        assert name not in blocks, f"duplicate locked requirement: {name}"
        block = [line]
        index += 1
        while index < len(lines) and lines[index].startswith("    --hash=sha256:"):
            block.append(lines[index])
            index += 1
        blocks[name] = block
    return blocks


def _assert_lock_is_public_and_hash_complete(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in (
        "--extra-index-url",
        "--trusted-host",
        "--find-links",
        "file:",
        "git+",
        "--editable",
        "../",
        "..\\",
        "/home/",
        "/users/",
        "/tmp/",
        "\\users\\",
        "appdata",
    ):
        assert forbidden not in lowered
    assert re.search(r"(?im)^[a-z]:[/\\]", text) is None

    blocks = _requirement_blocks(text)
    assert blocks
    for name, block in blocks.items():
        assert any(
            re.fullmatch(r"    --hash=sha256:[0-9a-f]{64}( \\)?", line) for line in block[1:]
        ), f"{path.name}: {name} lacks a SHA-256"
    return blocks


def test_runtime_lock_has_safe_index_and_binary_policy_order() -> None:
    lines = RUNTIME_LOCK.read_text(encoding="utf-8").splitlines()

    assert "--index-url https://pypi.org/simple" in lines
    only_binary = lines.index("--only-binary :all:")
    pyserini_source = lines.index("--no-binary pyserini")
    assert only_binary < pyserini_source


def test_runtime_lock_binds_exact_torch_and_pyserini_artifacts() -> None:
    blocks = _assert_lock_is_public_and_hash_complete(RUNTIME_LOCK)

    assert blocks["torch"] == [
        f"torch @ {TORCH_URL} \\",
        f"    --hash=sha256:{TORCH_SHA256}",
    ]
    assert blocks["pyserini"] == [
        "pyserini==1.2.0 \\",
        f"    --hash=sha256:{PYSERINI_SHA256}",
    ]
    for required in (
        "accelerate",
        "bitsandbytes",
        "datasets",
        "faiss-cpu",
        "numpy",
        "pyjnius",
        "pytest",
        "pyyaml",
        "scikit-learn",
        "sentence-transformers",
        "transformers",
    ):
        assert required in blocks


def test_build_bootstrap_lock_is_public_and_hash_complete() -> None:
    blocks = _assert_lock_is_public_and_hash_complete(BUILD_LOCK)

    assert {
        "build",
        "colorama",
        "cython",
        "packaging",
        "pip",
        "pyproject-hooks",
        "setuptools",
        "tomli",
        "truststore",
        "wheel",
    } == set(blocks)
    assert blocks["build"][0] == "build==1.5.0 \\"
    assert blocks["pyproject-hooks"][0] == "pyproject-hooks==1.2.0 \\"
    assert blocks["tomli"][0] == "tomli==2.4.1 \\"


def test_lock_inputs_use_direct_torch_binding_without_secondary_index() -> None:
    runtime_input = (
        REPO_ROOT / "requirements" / "lock-input" / "windows-cp310-cu126.in"
    ).read_text(encoding="utf-8")

    assert f"torch @ {TORCH_URL}#sha256={TORCH_SHA256}" in runtime_input
    assert "--extra-index-url" not in runtime_input
    assert "--find-links" not in runtime_input
    assert "file:" not in runtime_input
    assert re.search(r"(?im)^[a-z]:[/\\]", runtime_input) is None

    build_input = (
        REPO_ROOT / "requirements" / "lock-input" / "windows-cp310-cu126-build.in"
    ).read_text(encoding="utf-8")
    assert "build==1.5.0" in build_input.splitlines()


def test_pyproject_declares_direct_numpy_and_package_build_tools() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    runtime_section = text.split("[project.optional-dependencies]", maxsplit=1)[0]
    dev_section = text.split("dev = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"numpy>=1.26,<3"' in runtime_section
    assert '"build>=1.2,<2"' in dev_section
    assert '"wheel>=0.45,<1"' in dev_section


def test_jarvis_uv_contract_protects_template_cuda_stack() -> None:
    assert not (REPO_ROOT / "uv.toml").exists()
    config = (REPO_ROOT / "requirements" / "jarvislabs-uv.toml").read_text(encoding="utf-8")
    active_config_lines = [
        line.strip()
        for line in config.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert active_config_lines == [
        'exclude-dependencies = ["torch", "torchvision", "torchaudio", "triton"]'
    ]

    requirements = (REPO_ROOT / "requirements" / "jarvislabs-cp310.txt").read_text(encoding="utf-8")
    requirement_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert "-e .[quantization]" in requirement_lines
    assert not any(line.lower().startswith("torch") for line in requirement_lines)


def test_manifest_excludes_process_and_test_trees_but_keeps_resources() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for required in (
        "include LICENSE",
        "include README.md",
        "include pyproject.toml",
        "recursive-include factuality_rag *.py",
        "recursive-include factuality_rag/resources *.yaml *.json",
        "exclude *.md",
        "prune tests",
        "prune requirements",
        "prune .*",
        "global-exclude *.py[cod]",
    ):
        assert required in manifest.splitlines()

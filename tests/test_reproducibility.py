"""Tests for deterministic hashing and immutable run manifests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import pytest

import factuality_rag.reproducibility as reproducibility
from factuality_rag.data.splits import build_group_disjoint_split
from factuality_rag.reproducibility import (
    build_run_manifest,
    canonical_json_bytes,
    create_run_directory,
    seal_manifest,
    sha256_file,
    sha256_json,
    sha256_ordered_ids,
    validate_publication_run_manifest,
    validate_relative_artifact_path,
    verify_manifest,
    write_immutable_json,
)

_SCORER_ID = "factuality-rag.production-scorer"
_SCORER_REVISION = "c" * 40
_MODEL_REVISION = "d" * 40


def _registered_fixture_path() -> Path:
    colocated = Path(__file__).resolve().parent / "data" / "evaluator_sanity_v1.json"
    if colocated.is_file():
        return colocated
    staged = (
        Path(__file__).resolve().parent.parent
        / "eval_agent"
        / "tests"
        / "data"
        / "evaluator_sanity_v1.json"
    )
    if not staged.is_file():
        raise FileNotFoundError("registered evaluator fixture is unavailable")
    return staged


def _registered_fixture() -> Dict[str, Any]:
    return json.loads(_registered_fixture_path().read_text(encoding="utf-8"))


def _registered_results() -> list[Dict[str, Any]]:
    return [
        {"id": case["id"], **deepcopy(case["expected"])} for case in _registered_fixture()["cases"]
    ]


def test_canonical_json_hash_is_key_order_independent() -> None:
    left = {"b": [2, 1], "a": {"x": True}}
    right = {"a": {"x": True}, "b": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)


def test_manifest_seal_detects_tampering() -> None:
    sealed = seal_manifest({"schema": "test.v1", "count": 3})
    assert verify_manifest(sealed)
    tampered = dict(sealed)
    tampered["count"] = 4
    assert not verify_manifest(tampered)


def test_manifest_rejects_secret_fields_and_credential_urls() -> None:
    for sensitive_key in (
        "token",
        "hf_token",
        "githubToken",
        "aws_session_token",
        "accessToken",
        "apiKey",
        "AWSSecretAccessKey",
        "authorizationHeader",
        "client_secret",
        "credential",
    ):
        with pytest.raises(ValueError, match="Sensitive field"):
            seal_manifest({sensitive_key: "literal-secret"})
    for sensitive_url in (
        "https://user:pass@example.test/data",
        "https://user@example.test/data",
        "https://example.test/data#token=value",
        "https://example.test/data?X-Amz-Signature=value",
        "https://example.test/data?X-Goog-Credential=value",
        "https://example.test/data?GoogleAccessId=value&Signature=signed",
        "https://example.test/data?Policy=value&Key-Pair-Id=key&Signature=signed",
        "https://example.test/data?apikey=value",
        "https://example.test/data?expires=1;sig=credential",
        "ws://user@example.test/socket",
        "wss://user:pass@example.test/socket",
        "https://[malformed.example/data",
    ):
        with pytest.raises(ValueError, match="Credential-bearing URL"):
            seal_manifest({"source_url": sensitive_url})


@pytest.mark.parametrize(
    "value",
    [
        "ghp_placeholder",
        "github_pat_placeholder",
        "glpat-placeholder",
        "xoxb-placeholder-value",
    ],
)
def test_manifest_rejects_known_token_value_shapes_without_echoing_value(value: str) -> None:
    with pytest.raises(ValueError, match="Token-like secret value") as caught:
        seal_manifest({"note": value})
    assert value not in str(caught.value)


@pytest.mark.parametrize(
    "key",
    [
        "ghp_placeholder",
        "https://user:pass@example.test/data",
        "wss://example.test/socket?apikey=value",
    ],
)
def test_manifest_scans_mapping_keys_without_echoing_secret_bearing_key(key: str) -> None:
    with pytest.raises(ValueError, match="Secret-bearing mapping key") as caught:
        seal_manifest({key: "redacted"})
    assert key not in str(caught.value)


def test_secret_scan_allows_explicit_noncredential_token_accounting_keys() -> None:
    payload = {
        "token_count": 10,
        "token_budget": 20,
        "max_tokens": 30,
        "source_url": "https://example.test/data?token_count=10",
    }
    assert verify_manifest(seal_manifest(payload))


def test_immutable_json_is_idempotent_but_rejects_overwrite(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    payload = seal_manifest({"schema": "test.v1", "value": 1})
    assert write_immutable_json(path, payload) == "created"
    assert write_immutable_json(path, payload) == "unchanged"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_immutable_json(path, seal_manifest({"schema": "test.v1", "value": 2}))
    assert sha256_file(path) == sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_sha256_file_rejects_nonpositive_chunk_size(tmp_path, chunk_size: int) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(path, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [True, 1.5, "1024"])
def test_sha256_file_rejects_chunk_size_type_confusion(tmp_path, chunk_size: Any) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"artifact")
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        sha256_file(path, chunk_size=chunk_size)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/output.json",
        "../escape.json",
        "runs/../../escape",
        "C:/drive.json",
        "",
        ".",
        "bad\npath.json",
        " padded.json",
        "padded.json ",
        "nested//artifact.json",
        "nested/./artifact.json",
        "nested/colon:name.json",
        "CON",
        "nested/aux.txt",
        "nested/COM1.json",
        "nested/name./artifact.json",
        "nested/name /artifact.json",
    ],
)
def test_artifact_paths_must_be_relative(path: str) -> None:
    with pytest.raises(ValueError, match="Artifact path"):
        validate_relative_artifact_path(path)


def test_ordered_id_hash_rejects_missing_or_duplicate_ids() -> None:
    assert sha256_ordered_ids(["one", "two"]) != sha256_ordered_ids(["two", "one"])
    with pytest.raises(ValueError, match="non-empty"):
        sha256_ordered_ids([""])
    with pytest.raises(ValueError, match="unique"):
        sha256_ordered_ids(["same", "same"])
    with pytest.raises(ValueError, match="unpadded"):
        sha256_ordered_ids([" padded"])
    with pytest.raises(TypeError, match="strings"):
        sha256_ordered_ids([1])  # type: ignore[list-item]


def test_run_directory_creation_fails_on_collision(tmp_path) -> None:
    created = create_run_directory(tmp_path / "runs", "smoke-001")
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path / "runs", "smoke-001")


@pytest.mark.parametrize(
    "run_id, message",
    [
        ("run.", "must not end with a dot or space"),
        ("CON", "reserved Windows device"),
        ("con.txt", "reserved Windows device"),
        ("COM1", "reserved Windows device"),
        ("LPT9.log", "reserved Windows device"),
    ],
)
def test_run_directory_rejects_windows_alias_run_ids(
    tmp_path: Path, run_id: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        create_run_directory(tmp_path / "runs", run_id)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _fixture_split() -> Dict[str, Any]:
    examples = [
        {
            "example_id": f"fixture:{index:02d}",
            "family_ids": [f"family:{index:02d}"],
            "source": "fixture",
        }
        for index in range(15)
    ]
    return build_group_disjoint_split(
        examples,
        ratios={"train": 0.6, "tuning": 0.2, "sealed_final": 0.2},
        seed=42,
    )


def _fixture_corpus_manifest() -> Dict[str, Any]:
    return seal_manifest(
        {
            "schema": "factuality-rag.corpus-manifest.v1",
            "corpus_id": "fixture-corpus",
            "corpus_snapshot_sha256": "e" * 64,
            "passage_count": 10000,
        }
    )


def _fixture_index_manifest(corpus_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    corpus_manifest_text = _json_text(corpus_manifest)
    return seal_manifest(
        {
            "schema": "factuality-rag.index-manifest.v1",
            "index_id": "fixture-index",
            "corpus_manifest_sha256": hashlib.sha256(
                corpus_manifest_text.encode("utf-8")
            ).hexdigest(),
            "corpus_snapshot_sha256": corpus_manifest["corpus_snapshot_sha256"],
            "exact_indexed_passage_count": corpus_manifest["passage_count"],
        }
    )


def _run_manifest(**overrides: Any) -> Dict[str, Any]:
    artifact_contents = dict(overrides.pop("artifact_contents", {}))
    report_overrides = dict(overrides.pop("report_overrides", {}))
    split_mutator: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = overrides.pop(
        "split_mutator", None
    )
    with tempfile.TemporaryDirectory() as temporary_root:
        root = Path(temporary_root)
        paths = {
            "config_path": "configs/smoke.yaml",
            "dataset_manifest_path": "manifests/dataset.json",
            "split_manifest_path": "manifests/split.json",
            "corpus_manifest_path": "manifests/corpus.json",
            "index_manifest_path": "manifests/index.json",
            "evaluator_fixture_path": "tests/data/evaluator_sanity_v1.json",
        }
        split = _fixture_split()
        if split_mutator is not None:
            split = split_mutator(deepcopy(split))
        evaluator_fixture = _registered_fixture()
        dataset_manifest = seal_manifest(
            {
                "schema": "factuality-rag.dataset-manifest.v1",
                "dataset_id": "fixture-dataset",
                "source_snapshot_sha256": split["source"]["source_snapshot_sha256"],
                "example_count": split["source"]["example_count"],
            }
        )
        corpus_manifest = _fixture_corpus_manifest()
        corpus_manifest_text = _json_text(corpus_manifest)
        index_manifest = _fixture_index_manifest(corpus_manifest)
        defaults: Dict[str, Any] = {
            "config_path": _json_text({"seed": 42, "route": "R1"}),
            "dataset_manifest_path": _json_text(dataset_manifest),
            "split_manifest_path": _json_text(split),
            "corpus_manifest_path": corpus_manifest_text,
            "index_manifest_path": _json_text(index_manifest),
            "evaluator_fixture_path": _json_text(evaluator_fixture),
        }
        defaults.update(artifact_contents)
        for name, relative_path in paths.items():
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            content = defaults[name]
            path.write_text(
                content if isinstance(content, str) else _json_text(content),
                encoding="utf-8",
            )

        fixture_path = root / paths["evaluator_fixture_path"]
        try:
            fixture_content_sha256 = sha256_json(
                json.loads(fixture_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, ValueError):
            fixture_content_sha256 = "0" * 64
        report: Dict[str, Any] = {
            "passed": True,
            "production_gate_passed": True,
            "atol": 1e-12,
            "scorer_id": _SCORER_ID,
            "scorer_revision": _SCORER_REVISION,
            "schema_version": "evaluator-sanity-v1",
            "fixture_sha256": sha256_file(fixture_path),
            "fixture_content_sha256": fixture_content_sha256,
            "case_count": len(evaluator_fixture["cases"]),
            "results": _registered_results(),
        }
        report.update(report_overrides)

        requested_partition = overrides.get("split_partition", "tuning")
        selection_partition = (
            requested_partition
            if isinstance(requested_partition, str) and requested_partition in split["partitions"]
            else "tuning"
        )
        default_selected_ids = list(split["partitions"][selection_partition]["example_ids"][:2])
        values: Dict[str, Any] = {
            "run_id": "smoke-001",
            "run_kind": "smoke",
            "git_commit": "b" * 40,
            "git_dirty": True,
            "artifact_root": root,
            "config": {"seed": 42, "route": "R1"},
            **paths,
            "split_partition": requested_partition,
            "selected_example_ids": default_selected_ids,
            "exact_indexed_passage_count": 10000,
            "evaluator_fixture_report": report,
            "model_revisions": {
                "example/model": _MODEL_REVISION,
                _SCORER_ID: _SCORER_REVISION,
            },
            "seed": 42,
            "hardware": {"gpu": "fixture"},
            "software": {"python": "3.10"},
            "resource_ceilings": {"gpu_hours": 1, "token_count": 1000},
            "output_paths": ["raw/predictions.jsonl", "metrics.json"],
            "mock": True,
            "created_at_utc": "2026-08-18T00:00:00+00:00",
        }
        values.update(overrides)
        return build_run_manifest(**values)


def test_build_run_manifest_records_hashes_and_verifies() -> None:
    manifest = _run_manifest()
    assert verify_manifest(manifest)
    assert manifest["schema"] == "factuality-rag.run-manifest.v1"
    assert manifest["git"] == {"commit": "b" * 40, "dirty": True}
    assert manifest["config_sha256"] == sha256_json({"seed": 42, "route": "R1"})
    assert manifest["data"]["selected_example_count"] == 2
    assert manifest["retrieval"]["exact_indexed_passage_count"] == 10000
    assert manifest["evaluator"]["production_gate_passed"] is True
    assert manifest["evaluator"]["fixture_report"]["scorer_revision"] == _SCORER_REVISION
    assert manifest["evaluator"]["fixture_report_sha256"] == sha256_json(
        manifest["evaluator"]["fixture_report"]
    )
    assert manifest["output_paths"] == ["raw/predictions.jsonl", "metrics.json"]


def test_final_run_requires_clean_full_revision() -> None:
    with pytest.raises(ValueError, match="clean Git revision"):
        _run_manifest(run_kind="final", git_dirty=True, mock=False, split_partition="sealed_final")
    with pytest.raises(ValueError, match="exactly 40 or 64 lowercase"):
        _run_manifest(git_commit="abc123")
    with pytest.raises(ValueError, match="exactly 40 or 64 lowercase"):
        _run_manifest(git_commit="A" * 40)
    with pytest.raises(ValueError, match="exactly 40 or 64 lowercase"):
        _run_manifest(git_commit="a" * 41)
    with pytest.raises(ValueError, match="all-zero"):
        _run_manifest(git_commit="0" * 40)


@pytest.mark.parametrize(
    "run_id, message",
    [
        ("smoke.", "must not end with a dot or space"),
        ("NUL", "reserved Windows device"),
        ("aux.json", "reserved Windows device"),
        ("COM9.run", "reserved Windows device"),
    ],
)
def test_run_manifest_rejects_windows_alias_run_ids(run_id: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _run_manifest(run_id=run_id)


def test_final_and_tuning_partition_firewalls() -> None:
    with pytest.raises(ValueError, match="mock components"):
        _run_manifest(run_kind="final", git_dirty=False, mock=True, split_partition="sealed_final")
    with pytest.raises(ValueError, match="sealed_final"):
        _run_manifest(run_kind="tuning", split_partition="sealed_final")
    final = _run_manifest(
        run_kind="final", git_dirty=False, mock=False, split_partition="sealed_final"
    )
    assert verify_manifest(final)


def test_publication_validator_accepts_only_clean_sealed_final_runs() -> None:
    final = _run_manifest(
        run_kind="final", git_dirty=False, mock=False, split_partition="sealed_final"
    )

    validated = validate_publication_run_manifest(final, expected_run_id="smoke-001")

    assert validated == final
    assert validated is not final


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda value: value.update(run_kind="pilot"), "run_kind='final'"),
        (lambda value: value["git"].update(dirty=True), "git.dirty=false"),
        (lambda value: value["data"].update(split_partition="tuning"), "sealed_final"),
        (
            lambda value: value["evaluator"].update(production_gate_passed=False),
            "evaluator.production_gate_passed=true",
        ),
        (
            lambda value: value["model_revisions"].update({"example/model": "main"}),
            "exactly 40 or 64 lowercase",
        ),
    ],
)
def test_publication_validator_rejects_resealed_ineligible_runs(
    mutator: Callable[[Dict[str, Any]], None], message: str
) -> None:
    final = _run_manifest(
        run_kind="final", git_dirty=False, mock=False, split_partition="sealed_final"
    )
    plain = deepcopy(final)
    plain.pop("manifest_sha256")
    mutator(plain)
    resealed = seal_manifest(plain)

    with pytest.raises(ValueError, match=message):
        validate_publication_run_manifest(resealed)


@pytest.mark.parametrize(
    "report_overrides, message",
    [
        ({"passed": False}, "sanity gate must pass"),
        ({"production_gate_passed": False}, "sanity gate must pass"),
        ({"atol": True}, "atol must be numeric"),
        ({"atol": float("nan")}, "finite and between"),
        ({"atol": -1e-13}, "finite and between"),
        ({"atol": 1e-9}, "finite and between"),
        ({"fixture_sha256": "0" * 64}, "actual fixture bytes"),
        ({"fixture_content_sha256": "0" * 64}, "canonical fixture content"),
        ({"scorer_id": " scorer"}, "stable identifier"),
        ({"scorer_revision": "latest"}, "exactly 40 or 64 lowercase"),
        ({"scorer_revision": "A" * 40}, "exactly 40 or 64 lowercase"),
        ({"case_count": True}, "case_count must be an integer"),
        ({"case_count": 2}, "registered fixture"),
    ],
)
def test_evaluator_fixture_report_is_strictly_validated(
    report_overrides: Mapping[str, Any], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _run_manifest(report_overrides=report_overrides)


def test_evaluator_fixture_report_rejects_extra_schema_fields() -> None:
    with pytest.raises(ValueError, match="exactly the versioned report schema"):
        _run_manifest(report_overrides={"unexpected": True})


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_id", "requires a non-empty"),
        ("false_passed", "exact registered result schema"),
        ("wrong_oracle", "registered oracle"),
        ("wrong_case_id", "registered fixture case"),
        ("missing_result", "case_count must match results"),
    ],
)
def test_evaluator_report_results_must_exactly_match_registered_oracles(
    mutation: str, message: str
) -> None:
    results = _registered_results()
    if mutation == "missing_id":
        results[0].pop("id")
    elif mutation == "false_passed":
        results[0]["passed"] = False
    elif mutation == "wrong_oracle":
        results[0]["supported_claim_count"] = 0
    elif mutation == "wrong_case_id":
        results[0]["id"] = "case-999"
    else:
        results.pop()
    with pytest.raises(ValueError, match=message):
        _run_manifest(report_overrides={"results": results})


def test_evaluator_scorer_revision_must_be_bound_in_component_revisions() -> None:
    with pytest.raises(ValueError, match="must be bound in model_revisions"):
        _run_manifest(model_revisions={"example/model": _MODEL_REVISION})
    with pytest.raises(ValueError, match="must be bound in model_revisions"):
        _run_manifest(
            model_revisions={
                "example/model": _MODEL_REVISION,
                _SCORER_ID: "e" * 40,
            }
        )


def test_all_nonfinal_run_kinds_are_denied_sealed_final() -> None:
    for run_kind in ("smoke", "pilot", "tuning", "ablation"):
        with pytest.raises(ValueError, match="only final runs"):
            _run_manifest(run_kind=run_kind, split_partition="sealed_final")


def test_run_manifest_requires_positive_counts_outputs_and_utc_time() -> None:
    with pytest.raises(ValueError, match="selected_example_ids must be non-empty"):
        _run_manifest(selected_example_ids=[])
    with pytest.raises(ValueError, match="exact_indexed_passage_count must be positive"):
        _run_manifest(exact_indexed_passage_count=0)
    with pytest.raises(ValueError, match="output_paths must be non-empty"):
        _run_manifest(output_paths=[])
    with pytest.raises(ValueError, match="UTC timestamp"):
        _run_manifest(created_at_utc="2026-08-18T00:00:00")


def test_run_manifest_binds_artifact_bytes_and_rejects_missing_artifacts() -> None:
    original = _run_manifest()
    split = _fixture_split()
    changed_dataset = seal_manifest(
        {
            "schema": "factuality-rag.dataset-manifest.v1",
            "dataset_id": "changed-dataset",
            "source_snapshot_sha256": split["source"]["source_snapshot_sha256"],
            "example_count": split["source"]["example_count"],
        }
    )
    changed = _run_manifest(artifact_contents={"dataset_manifest_path": changed_dataset})
    assert original["data"]["dataset_manifest_sha256"] != changed["data"]["dataset_manifest_sha256"]
    with pytest.raises(FileNotFoundError):
        _run_manifest(dataset_manifest_path="manifests/missing.json")


@pytest.mark.parametrize(
    "artifact_name, manifest_section, digest_field",
    [
        ("dataset_manifest_path", "data", "dataset_manifest_sha256"),
        ("evaluator_fixture_path", "evaluator", "fixture_sha256"),
    ],
)
def test_run_manifest_hashes_and_validates_each_exact_single_read_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    manifest_section: str,
    digest_field: str,
) -> None:
    original_read = reproducibility._read_bound_artifact
    captured: Dict[str, bytes] = {}

    def replace_after_snapshot(root: Path, relative_path: str) -> bytes:
        snapshot = original_read(root, relative_path)
        if (
            relative_path
            == {
                "dataset_manifest_path": "manifests/dataset.json",
                "evaluator_fixture_path": "tests/data/evaluator_sanity_v1.json",
            }[artifact_name]
        ):
            captured[artifact_name] = snapshot
            (root / relative_path).write_bytes(b"{}")
        return snapshot

    monkeypatch.setattr(reproducibility, "_read_bound_artifact", replace_after_snapshot)
    manifest = _run_manifest()
    assert (
        manifest[manifest_section][digest_field]
        == hashlib.sha256(captured[artifact_name]).hexdigest()
    )


def test_config_file_is_strict_and_exactly_matches_runtime_config_and_seed() -> None:
    with pytest.raises(ValueError, match="strict UTF-8 YAML"):
        _run_manifest(artifact_contents={"config_path": "seed: 42\nroute: R1\nseed: 42\n"})
    with pytest.raises(ValueError, match="strict UTF-8 YAML"):
        _run_manifest(
            artifact_contents={"config_path": "base: &base\n  seed: 42\n<<: *base\nroute: R1\n"}
        )
    with pytest.raises(ValueError, match="exactly match"):
        _run_manifest(artifact_contents={"config_path": {"seed": 42, "route": "R2"}})
    with pytest.raises(ValueError, match="exactly match"):
        _run_manifest(
            config={"seed": 42, "route": True},
            artifact_contents={"config_path": {"seed": 42, "route": 1}},
        )
    with pytest.raises(ValueError, match="config seed must exactly match"):
        _run_manifest(seed=43)
    with pytest.raises(ValueError, match="config seed must be an integer"):
        _run_manifest(
            config={"seed": True, "route": "R1"},
            artifact_contents={"config_path": {"seed": True, "route": "R1"}},
        )


@pytest.mark.parametrize(
    "artifact_name, schema, manifest_name",
    [
        (
            "dataset_manifest_path",
            "factuality-rag.dataset-manifest.v1",
            "dataset manifest",
        ),
        (
            "corpus_manifest_path",
            "factuality-rag.corpus-manifest.v1",
            "corpus manifest",
        ),
        (
            "index_manifest_path",
            "factuality-rag.index-manifest.v1",
            "index manifest",
        ),
    ],
)
def test_data_artifact_manifests_require_stable_schemas_and_valid_seals(
    artifact_name: str, schema: str, manifest_name: str
) -> None:
    fields: Dict[str, Any] = {"schema": schema, "artifact_id": "fixture"}
    if artifact_name == "index_manifest_path":
        fields["exact_indexed_passage_count"] = 10000
    with pytest.raises(ValueError, match=f"{manifest_name} seal is invalid"):
        _run_manifest(artifact_contents={artifact_name: fields})

    wrong_schema = dict(fields)
    wrong_schema["schema"] = "mutable-schema"
    with pytest.raises(ValueError, match="must use stable schema"):
        _run_manifest(artifact_contents={artifact_name: seal_manifest(wrong_schema)})

    with pytest.raises(ValueError, match="fields do not match its exact schema"):
        _run_manifest(artifact_contents={artifact_name: seal_manifest({"schema": schema})})

    tampered = seal_manifest(fields)
    tampered["artifact_id"] = "tampered"
    with pytest.raises(ValueError, match=f"{manifest_name} seal is invalid"):
        _run_manifest(artifact_contents={artifact_name: tampered})


def test_data_artifact_manifests_reject_duplicate_json_keys() -> None:
    duplicate = (
        '{"schema":"factuality-rag.dataset-manifest.v1",'
        '"schema":"factuality-rag.dataset-manifest.v1",'
        '"manifest_sha256":"' + "0" * 64 + '"}'
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _run_manifest(artifact_contents={"dataset_manifest_path": duplicate})


def test_dataset_manifest_is_cross_bound_to_audited_split_source() -> None:
    split = _fixture_split()
    mismatched_snapshot = seal_manifest(
        {
            "schema": "factuality-rag.dataset-manifest.v1",
            "dataset_id": "fixture-dataset",
            "source_snapshot_sha256": "a" * 64,
            "example_count": split["source"]["example_count"],
        }
    )
    with pytest.raises(ValueError, match="audited split source snapshot"):
        _run_manifest(artifact_contents={"dataset_manifest_path": mismatched_snapshot})

    mismatched_count = seal_manifest(
        {
            "schema": "factuality-rag.dataset-manifest.v1",
            "dataset_id": "fixture-dataset",
            "source_snapshot_sha256": split["source"]["source_snapshot_sha256"],
            "example_count": split["source"]["example_count"] + 1,
        }
    )
    with pytest.raises(ValueError, match="audited split source snapshot"):
        _run_manifest(artifact_contents={"dataset_manifest_path": mismatched_count})


@pytest.mark.parametrize(
    "source_field",
    ["source_snapshot_sha256", "normalized_examples_sha256"],
)
def test_run_manifest_rejects_resealed_split_zero_digest_placeholders(
    source_field: str,
) -> None:
    def zero_split_source(split: Dict[str, Any]) -> Dict[str, Any]:
        plain = dict(split)
        plain.pop("manifest_sha256")
        plain["source"] = dict(plain["source"])
        plain["source"][source_field] = "0" * 64
        return seal_manifest(plain)

    with pytest.raises(ValueError, match="all-zero|split manifest audit failed"):
        _run_manifest(split_mutator=zero_split_source)


def test_run_manifest_rejects_zero_split_manifest_seal() -> None:
    def zero_split_seal(split: Dict[str, Any]) -> Dict[str, Any]:
        split["manifest_sha256"] = "0" * 64
        return split

    with pytest.raises(ValueError, match="split manifest seal is invalid"):
        _run_manifest(split_mutator=zero_split_seal)


def test_run_manifest_rejects_zero_corpus_digest_even_when_index_matches() -> None:
    corpus_manifest = seal_manifest(
        {
            "schema": "factuality-rag.corpus-manifest.v1",
            "corpus_id": "fixture-corpus",
            "corpus_snapshot_sha256": "0" * 64,
            "passage_count": 10000,
        }
    )
    index_manifest = _fixture_index_manifest(corpus_manifest)
    with pytest.raises(ValueError, match="all-zero SHA-256 placeholder"):
        _run_manifest(
            artifact_contents={
                "corpus_manifest_path": corpus_manifest,
                "index_manifest_path": index_manifest,
            }
        )


def test_run_manifest_rejects_zero_index_binding_digest() -> None:
    corpus_manifest = _fixture_corpus_manifest()
    index_manifest = _fixture_index_manifest(corpus_manifest)
    plain = dict(index_manifest)
    plain.pop("manifest_sha256")
    plain["corpus_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="all-zero SHA-256 placeholder"):
        _run_manifest(artifact_contents={"index_manifest_path": seal_manifest(plain)})


@pytest.mark.parametrize(
    "field, value",
    [
        ("corpus_manifest_sha256", "a" * 64),
        ("corpus_snapshot_sha256", "b" * 64),
        ("exact_indexed_passage_count", 9999),
    ],
)
def test_index_manifest_is_cross_bound_to_exact_corpus_manifest(field: str, value: Any) -> None:
    corpus_manifest = _fixture_corpus_manifest()
    index_manifest = _fixture_index_manifest(corpus_manifest)
    plain = dict(index_manifest)
    plain.pop("manifest_sha256")
    plain[field] = value
    overrides: Dict[str, Any] = {"artifact_contents": {"index_manifest_path": seal_manifest(plain)}}
    if field == "exact_indexed_passage_count":
        overrides["exact_indexed_passage_count"] = value
    with pytest.raises(ValueError, match="bound corpus manifest"):
        _run_manifest(**overrides)


def test_run_manifest_binds_selected_ids_to_the_audited_partition() -> None:
    with pytest.raises(ValueError, match="outside split_partition"):
        _run_manifest(selected_example_ids=["fixture:not-in-tuning"])


def test_run_manifest_rejects_tampered_split_even_if_resealed() -> None:
    def mutate(split: Dict[str, Any]) -> Dict[str, Any]:
        plain = dict(split)
        plain.pop("manifest_sha256")
        plain["audit"] = dict(plain["audit"])
        plain["audit"]["all_examples_assigned_once"] = False
        return seal_manifest(plain)

    with pytest.raises(ValueError, match="split manifest audit failed"):
        _run_manifest(split_mutator=mutate)


def test_run_manifest_binds_indexed_count_to_index_manifest() -> None:
    with pytest.raises(ValueError, match="does not match the actual index manifest"):
        _run_manifest(exact_indexed_passage_count=9999)
    with pytest.raises(ValueError, match="positive integer"):
        corpus_manifest = _fixture_corpus_manifest()
        index_manifest = _fixture_index_manifest(corpus_manifest)
        plain_index = dict(index_manifest)
        plain_index.pop("manifest_sha256")
        plain_index["exact_indexed_passage_count"] = True
        _run_manifest(artifact_contents={"index_manifest_path": seal_manifest(plain_index)})


def test_run_manifest_rejects_non_strict_json_artifacts() -> None:
    duplicate_fixture = '{"schema_version":"evaluator-sanity-v1","cases":[],"cases":[]}'
    with pytest.raises(ValueError, match="duplicate JSON object key|strict UTF-8 JSON"):
        _run_manifest(artifact_contents={"evaluator_fixture_path": duplicate_fixture})


def test_run_manifest_rejects_missing_model_revision() -> None:
    with pytest.raises(ValueError, match="model_revisions"):
        _run_manifest(model_revisions={})
    with pytest.raises(ValueError, match="exactly 40 or 64 lowercase"):
        _run_manifest(model_revisions={"example/model": "latest"})
    with pytest.raises(ValueError, match="exactly 40 or 64 lowercase"):
        _run_manifest(
            model_revisions={
                "example/model": "A" * 40,
                _SCORER_ID: _SCORER_REVISION,
            }
        )
    with pytest.raises(ValueError, match="all-zero"):
        _run_manifest(
            model_revisions={
                "example/model": "0" * 40,
                _SCORER_ID: _SCORER_REVISION,
            }
        )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"run_id": True}, "run_id must be a string"),
        ({"run_kind": []}, "run_kind must be a string"),
        ({"git_commit": True}, "git_commit must be a string"),
        ({"git_dirty": 0}, "git_dirty must be boolean"),
        ({"mock": 0}, "mock must be boolean"),
        ({"split_partition": 1}, "split_partition must be a string"),
        ({"selected_example_ids": True}, "selected_example_ids must be an ordered"),
        ({"selected_example_ids": [1]}, "example IDs must be strings"),
        ({"exact_indexed_passage_count": True}, "must be an integer"),
        ({"seed": "42"}, "seed must be an integer"),
        ({"output_paths": "out"}, "output_paths must be an ordered"),
        ({"artifact_root": True}, "artifact_root must be a path"),
        ({"config": []}, "config must be a mapping"),
        ({"software": {"python": 3.10}}, "software values"),
        ({"created_at_utc": True}, "created_at_utc must be a string"),
    ],
)
def test_run_manifest_rejects_type_confusion(override: Mapping[str, Any], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _run_manifest(**override)


def test_run_manifest_rejects_padded_or_overlapping_paths() -> None:
    with pytest.raises(ValueError, match="unpadded"):
        _run_manifest(config_path=" configs/smoke.yaml")
    with pytest.raises(ValueError, match="must be unique"):
        _run_manifest(dataset_manifest_path="configs/smoke.yaml")
    with pytest.raises(ValueError, match="must not overwrite"):
        _run_manifest(output_paths=["manifests/index.json"])
    with pytest.raises(ValueError, match="case-insensitively"):
        _run_manifest(dataset_manifest_path="CONFIGS/SMOKE.yaml")
    with pytest.raises(ValueError, match="case-insensitive duplicates"):
        _run_manifest(output_paths=["results.json", "RESULTS.JSON"])
    with pytest.raises(ValueError, match="must not overwrite"):
        _run_manifest(output_paths=["MANIFESTS/INDEX.JSON"])
    with pytest.raises(ValueError, match="must not end with a dot or space"):
        _run_manifest(output_paths=["MANIFESTS/INDEX.JSON."])
    with pytest.raises(ValueError, match="reserved Windows device"):
        _run_manifest(output_paths=["results/CON.txt"])

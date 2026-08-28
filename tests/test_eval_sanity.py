"""Regression tests for the sealed evaluator sanity fixture."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pytest

from factuality_rag.eval.sanity import (
    EVALUATOR_SANITY_V1_CONTENT_SHA256,
    FixtureExpectationError,
    FixtureFormatError,
    FixtureIntegrityError,
    load_sanity_fixture,
    load_sanity_fixture_bytes,
    score_sanity_case,
    validate_evaluator_fixture,
)
from factuality_rag.reproducibility import sha256_file


FIXTURE_PATH = Path(__file__).parent / "data" / "evaluator_sanity_v1.json"
SCORER_REVISION = "a" * 40
SCORER_INPUT_KEYS = {
    "id",
    "answer",
    "abstained",
    "claims",
    "reference_claims",
    "evidence",
}


def _score_input_only(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic fixture scorer using only production-visible raw inputs."""

    claims = case["claims"]
    if case["abstained"]:
        return {
            "abstained": True,
            "claim_count": 0,
            "correct_claim_count": 0,
            "supported_claim_count": 0,
            "cited_claim_count": 0,
            "correctness": None,
            "evidence_support": None,
            "citation_coverage": None,
            "fully_correct": None,
            "fully_supported": None,
        }

    reference_claims = set(case["reference_claims"])
    evidence_text = {item["id"]: item["text"] for item in case["evidence"]}
    claim_count = len(claims)
    correct_claim_count = sum(claim["text"] in reference_claims for claim in claims)
    cited_claim_count = sum(bool(claim["citation_ids"]) for claim in claims)
    supported_claim_count = sum(
        any(evidence_text[citation_id] == claim["text"] for citation_id in claim["citation_ids"])
        for claim in claims
    )
    return {
        "abstained": False,
        "claim_count": claim_count,
        "correct_claim_count": correct_claim_count,
        "supported_claim_count": supported_claim_count,
        "cited_claim_count": cited_claim_count,
        "correctness": correct_claim_count / claim_count,
        "evidence_support": supported_claim_count / claim_count,
        "citation_coverage": cited_claim_count / claim_count,
        "fully_correct": correct_claim_count == claim_count,
        "fully_supported": supported_claim_count == claim_count,
    }


def _write_fixture(path: Path, value: Mapping[str, Any]) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_canonical_fixture_is_sealed_and_matches_all_oracles() -> None:
    report = validate_evaluator_fixture(FIXTURE_PATH)

    assert report["passed"] is True
    assert report["production_gate_passed"] is False
    assert report["fixture_sha256"] == sha256_file(FIXTURE_PATH)
    assert report["fixture_content_sha256"] == EVALUATOR_SANITY_V1_CONTENT_SHA256
    assert report["atol"] == 1e-12
    assert report["case_count"] == 6
    assert {result["id"] for result in report["results"]} == {
        "case-001",
        "case-002",
        "case-003",
        "case-004",
        "case-005",
        "case-006",
    }


def test_already_read_fixture_bytes_can_be_validated_without_path_reread() -> None:
    fixture = load_sanity_fixture_bytes(FIXTURE_PATH.read_bytes())

    assert fixture["schema_version"] == "evaluator-sanity-v1"
    assert [case["id"] for case in fixture["cases"]] == [
        "case-001",
        "case-002",
        "case-003",
        "case-004",
        "case-005",
        "case-006",
    ]


@pytest.mark.parametrize("not_bytes", ["fixture", bytearray(b"{}"), memoryview(b"{}")])
def test_exact_snapshot_validator_rejects_non_bytes(not_bytes: object) -> None:
    with pytest.raises(TypeError, match="raw_bytes must be bytes"):
        load_sanity_fixture_bytes(not_bytes)  # type: ignore[arg-type]


def test_correctness_and_evidence_support_are_independent() -> None:
    fixture = load_sanity_fixture(FIXTURE_PATH)
    results = {case["category"]: score_sanity_case(case) for case in fixture["cases"]}

    assert results["correct_unsupported"]["correctness"] == 1.0
    assert results["correct_unsupported"]["evidence_support"] == 0.0
    assert results["wrong_supported"]["correctness"] == 0.0
    assert results["wrong_supported"]["evidence_support"] == 1.0
    assert results["partial_support"]["correctness"] == 1.0
    assert results["partial_support"]["evidence_support"] == 0.5


def test_uncited_nonempty_claim_fails_closed_as_unsupported() -> None:
    fixture = load_sanity_fixture(FIXTURE_PATH)
    case = next(case for case in fixture["cases"] if case["category"] == "uncited_nonempty")

    result = score_sanity_case(case)

    assert case["oracle"]["claims"][0]["entailing_evidence_ids"]
    assert case["claims"][0]["citation_ids"] == []
    assert result["correctness"] == 1.0
    assert result["citation_coverage"] == 0.0
    assert result["evidence_support"] == 0.0
    assert result["fully_supported"] is False


def test_abstention_does_not_become_a_zero_quality_answer() -> None:
    fixture = load_sanity_fixture(FIXTURE_PATH)
    case = next(case for case in fixture["cases"] if case["category"] == "abstention")

    result = score_sanity_case(case)

    assert result["abstained"] is True
    assert result["claim_count"] == 0
    assert result["correctness"] is None
    assert result["evidence_support"] is None
    assert result["fully_correct"] is None
    assert result["fully_supported"] is None


def test_tampered_fixture_is_rejected_before_evaluation(tmp_path: Path) -> None:
    tampered_path = tmp_path / "evaluator_sanity_v1.json"
    tampered_path.write_bytes(FIXTURE_PATH.read_bytes().replace(b"Paris", b"Lyon", 1))

    with pytest.raises(FixtureIntegrityError, match="content digest mismatch"):
        validate_evaluator_fixture(tampered_path)


def test_content_seal_is_stable_across_line_endings(tmp_path: Path) -> None:
    alternate_path = tmp_path / "evaluator_sanity_v1.json"
    original_bytes = FIXTURE_PATH.read_bytes()
    lf_bytes = original_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    alternate_bytes = lf_bytes.replace(b"\n", b"\r\n") if original_bytes == lf_bytes else lf_bytes
    assert alternate_bytes != original_bytes
    alternate_path.write_bytes(alternate_bytes)

    report = validate_evaluator_fixture(alternate_path)

    assert report["passed"] is True
    assert report["fixture_sha256"] != sha256_file(FIXTURE_PATH)
    assert report["fixture_sha256"] == sha256_file(alternate_path)
    assert report["fixture_content_sha256"] == EVALUATOR_SANITY_V1_CONTENT_SHA256


def test_fixture_bytes_are_read_once_for_hash_and_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def counting_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    validate_evaluator_fixture(FIXTURE_PATH)

    assert reads == [FIXTURE_PATH]


def test_malformed_fixture_is_rejected_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    malformed = copy.deepcopy(load_sanity_fixture(FIXTURE_PATH))
    del malformed["cases"][0]["claims"][0]["citation_ids"]
    malformed_path = tmp_path / "malformed.json"
    matching_content_digest = _write_fixture(malformed_path, malformed)

    with pytest.raises(FixtureFormatError, match="invalid keys"):
        validate_evaluator_fixture(malformed_path, expected_content_sha256=matching_content_digest)


def test_oracle_mismatch_is_rejected(tmp_path: Path) -> None:
    inconsistent = copy.deepcopy(load_sanity_fixture(FIXTURE_PATH))
    inconsistent["cases"][0]["expected"]["evidence_support"] = 0.25
    inconsistent_path = tmp_path / "inconsistent.json"
    matching_content_digest = _write_fixture(inconsistent_path, inconsistent)

    with pytest.raises(FixtureExpectationError, match="evidence_support"):
        validate_evaluator_fixture(
            inconsistent_path, expected_content_sha256=matching_content_digest
        )


def test_numeric_comparison_uses_absolute_tolerance() -> None:
    def scorer_with_small_drift(case: Mapping[str, Any]) -> Dict[str, Any]:
        result = _score_input_only(case)
        if len(case["claims"]) == 2:
            result["evidence_support"] += 5e-10
        return result

    validate_evaluator_fixture(FIXTURE_PATH, scorer_with_small_drift, atol=1e-9)
    with pytest.raises(FixtureExpectationError, match="evidence_support"):
        validate_evaluator_fixture(FIXTURE_PATH, scorer_with_small_drift, atol=1e-12)


def test_custom_scorer_receives_only_raw_inputs_without_oracle_labels() -> None:
    seen_cases = 0

    def scorer_without_oracle(case: Mapping[str, Any]) -> Dict[str, Any]:
        nonlocal seen_cases
        seen_cases += 1
        assert set(case) == SCORER_INPUT_KEYS
        assert "category" not in case
        assert "oracle" not in case
        assert "expected" not in case
        assert case["id"].startswith("case-")
        for claim in case["claims"]:
            assert set(claim) == {"id", "text", "citation_ids"}
            assert "is_correct" not in claim
            assert "entailing_evidence_ids" not in claim
        for evidence in case["evidence"]:
            assert set(evidence) == {"id", "text"}
        return _score_input_only(case)

    report = validate_evaluator_fixture(
        FIXTURE_PATH,
        scorer_without_oracle,
        scorer_id="fixture/test-scorer",
        scorer_revision=SCORER_REVISION,
    )
    assert report["passed"] is True
    assert report["production_gate_passed"] is True
    assert report["scorer_id"] == "fixture/test-scorer"
    assert seen_cases == report["case_count"]


def test_production_gate_requires_named_revision_pinned_scorer() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        validate_evaluator_fixture(FIXTURE_PATH, scorer_id="missing-revision")
    with pytest.raises(ValueError, match="scorer_id"):
        validate_evaluator_fixture(
            FIXTURE_PATH,
            _score_input_only,
            scorer_id="",
            scorer_revision=SCORER_REVISION,
        )
    with pytest.raises(ValueError, match="scorer_id"):
        validate_evaluator_fixture(
            FIXTURE_PATH,
            _score_input_only,
            scorer_id=" evaluator ",
            scorer_revision=SCORER_REVISION,
        )


@pytest.mark.parametrize(
    "revision",
    [
        "latest",
        "main",
        "head",
        "master",
        "tip",
        "unknown",
        "unpinned",
        "revision-1",
        "A" * 40,
        "0" * 40,
        "0" * 64,
        "0" * 39,
        "0" * 41,
    ],
)
def test_production_gate_rejects_mutable_or_non_digest_revisions(revision: str) -> None:
    with pytest.raises(ValueError, match="scorer_revision"):
        validate_evaluator_fixture(
            FIXTURE_PATH,
            _score_input_only,
            scorer_id="fixture/test-scorer",
            scorer_revision=revision,
        )


def test_builtin_oracle_aggregator_cannot_be_custom_or_production_scorer() -> None:
    with pytest.raises(ValueError, match="built-in oracle aggregator"):
        validate_evaluator_fixture(FIXTURE_PATH, score_sanity_case)
    with pytest.raises(ValueError, match="built-in oracle aggregator"):
        validate_evaluator_fixture(
            FIXTURE_PATH,
            score_sanity_case,
            scorer_id="fixture/test-scorer",
            scorer_revision=SCORER_REVISION,
        )


def test_caller_pinned_fixture_override_is_nonproduction_only(tmp_path: Path) -> None:
    altered = copy.deepcopy(load_sanity_fixture(FIXTURE_PATH))
    altered["cases"][0]["answer"] = "Altered answer text."
    altered_path = tmp_path / "altered.json"
    altered_digest = _write_fixture(altered_path, altered)

    report = validate_evaluator_fixture(
        altered_path,
        _score_input_only,
        expected_content_sha256=altered_digest,
    )
    assert report["passed"] is True
    assert report["production_gate_passed"] is False

    with pytest.raises(FixtureIntegrityError, match="registered canonical fixture digest"):
        validate_evaluator_fixture(
            altered_path,
            _score_input_only,
            scorer_id="fixture/test-scorer",
            scorer_revision=SCORER_REVISION,
            expected_content_sha256=altered_digest,
        )


def test_lax_tolerance_can_never_issue_a_production_pass() -> None:
    def scorer_with_bad_rates(case: Mapping[str, Any]) -> Dict[str, Any]:
        result = _score_input_only(case)
        for key in ("correctness", "evidence_support", "citation_coverage"):
            if result[key] is not None:
                result[key] = 0.123
        return result

    nonproduction = validate_evaluator_fixture(
        FIXTURE_PATH,
        scorer_with_bad_rates,
        atol=1.0,
    )
    assert nonproduction["passed"] is True
    assert nonproduction["production_gate_passed"] is False
    assert nonproduction["atol"] == 1.0

    with pytest.raises(ValueError, match="production atol"):
        validate_evaluator_fixture(
            FIXTURE_PATH,
            scorer_with_bad_rates,
            scorer_id="fixture/test-scorer",
            scorer_revision=SCORER_REVISION,
            atol=1.0,
        )


def test_tolerance_never_relaxes_structural_counts() -> None:
    def scorer_with_wrong_count(case: Mapping[str, Any]) -> Dict[str, Any]:
        result = _score_input_only(case)
        if case["id"] == "case-001":
            result["claim_count"] += 1
        return result

    with pytest.raises(FixtureExpectationError, match="claim_count"):
        validate_evaluator_fixture(FIXTURE_PATH, scorer_with_wrong_count, atol=10.0)


@pytest.mark.parametrize("atol", [-1.0, float("inf"), float("nan"), True, "0.1"])
def test_invalid_tolerance_is_rejected(atol: Any) -> None:
    with pytest.raises(ValueError, match="atol"):
        validate_evaluator_fixture(FIXTURE_PATH, atol=atol)

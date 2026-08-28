"""Deterministic sanity checks for answer and citation evaluators.

The fixture deliberately treats factual correctness and evidential support as
independent claim-level labels.  This catches evaluators that accidentally use
reference correctness as a proxy for citation quality, or vice versa.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

SCHEMA_VERSION = "evaluator-sanity-v1"
EVALUATOR_SANITY_V1_CONTENT_SHA256 = (
    "4d1a496ab46dd4addc9123615ac3b4b56b96a60ae70f0b5fd8c14a01ec900863"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SCORER_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_SCORER_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MUTABLE_SCORER_REVISIONS = {
    "head",
    "latest",
    "main",
    "master",
    "tip",
    "unknown",
    "unpinned",
}
_MAX_PRODUCTION_ATOL = 1e-12
_REQUIRED_CATEGORIES = {
    "abstention",
    "correct_supported",
    "correct_unsupported",
    "partial_support",
    "uncited_nonempty",
    "wrong_supported",
}
_SCORER_CASE_KEY_ORDER = (
    "id",
    "answer",
    "abstained",
    "claims",
    "reference_claims",
    "evidence",
)
_SCORER_CASE_KEYS = set(_SCORER_CASE_KEY_ORDER)
_CASE_KEYS = _SCORER_CASE_KEYS | {"category", "oracle", "expected"}
_CLAIM_KEYS = {"id", "text", "citation_ids"}
_EVIDENCE_KEYS = {"id", "text"}
_ORACLE_KEYS = {"claims"}
_ORACLE_CLAIM_KEYS = {"claim_id", "is_correct", "entailing_evidence_ids"}
_RESULT_KEYS = {
    "abstained",
    "claim_count",
    "correct_claim_count",
    "supported_claim_count",
    "cited_claim_count",
    "correctness",
    "evidence_support",
    "citation_coverage",
    "fully_correct",
    "fully_supported",
}
_COUNT_KEYS = {
    "claim_count",
    "correct_claim_count",
    "supported_claim_count",
    "cited_claim_count",
}
_RATE_KEYS = {"correctness", "evidence_support", "citation_coverage"}
_OPTIONAL_BOOLEAN_KEYS = {"fully_correct", "fully_supported"}

PathLike = Union[str, "Path"]
SanityScorer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class FixtureValidationError(ValueError):
    """Base class for a rejected evaluator fixture."""


class FixtureIntegrityError(FixtureValidationError):
    """Raised when the fixture bytes do not match the trusted digest."""


class FixtureFormatError(FixtureValidationError):
    """Raised when fixture JSON does not conform to the versioned schema."""


class FixtureExpectationError(FixtureValidationError):
    """Raised when an evaluator output differs from the fixture oracle."""


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureFormatError("duplicate JSON object key: {!r}".format(key))
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> None:
    raise FixtureFormatError("non-finite JSON number is not allowed: {}".format(value))


def _require_exact_keys(value: Mapping[str, Any], expected: Set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise FixtureFormatError(
            "{} has invalid keys (missing={}, unexpected={})".format(where, missing, unexpected)
        )


def _validate_identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise FixtureFormatError("{} must be a non-empty portable identifier".format(where))
    return value


def _validate_identifier_list(value: Any, where: str) -> List[str]:
    if not isinstance(value, list):
        raise FixtureFormatError("{} must be a list".format(where))
    result: List[str] = []
    seen: Set[str] = set()
    for index, item in enumerate(value):
        identifier = _validate_identifier(item, "{}[{}]".format(where, index))
        if identifier in seen:
            raise FixtureFormatError(
                "{} contains duplicate identifier {!r}".format(where, identifier)
            )
        seen.add(identifier)
        result.append(identifier)
    return result


def _validate_result_shape(value: Any, where: str) -> None:
    if not isinstance(value, Mapping):
        raise FixtureFormatError("{} must be an object".format(where))
    _require_exact_keys(value, _RESULT_KEYS, where)

    if not isinstance(value["abstained"], bool):
        raise FixtureFormatError("{}.abstained must be boolean".format(where))

    for key in _COUNT_KEYS:
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise FixtureFormatError("{}.{} must be a non-negative integer".format(where, key))

    for key in _RATE_KEYS:
        rate = value[key]
        if rate is not None:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise FixtureFormatError("{}.{} must be a finite rate or null".format(where, key))
            if not math.isfinite(float(rate)) or not 0.0 <= float(rate) <= 1.0:
                raise FixtureFormatError("{}.{} must be in [0, 1]".format(where, key))

    for key in _OPTIONAL_BOOLEAN_KEYS:
        if value[key] is not None and not isinstance(value[key], bool):
            raise FixtureFormatError("{}.{} must be boolean or null".format(where, key))


def _validate_claim(value: Any, where: str) -> str:
    if not isinstance(value, Mapping):
        raise FixtureFormatError("{} must be an object".format(where))
    _require_exact_keys(value, _CLAIM_KEYS, where)
    claim_id = _validate_identifier(value["id"], "{}.id".format(where))
    text = value["text"]
    if not isinstance(text, str) or not text.strip():
        raise FixtureFormatError("{}.text must be a non-empty string".format(where))
    _validate_identifier_list(value["citation_ids"], "{}.citation_ids".format(where))
    return claim_id


def _validate_reference_claims(value: Any, where: str) -> List[str]:
    if not isinstance(value, list):
        raise FixtureFormatError("{} must be a list".format(where))
    result: List[str] = []
    seen: Set[str] = set()
    for index, item in enumerate(value):
        location = "{}[{}]".format(where, index)
        if not isinstance(item, str) or not item.strip():
            raise FixtureFormatError("{} must be a non-empty string".format(location))
        if item != item.strip():
            raise FixtureFormatError("{} must not contain surrounding whitespace".format(location))
        if item in seen:
            raise FixtureFormatError("{} contains duplicate claim text".format(where))
        seen.add(item)
        result.append(item)
    return result


def _validate_evidence(value: Any, where: str) -> Set[str]:
    if not isinstance(value, list):
        raise FixtureFormatError("{} must be a list".format(where))
    evidence_ids: Set[str] = set()
    for index, item in enumerate(value):
        location = "{}[{}]".format(where, index)
        if not isinstance(item, Mapping):
            raise FixtureFormatError("{} must be an object".format(location))
        _require_exact_keys(item, _EVIDENCE_KEYS, location)
        evidence_id = _validate_identifier(item["id"], "{}.id".format(location))
        if evidence_id in evidence_ids:
            raise FixtureFormatError(
                "{} contains duplicate evidence id {!r}".format(where, evidence_id)
            )
        text = item["text"]
        if not isinstance(text, str) or not text.strip():
            raise FixtureFormatError("{}.text must be a non-empty string".format(location))
        if text != text.strip():
            raise FixtureFormatError(
                "{}.text must not contain surrounding whitespace".format(location)
            )
        evidence_ids.add(evidence_id)
    return evidence_ids


def _validate_scorer_case(value: Any, where: str) -> Tuple[List[str], Set[str]]:
    if not isinstance(value, Mapping):
        raise FixtureFormatError("{} must be an object".format(where))
    _require_exact_keys(value, _SCORER_CASE_KEYS, where)
    _validate_identifier(value["id"], "{}.id".format(where))
    if not isinstance(value["answer"], str):
        raise FixtureFormatError("{}.answer must be a string".format(where))
    if not isinstance(value["abstained"], bool):
        raise FixtureFormatError("{}.abstained must be boolean".format(where))

    claims = value["claims"]
    if not isinstance(claims, list):
        raise FixtureFormatError("{}.claims must be a list".format(where))
    seen_claim_ids: Set[str] = set()
    for index, claim in enumerate(claims):
        claim_id = _validate_claim(claim, "{}.claims[{}]".format(where, index))
        if claim_id in seen_claim_ids:
            raise FixtureFormatError("{} contains duplicate claim id {!r}".format(where, claim_id))
        seen_claim_ids.add(claim_id)

    _validate_reference_claims(value["reference_claims"], "{}.reference_claims".format(where))
    evidence_ids = _validate_evidence(value["evidence"], "{}.evidence".format(where))
    for index, claim in enumerate(claims):
        unknown_citations = sorted(set(claim["citation_ids"]) - evidence_ids)
        if unknown_citations:
            raise FixtureFormatError(
                "{}.claims[{}].citation_ids contains unknown evidence IDs {}".format(
                    where, index, unknown_citations
                )
            )

    if value["abstained"]:
        if claims:
            raise FixtureFormatError(
                "{}: an abstention cannot contain factual claims".format(where)
            )
    else:
        if not value["answer"].strip():
            raise FixtureFormatError(
                "{}: a non-abstention must have a non-empty answer".format(where)
            )
        if not claims:
            raise FixtureFormatError(
                "{}: a non-abstention must contain at least one claim".format(where)
            )

    return list(seen_claim_ids), evidence_ids


def _validate_oracle(
    value: Any,
    where: str,
    *,
    claim_ids: Sequence[str],
    evidence_ids: Set[str],
) -> None:
    if not isinstance(value, Mapping):
        raise FixtureFormatError("{} must be an object".format(where))
    _require_exact_keys(value, _ORACLE_KEYS, where)
    annotations = value["claims"]
    if not isinstance(annotations, list):
        raise FixtureFormatError("{}.claims must be a list".format(where))

    seen_claim_ids: Set[str] = set()
    for index, annotation in enumerate(annotations):
        location = "{}.claims[{}]".format(where, index)
        if not isinstance(annotation, Mapping):
            raise FixtureFormatError("{} must be an object".format(location))
        _require_exact_keys(annotation, _ORACLE_CLAIM_KEYS, location)
        claim_id = _validate_identifier(annotation["claim_id"], "{}.claim_id".format(location))
        if claim_id in seen_claim_ids:
            raise FixtureFormatError(
                "{} contains duplicate claim annotation {!r}".format(where, claim_id)
            )
        seen_claim_ids.add(claim_id)
        if not isinstance(annotation["is_correct"], bool):
            raise FixtureFormatError("{}.is_correct must be boolean".format(location))
        entailing_ids = _validate_identifier_list(
            annotation["entailing_evidence_ids"],
            "{}.entailing_evidence_ids".format(location),
        )
        unknown_entailing = sorted(set(entailing_ids) - evidence_ids)
        if unknown_entailing:
            raise FixtureFormatError(
                "{}.entailing_evidence_ids contains unknown evidence IDs {}".format(
                    location, unknown_entailing
                )
            )

    if seen_claim_ids != set(claim_ids):
        raise FixtureFormatError(
            "{}.claims must annotate exactly the scorer-visible claim IDs".format(where)
        )


def _validate_case(value: Any, where: str) -> str:
    if not isinstance(value, Mapping):
        raise FixtureFormatError("{} must be an object".format(where))
    _require_exact_keys(value, _CASE_KEYS, where)
    case_id = _validate_identifier(value["id"], "{}.id".format(where))
    scorer_case = {key: value[key] for key in _SCORER_CASE_KEY_ORDER}
    claim_ids, evidence_ids = _validate_scorer_case(scorer_case, "{}.scorer_input".format(where))

    category = value["category"]
    if not isinstance(category, str) or category not in _REQUIRED_CATEGORIES:
        raise FixtureFormatError("{}.category is not recognized".format(where))
    _validate_oracle(
        value["oracle"],
        "{}.oracle".format(where),
        claim_ids=claim_ids,
        evidence_ids=evidence_ids,
    )
    _validate_result_shape(value["expected"], "{}.expected".format(where))
    return case_id


def _score_validated_case(case: Mapping[str, Any]) -> Dict[str, Any]:
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

    annotations = {annotation["claim_id"]: annotation for annotation in case["oracle"]["claims"]}
    claim_count = len(claims)
    correct_claim_count = sum(1 for claim in claims if annotations[claim["id"]]["is_correct"])
    cited_claim_count = sum(1 for claim in claims if claim["citation_ids"])
    supported_claim_count = sum(
        1
        for claim in claims
        if set(claim["citation_ids"]) & set(annotations[claim["id"]]["entailing_evidence_ids"])
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


def score_sanity_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    """Aggregate one trusted, fully annotated harness case.

    This built-in helper deliberately consumes harness-only oracle annotations.
    It is used to validate fixture consistency and can never qualify as a
    production evaluator.  Custom evaluators receive only raw scorer inputs.
    """

    _validate_case(case, "case")
    return _score_validated_case(case)


def _validate_category_semantics(case: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    category = case["category"]
    case_id = case["id"]
    valid = False
    if category == "correct_supported":
        valid = result["correctness"] == 1.0 and result["evidence_support"] == 1.0
    elif category == "correct_unsupported":
        valid = (
            result["correctness"] == 1.0
            and result["evidence_support"] == 0.0
            and result["citation_coverage"] > 0.0
        )
    elif category == "wrong_supported":
        valid = result["correctness"] == 0.0 and result["evidence_support"] == 1.0
    elif category == "partial_support":
        valid = result["claim_count"] >= 2 and 0.0 < result["evidence_support"] < 1.0
    elif category == "abstention":
        valid = result["abstained"] is True
    elif category == "uncited_nonempty":
        valid = (
            result["abstained"] is False
            and bool(case["answer"].strip())
            and result["citation_coverage"] == 0.0
            and result["evidence_support"] == 0.0
        )
    if not valid:
        raise FixtureFormatError(
            "case {!r} does not satisfy category {!r}".format(case_id, category)
        )


def _validate_document(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise FixtureFormatError("fixture root must be an object")
    _require_exact_keys(value, {"schema_version", "cases"}, "fixture")
    if value["schema_version"] != SCHEMA_VERSION:
        raise FixtureFormatError("unsupported schema_version {!r}".format(value["schema_version"]))
    cases = value["cases"]
    if not isinstance(cases, list):
        raise FixtureFormatError("fixture.cases must be a list")

    seen_case_ids: Set[str] = set()
    seen_categories: Set[str] = set()
    for index, case in enumerate(cases):
        where = "fixture.cases[{}]".format(index)
        case_id = _validate_case(case, where)
        if case_id in seen_case_ids:
            raise FixtureFormatError("fixture contains duplicate case id {!r}".format(case_id))
        seen_case_ids.add(case_id)
        category = case["category"]
        if category in seen_categories:
            raise FixtureFormatError("fixture contains duplicate category {!r}".format(category))
        seen_categories.add(category)
        _validate_category_semantics(case, _score_validated_case(case))

    if seen_categories != _REQUIRED_CATEGORIES:
        raise FixtureFormatError(
            "fixture categories must be exactly {}".format(sorted(_REQUIRED_CATEGORIES))
        )
    return value


def _validate_expected_digest(expected_sha256: str) -> str:
    if not isinstance(expected_sha256, str):
        raise FixtureIntegrityError("expected content digest must be a SHA-256 string")
    normalized_expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(normalized_expected):
        raise FixtureIntegrityError("expected content digest must be 64 hexadecimal digits")
    return normalized_expected


def _content_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FixtureFormatError("fixture cannot be canonicalized: {}".format(exc)) from exc
    return hashlib.sha256(encoded).hexdigest()


def _verify_content_digest(value: Any, expected_sha256: str) -> str:
    normalized_expected = _validate_expected_digest(expected_sha256)
    actual = _content_digest(value)
    if not secrets.compare_digest(actual, normalized_expected):
        raise FixtureIntegrityError(
            "fixture content digest mismatch: expected {}, got {}".format(
                normalized_expected, actual
            )
        )
    return actual


def _validate_fixture_bytes(
    raw_bytes: bytes,
    expected_content_sha256: str,
) -> Tuple[Dict[str, Any], str, str]:
    artifact_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        text = raw_bytes.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_json,
        )
    except FixtureFormatError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureFormatError("could not parse fixture JSON: {}".format(exc)) from exc
    content_sha256 = _verify_content_digest(value, expected_content_sha256)
    return _validate_document(value), artifact_sha256, content_sha256


def _load_and_validate(
    path: PathLike,
    expected_content_sha256: str,
) -> Tuple[Dict[str, Any], str, str]:
    fixture_path = Path(path)
    try:
        raw_bytes = fixture_path.read_bytes()
    except OSError as exc:
        raise FixtureIntegrityError("could not read fixture bytes: {}".format(exc)) from exc
    return _validate_fixture_bytes(raw_bytes, expected_content_sha256)


def load_sanity_fixture_bytes(
    raw_bytes: bytes,
    *,
    expected_content_sha256: str = EVALUATOR_SANITY_V1_CONTENT_SHA256,
) -> Dict[str, Any]:
    """Validate an exact already-read fixture snapshot without reopening a path."""

    if not isinstance(raw_bytes, bytes):
        raise TypeError("raw_bytes must be bytes")
    fixture, _, _ = _validate_fixture_bytes(raw_bytes, expected_content_sha256)
    return fixture


def load_sanity_fixture(
    path: PathLike,
    *,
    expected_content_sha256: str = EVALUATOR_SANITY_V1_CONTENT_SHA256,
) -> Dict[str, Any]:
    """Load a schema-checked fixture after verifying canonical JSON content."""

    fixture, _, _ = _load_and_validate(path, expected_content_sha256)
    return fixture


def _compare_result(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    atol: float,
    case_id: str,
) -> None:
    for key in sorted(_RESULT_KEYS):
        expected_value = expected[key]
        actual_value = actual[key]
        if (
            key in _RATE_KEYS
            and not isinstance(expected_value, bool)
            and not isinstance(actual_value, bool)
            and isinstance(expected_value, (int, float))
            and isinstance(actual_value, (int, float))
        ):
            matches = math.isclose(
                float(expected_value),
                float(actual_value),
                rel_tol=0.0,
                abs_tol=atol,
            )
        else:
            matches = type(expected_value) is type(actual_value) and expected_value == actual_value
        if not matches:
            raise FixtureExpectationError(
                "case {!r} metric {!r}: expected {!r}, got {!r}".format(
                    case_id, key, expected_value, actual_value
                )
            )


def validate_evaluator_fixture(
    path: PathLike,
    scorer: Optional[SanityScorer] = None,
    *,
    scorer_id: Optional[str] = None,
    scorer_revision: Optional[str] = None,
    expected_content_sha256: str = EVALUATOR_SANITY_V1_CONTENT_SHA256,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """Verify fixture integrity and compare a scorer with every oracle result.

    Args:
        path: Location of the versioned JSON artifact.
        scorer: Optional evaluator under test. It receives only an opaque case
            ID and raw answer, claim, reference-claim, citation, and evidence
            inputs. Category and oracle annotations never cross this boundary.
        scorer_id: Stable identifier for a production evaluator under test.
        scorer_revision: Immutable revision for that evaluator.
        expected_content_sha256: Trusted digest of canonical fixture JSON.
            Canonicalization makes the seal stable across LF and CRLF checkouts;
            ``fixture_sha256`` in the report still fingerprints the exact bytes.
            An override is permitted only for a non-production self-check.
        atol: Absolute tolerance used only for numeric comparisons.

    Returns:
        A compact report containing the verified digest and per-case results.
    """

    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        raise ValueError("atol must be a finite non-negative number")
    if not math.isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")

    if scorer is not None and not callable(scorer):
        raise TypeError("scorer must be callable")
    if scorer is score_sanity_case:
        raise ValueError("the built-in oracle aggregator cannot be used as a custom scorer")
    if (scorer_id is None) != (scorer_revision is None):
        raise ValueError("scorer_id and scorer_revision must be supplied together")
    production_requested = scorer_id is not None
    if production_requested:
        if float(atol) > _MAX_PRODUCTION_ATOL:
            raise ValueError(
                "production atol must be no greater than {:.0e}".format(_MAX_PRODUCTION_ATOL)
            )
        if scorer is None:
            raise ValueError("a custom scorer is required for a production gate")
        if not isinstance(scorer_id, str) or not _SCORER_IDENTITY_RE.fullmatch(scorer_id):
            raise ValueError("scorer_id must be a non-empty, unpadded stable identifier")
        if not isinstance(scorer_revision, str):
            raise ValueError("scorer_revision must be an immutable lowercase digest")
        if scorer_revision.lower() in _MUTABLE_SCORER_REVISIONS:
            raise ValueError("scorer_revision must be immutable, not a moving revision name")
        if not _SCORER_REVISION_RE.fullmatch(scorer_revision):
            raise ValueError("scorer_revision must be an immutable lowercase 40- or 64-hex digest")
        if not scorer_revision.strip("0"):
            raise ValueError("scorer_revision must not be an all-zero sentinel digest")
        normalized_expected = _validate_expected_digest(expected_content_sha256)
        if not secrets.compare_digest(normalized_expected, EVALUATOR_SANITY_V1_CONTENT_SHA256):
            raise FixtureIntegrityError(
                "a production gate requires the registered canonical fixture digest"
            )

    fixture, fixture_sha256, content_sha256 = _load_and_validate(path, expected_content_sha256)

    results: List[Dict[str, Any]] = []
    for case in fixture["cases"]:
        actual: Mapping[str, Any]
        if scorer is None:
            actual = _score_validated_case(case)
        else:
            scorer_case = {key: copy.deepcopy(case[key]) for key in _SCORER_CASE_KEY_ORDER}
            _validate_scorer_case(scorer_case, "scorer input for {!r}".format(case["id"]))
            actual = scorer(scorer_case)
        _validate_result_shape(actual, "scorer result for {!r}".format(case["id"]))
        _compare_result(case["expected"], actual, atol=float(atol), case_id=case["id"])
        results.append({"id": case["id"], **dict(actual)})

    return {
        "passed": True,
        "production_gate_passed": production_requested,
        "scorer_id": scorer_id,
        "scorer_revision": scorer_revision,
        "schema_version": fixture["schema_version"],
        "fixture_sha256": fixture_sha256,
        "fixture_content_sha256": content_sha256,
        "atol": float(atol),
        "case_count": len(results),
        "results": results,
    }


__all__ = [
    "EVALUATOR_SANITY_V1_CONTENT_SHA256",
    "FixtureExpectationError",
    "FixtureFormatError",
    "FixtureIntegrityError",
    "FixtureValidationError",
    "SCHEMA_VERSION",
    "load_sanity_fixture",
    "load_sanity_fixture_bytes",
    "score_sanity_case",
    "validate_evaluator_fixture",
]

#!/usr/bin/env python
"""
scripts/analyze_scorer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 4B: independently judged scorer discrimination analysis.

Evaluates persisted passage scores against independently supplied relevance
judgments. Reports ROC-AUC, average precision, and an exploratory in-sample threshold.

Non-mock analysis deliberately fails closed unless the predictions contain all
scored candidates and a separate, revision-bound judgment artifact is supplied.
It never manufactures labels from the scorer's own ranking.

Usage::

    python scripts/analyze_scorer.py \\
        --predictions runs/<run-id>/predictions.jsonl \\
        --judgments data/scorer_judgments.json \\
        --sample 500 \\
        --seed 42 \\
        --output analysis/scorer_auc.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independently judged scorer analysis.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Scorer config for synthetic --mock smoke tests only.",
    )
    p.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions.jsonl from a completed run.",
    )
    p.add_argument(
        "--judgments",
        type=str,
        default=None,
        help=(
            "Path to factuality-rag.scorer-judgments.v1 JSON produced "
            "independently of the scorer. Required outside mock mode."
        ),
    )
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="analysis/scorer_auc.json")
    p.add_argument("--mock", action="store_true", help="Run in mock mode for testing.")
    return p.parse_args()


JUDGMENTS_SCHEMA = "factuality-rag.scorer-judgments.v1"
ANALYSIS_SCHEMA = "factuality-rag.scorer-analysis.v1"


class IndependentJudgment(NamedTuple):
    """Explicit labeled universe and its relevant subset for one example."""

    judged_passage_ids: Optional[Set[str]]
    relevant_passage_ids: Set[str]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token!r}")
    return value


def _reject_non_finite_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number: {token!r}")


def _strict_json_loads(text: str, artifact_name: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_finite_float,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{artifact_name} is not valid JSON") from exc
    except ValueError as exc:
        raise ValueError(f"{artifact_name} is not strict JSON: {exc}") from exc


def _read_json(path: Path, artifact_name: str) -> Tuple[Mapping[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"{artifact_name} file not found: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact_name} must be valid UTF-8 JSON") from exc
    parsed = _strict_json_loads(text, artifact_name)
    if not isinstance(parsed, dict):
        raise ValueError(f"{artifact_name} must be a JSON object")
    return parsed, _sha256_bytes(raw)


def _read_predictions(path: Path) -> Tuple[List[Mapping[str, Any]], str]:
    if not path.is_file():
        raise ValueError(f"predictions file not found: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("predictions must be valid UTF-8 JSONL") from exc

    records: List[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        record = _strict_json_loads(line, f"predictions line {line_number}")
        if not isinstance(record, dict):
            raise ValueError(f"predictions line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValueError("predictions must contain at least one record")
    return records, _sha256_bytes(raw)


def _require_trimmed_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_trimmed_id(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256 digest")
    if set(digest) == {"0"}:
        raise ValueError(f"{field} cannot be an all-zero placeholder")
    return digest


def _load_independent_judgments(
    path: Path,
    expected_predictions_sha256: Optional[str] = None,
) -> Tuple[Dict[str, IndependentJudgment], Dict[str, Any]]:
    payload, artifact_sha256 = _read_json(path, "judgments")
    bound_keys = {
        "schema",
        "independent_of_scorer",
        "predictions_sha256",
        "label_source",
        "label_source_revision",
        "judgments",
    }
    legacy_keys = bound_keys - {"predictions_sha256"}
    is_bound = "predictions_sha256" in payload
    if set(payload) != (bound_keys if is_bound else legacy_keys):
        raise ValueError("judgments has missing or unknown top-level fields")
    if expected_predictions_sha256 is not None and not is_bound:
        raise ValueError(
            "judgments must bind the exact predictions artifact with predictions_sha256"
        )
    if payload.get("schema") != JUDGMENTS_SCHEMA:
        raise ValueError(f"judgments schema must be {JUDGMENTS_SCHEMA!r}")
    if payload.get("independent_of_scorer") is not True:
        raise ValueError("judgments must be produced independently of the scorer")

    bound_predictions_sha256: Optional[str] = None
    if is_bound:
        bound_predictions_sha256 = _require_sha256(
            payload.get("predictions_sha256"), "predictions_sha256"
        )
    if expected_predictions_sha256 is not None:
        expected_digest = _require_sha256(
            expected_predictions_sha256, "expected_predictions_sha256"
        )
        if bound_predictions_sha256 != expected_digest:
            raise ValueError(
                "judgments predictions_sha256 does not match the supplied predictions artifact"
            )

    label_source = _require_trimmed_id(payload.get("label_source"), "label_source")
    revision = _require_trimmed_id(payload.get("label_source_revision"), "label_source_revision")
    if len(revision) not in (40, 64) or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("label_source_revision must be a lowercase 40- or 64-hex revision")
    if set(revision) == {"0"}:
        raise ValueError("label_source_revision cannot be an all-zero placeholder")
    if revision == bound_predictions_sha256:
        raise ValueError(
            "label_source_revision must identify an independent source, not the predictions artifact"
        )

    raw_judgments = payload.get("judgments")
    if not isinstance(raw_judgments, list) or not raw_judgments:
        raise ValueError("judgments must be a non-empty list")

    by_example: Dict[str, IndependentJudgment] = {}
    for index, item in enumerate(raw_judgments):
        expected_item_keys = (
            {"example_id", "judged_passage_ids", "relevant_passage_ids"}
            if is_bound
            else {"example_id", "relevant_passage_ids"}
        )
        if not isinstance(item, dict) or set(item) != expected_item_keys:
            raise ValueError(f"judgments[{index}] has an invalid schema")
        example_id = _require_trimmed_id(item.get("example_id"), f"judgments[{index}].example_id")
        if example_id in by_example:
            raise ValueError(f"duplicate judgment example_id: {example_id}")
        raw_ids = item.get("relevant_passage_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"judgments[{index}].relevant_passage_ids must be non-empty")
        relevant_ids = {
            _require_trimmed_id(value, f"judgments[{index}].relevant_passage_ids")
            for value in raw_ids
        }
        if len(relevant_ids) != len(raw_ids):
            raise ValueError(f"judgments[{index}] contains duplicate relevant passage IDs")

        judged_ids: Optional[Set[str]] = None
        if is_bound:
            raw_judged_ids = item.get("judged_passage_ids")
            if not isinstance(raw_judged_ids, list) or not raw_judged_ids:
                raise ValueError(f"judgments[{index}].judged_passage_ids must be non-empty")
            judged_ids = {
                _require_trimmed_id(value, f"judgments[{index}].judged_passage_ids")
                for value in raw_judged_ids
            }
            if len(judged_ids) != len(raw_judged_ids):
                raise ValueError(f"judgments[{index}] contains duplicate judged passage IDs")
            if not relevant_ids <= judged_ids:
                raise ValueError(
                    f"judgments[{index}].relevant_passage_ids must be a subset of "
                    "judged_passage_ids"
                )
        by_example[example_id] = IndependentJudgment(judged_ids, relevant_ids)

    provenance = {
        "judgments_sha256": artifact_sha256,
        "label_source": label_source,
        "label_source_revision": revision,
        "predictions_sha256": bound_predictions_sha256,
        "predictions_binding_verified": expected_predictions_sha256 is not None,
    }
    return by_example, provenance


def _selection_rank(seed: int, example_id: str) -> bytes:
    material = f"scorer-analysis.sample.v1\0{seed}\0{example_id}".encode("utf-8")
    return hashlib.sha256(material).digest()


def _collect_independently_labeled_scores(
    predictions: Sequence[Mapping[str, Any]],
    judgments: Mapping[str, Union[IndependentJudgment, Set[str]]],
    sample_limit: int,
    seed: int = 42,
    *,
    accounting: Optional[Dict[str, Any]] = None,
    selection: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[int], int]:
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit <= 0:
        raise ValueError("sample must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    scores: List[float] = []
    labels: List[int] = []
    seen_examples: Set[str] = set()
    validated: List[Tuple[int, str, List[Tuple[str, float]], Set[str], Set[str]]] = []
    total_candidates = 0
    total_judged_candidates = 0

    for record_index, prediction in enumerate(predictions):
        example_id = _require_trimmed_id(
            prediction.get("example_id"), f"predictions[{record_index}].example_id"
        )
        if example_id in seen_examples:
            raise ValueError(f"duplicate prediction example_id: {example_id}")
        seen_examples.add(example_id)
        if example_id not in judgments:
            raise ValueError(f"missing independent judgment for example_id: {example_id}")

        candidates = prediction.get("scored_passages")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(
                f"prediction {example_id!r} must contain at least two scored_passages; "
                "thresholded trusted_passages are not sufficient for AUC"
            )
        raw_judgment = judgments[example_id]
        if isinstance(raw_judgment, IndependentJudgment):
            judged_ids = raw_judgment.judged_passage_ids
            relevant_ids = raw_judgment.relevant_passage_ids
        elif isinstance(raw_judgment, set):
            # Compatibility for direct callers of the pre-binding helper. The
            # production CLI never reaches this path because it requires a
            # predictions-bound artifact with judged_passage_ids.
            judged_ids = None
            relevant_ids = raw_judgment
        else:
            raise ValueError(f"invalid independent judgment for example_id: {example_id}")

        candidate_ids: Set[str] = set()
        validated_candidates: List[Tuple[str, float]] = []
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or set(candidate) != {"id", "final_score"}:
                raise ValueError(
                    f"prediction {example_id!r} scored_passages[{candidate_index}] must contain "
                    "exactly id and final_score"
                )
            passage_id = _require_trimmed_id(
                candidate.get("id"),
                f"prediction {example_id!r} scored_passages[{candidate_index}].id",
            )
            if passage_id in candidate_ids:
                raise ValueError(
                    f"prediction {example_id!r} has duplicate passage ID {passage_id!r}"
                )
            candidate_ids.add(passage_id)
            score = candidate.get("final_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"prediction {example_id!r} passage {passage_id!r} score is invalid"
                )
            numeric_score = float(score)
            if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
                raise ValueError(
                    f"prediction {example_id!r} passage {passage_id!r} score must be finite in [0, 1]"
                )
            validated_candidates.append((passage_id, numeric_score))

        if judged_ids is None:
            judged_ids = set(candidate_ids)
        unknown_judged = judged_ids - candidate_ids
        if unknown_judged:
            raise ValueError(
                f"judgment {example_id!r} names passage IDs absent from scored_passages"
            )
        if not relevant_ids <= judged_ids:
            raise ValueError(
                f"judgment {example_id!r} relevant passage IDs must be a subset of judged IDs"
            )
        missing_relevant = relevant_ids - candidate_ids
        if missing_relevant:
            raise ValueError(
                f"prediction {example_id!r} omits independently relevant candidate IDs"
            )
        total_candidates += len(candidate_ids)
        total_judged_candidates += len(judged_ids)
        validated.append((record_index, example_id, validated_candidates, judged_ids, relevant_ids))

    extra_judgments = set(judgments) - seen_examples
    if extra_judgments:
        raise ValueError("judgments contain example IDs absent from predictions")

    selected = sorted(
        validated,
        key=lambda item: (_selection_rank(seed, item[1]), item[1]),
    )[:sample_limit]
    selected_candidates = 0
    selected_judged_candidates = 0
    for _, example_id, candidates, judged_ids, relevant_ids in selected:
        example_labels: List[int] = []
        selected_candidates += len(candidates)
        selected_judged_candidates += len(judged_ids)
        for passage_id, numeric_score in candidates:
            if passage_id not in judged_ids:
                continue
            label = int(passage_id in relevant_ids)
            scores.append(numeric_score)
            labels.append(label)
            example_labels.append(label)

        if set(example_labels) != {0, 1}:
            raise ValueError(
                f"prediction {example_id!r} must include at least one independently judged "
                "relevant and one independently judged non-relevant candidate"
            )

    if accounting is not None:
        accounting.update(
            {
                "n_candidates_all_examples": total_candidates,
                "n_judged_candidates_all_examples": total_judged_candidates,
                "n_excluded_unjudged_candidates_all_examples": (
                    total_candidates - total_judged_candidates
                ),
                "n_candidates_selected_examples": selected_candidates,
                "n_judged_candidates_selected_examples": selected_judged_candidates,
                "n_excluded_unjudged_candidates_selected_examples": (
                    selected_candidates - selected_judged_candidates
                ),
            }
        )
    if selection is not None:
        selection.update(
            {
                "method": "sha256_rank_v1",
                "seed": seed,
                "requested_sample_limit": sample_limit,
                "available_examples": len(validated),
                "selected_example_ids": [item[1] for item in selected],
            }
        )
    return scores, labels, len(selected)


def _compute_metrics(all_scores: Sequence[float], all_labels: Sequence[int]) -> Dict[str, Any]:
    if not all_scores or len(all_scores) != len(all_labels):
        raise ValueError("scores and labels must be non-empty and have equal lengths")
    scores_arr = np.asarray(all_scores, dtype=np.float64)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    if not np.isfinite(scores_arr).all() or ((scores_arr < 0) | (scores_arr > 1)).any():
        raise ValueError("all scores must be finite values in [0, 1]")
    if set(labels_arr.tolist()) != {0, 1}:
        raise ValueError("AUC requires both independently labeled classes")

    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

    roc_auc = float(roc_auc_score(labels_arr, scores_arr))
    average_precision = float(average_precision_score(labels_arr, scores_arr))
    if not math.isfinite(roc_auc) or not math.isfinite(average_precision):
        raise ValueError("AUC computation returned a non-finite result")

    best_thresh = 0.5
    best_j = -1.0
    for thresh in np.arange(0.0, 1.0001, 0.01):
        predicted = (scores_arr >= thresh).astype(int)
        tp = ((predicted == 1) & (labels_arr == 1)).sum()
        fp = ((predicted == 1) & (labels_arr == 0)).sum()
        fn = ((predicted == 0) & (labels_arr == 1)).sum()
        tn = ((predicted == 0) & (labels_arr == 0)).sum()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        youden_j = float(tpr - fpr)
        if youden_j > best_j:
            best_j = youden_j
            best_thresh = float(thresh)

    return {
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(average_precision, 4),
        "exploratory_in_sample_threshold": round(best_thresh, 3),
        "exploratory_youden_j": round(best_j, 4),
        "n_passages": len(all_scores),
        "n_positive": int(labels_arr.sum()),
        "n_negative": int((1 - labels_arr).sum()),
        "mean_score_positive": round(float(scores_arr[labels_arr == 1].mean()), 4),
        "mean_score_negative": round(float(scores_arr[labels_arr == 0].mean()), 4),
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete JSON artifact atomically without replacing a prior file."""
    try:
        raw = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis output is not strict JSON serializable") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(raw)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace existing analysis output: {path}") from exc
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    if isinstance(args.sample, bool) or args.sample <= 0:
        raise ValueError("sample must be a positive integer")
    out_path = Path(args.output)
    if out_path.exists():
        raise FileExistsError(f"refusing to replace existing analysis output: {out_path}")

    provenance: Dict[str, Any]
    candidate_accounting: Dict[str, Any]
    if args.mock:
        if args.predictions or args.judgments:
            raise ValueError("mock mode cannot consume predictions or judgment artifacts")
        from factuality_rag.pipeline.orchestrator import _load_config
        from factuality_rag.scorer.passage import PassageScorer

        cfg = _load_config(args.config)
        scorer_cfg = cfg.get("scorer", {})
        weights = scorer_cfg.get("weights", {})
        scorer = PassageScorer(
            nli_model_hf=cfg.get("models", {}).get(
                "nli_verifier",
                "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            ),
            overlap_metric=scorer_cfg.get("overlap_metric", "token"),
            mock_mode=True,
            w_nli=weights.get("w_nli", 0.5),
            w_overlap=weights.get("w_overlap", 0.2),
            w_ret=weights.get("w_ret", 0.3),
            nli_mode=scorer_cfg.get("nli_mode", "passage"),
        )
        queries = ["What is the capital of France?"] * args.sample
        gold_passages = [
            {
                "id": f"gold_{i}",
                "text": f"Paris is the capital of France {i}",
                "combined_score": 0.8,
                "label": 1,
            }
            for i in range(args.sample)
        ]
        distractor_passages = [
            {
                "id": f"dist_{i}",
                "text": f"Berlin is a large city in Europe {i}",
                "combined_score": 0.3,
                "label": 0,
            }
            for i in range(args.sample)
        ]
        all_scores: List[float] = []
        all_labels: List[int] = []
        for index in range(args.sample):
            scored = scorer.score_passages(
                queries[index], [gold_passages[index], distractor_passages[index]]
            )
            for passage in scored:
                all_scores.append(float(passage["final_score"]))
                all_labels.append(int(passage["label"]))
        metrics = _compute_metrics(all_scores, all_labels)
        n_examples = args.sample
        status = "synthetic_mock_smoke_test_only_non_claim"
        claim_status = "NOT_ELIGIBLE_FOR_EMPIRICAL_OR_PERFORMANCE_CLAIMS"
        threshold_use = "synthetic_smoke_test_only_not_a_threshold_estimate"
        candidate_accounting = {
            "n_candidates_all_examples": args.sample * 2,
            "n_judged_candidates_all_examples": args.sample * 2,
            "n_excluded_unjudged_candidates_all_examples": 0,
            "n_candidates_selected_examples": args.sample * 2,
            "n_judged_candidates_selected_examples": args.sample * 2,
            "n_excluded_unjudged_candidates_selected_examples": 0,
        }
        provenance = {
            "seed": args.seed,
            "data_origin": "synthetic_mock_fixture",
            "contains_empirical_evidence": False,
            "selection": {
                "method": "synthetic_fixture_enumeration",
                "seed": args.seed,
                "requested_sample_limit": args.sample,
                "available_examples": args.sample,
                "selected_example_ids": [f"synthetic-{index}" for index in range(args.sample)],
            },
        }
    else:
        if args.config is not None:
            raise ValueError(
                "--config is only valid with --mock; persisted scores are analyzed as-is"
            )
        if not args.predictions or not args.judgments:
            raise ValueError(
                "non-mock analysis requires both --predictions and --judgments; "
                "synthetic real-data labels are forbidden"
            )
        predictions, predictions_sha256 = _read_predictions(Path(args.predictions))
        judgments, provenance = _load_independent_judgments(
            Path(args.judgments), expected_predictions_sha256=predictions_sha256
        )
        candidate_accounting = {}
        selection: Dict[str, Any] = {}
        all_scores, all_labels, n_examples = _collect_independently_labeled_scores(
            predictions,
            judgments,
            args.sample,
            seed=args.seed,
            accounting=candidate_accounting,
            selection=selection,
        )
        metrics = _compute_metrics(all_scores, all_labels)
        status = "valid_independent_judgments"
        claim_status = "EXPLORATORY_ONLY_REQUIRES_FROZEN_PROTOCOL_FOR_CONFIRMATORY_CLAIMS"
        threshold_use = "exploratory_in_sample_not_for_confirmatory_claims"
        provenance["selection"] = selection

    output = {
        "schema": ANALYSIS_SCHEMA,
        "status": status,
        "claim_status": claim_status,
        "n_examples": n_examples,
        "threshold_use": threshold_use,
        "provenance": provenance,
        "candidate_accounting": candidate_accounting,
        "metrics": metrics,
    }
    _write_new_json(out_path, output)

    logger.info("Scorer AUC analysis saved → %s", out_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

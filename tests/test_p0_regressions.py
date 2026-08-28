"""Root-cause regressions for the P0 correctness and reproducibility blockers."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from factuality_rag.determinism import stable_seed
from factuality_rag.gating.probe import GatingProbe
from factuality_rag.retriever.hybrid import BM25BackendError, HybridRetriever
from factuality_rag.scorer.passage import PassageScorer
from scripts import analyze_gating, analyze_scorer


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stable_seed_is_namespaced_and_type_strict() -> None:
    assert stable_seed("component", "query", 0) == stable_seed("component", "query", 0)
    assert stable_seed("component-a", "query", 0) != stable_seed("component-b", "query", 0)
    assert stable_seed("component", "query", 0) != stable_seed("component", "query", 1)
    with pytest.raises(ValueError, match="namespace"):
        stable_seed(" component", "query")
    with pytest.raises(TypeError, match="seed parts"):
        stable_seed("component", True)


def test_mock_logits_single_and_multistep_share_step_zero() -> None:
    probe = GatingProbe("mock", mock_mode=True)
    single = probe._get_next_token_logits("deterministic prompt")
    multi = probe._get_multi_token_logits("deterministic prompt", k=2)
    np.testing.assert_array_equal(single, multi[0])
    assert not np.array_equal(multi[0], multi[1])


def _mock_digest_for_hash_seed(hash_seed: str) -> str:
    code = """
import hashlib
import json
from factuality_rag.gating.probe import GatingProbe
from factuality_rag.retriever.hybrid import HybridRetriever
from factuality_rag.scorer.passage import PassageScorer

probe = GatingProbe('mock', mock_mode=True)
single = probe._get_next_token_logits('same prompt')
multi = probe._get_multi_token_logits('same prompt', k=2)
scorer = PassageScorer('mock', mock_mode=True, cross_encoder_model='mock')
nli = scorer._nli_entailment('premise', 'hypothesis')
ranked = scorer._cross_encoder_rerank(
    'same query',
    [
        {'id': 'a', 'text': 'a', 'combined_score': 0.1},
        {'id': 'b', 'text': 'b', 'combined_score': 0.2},
    ],
)
retriever = HybridRetriever('unused', 'unused')
retriever._mock_mode = True
retriever._id_map = ['a', 'b']
bm25 = retriever._bm25_search('same query', 2)
payload = b''.join([single.tobytes(), multi[0].tobytes(), multi[1].tobytes()])
payload += json.dumps(
    {'nli': nli, 'ranked': ranked, 'bm25': bm25},
    sort_keys=True,
    separators=(',', ':'),
).encode('utf-8')
print(hashlib.sha256(payload).hexdigest())
"""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    digest = completed.stdout.strip()
    assert len(digest) == 64
    int(digest, 16)
    return digest


def test_mock_pipeline_is_stable_across_python_hash_seeds() -> None:
    assert _mock_digest_for_hash_seed("1") == _mock_digest_for_hash_seed("8675309")


def test_no_mock_path_uses_process_salted_builtin_hash() -> None:
    paths = [
        REPO_ROOT / "factuality_rag" / "gating" / "probe.py",
        REPO_ROOT / "factuality_rag" / "scorer" / "passage.py",
        REPO_ROOT / "factuality_rag" / "retriever" / "hybrid.py",
    ]
    for path in paths:
        assert "hash(" not in path.read_text(encoding="utf-8"), path


def test_missing_pyserini_index_does_not_import_or_mutate_java_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_index = tmp_path / "missing-lucene-index"
    retriever = HybridRetriever("unused", str(missing_index))
    requested_pyserini_imports: List[str] = []
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyserini" or name.startswith("pyserini."):
            requested_pyserini_imports.append(name)
            raise AssertionError("missing indexes must not initialize Pyserini/JNI")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(os.environ, "JAVA_HOME", raising=False)
    monkeypatch.setenv("PATH", "sentinel-path")
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(BM25BackendError, match="index not found"):
        retriever._bm25_search("query", 3)
    assert requested_pyserini_imports == []
    assert "JAVA_HOME" not in os.environ
    assert os.environ["PATH"] == "sentinel-path"


def test_direct_nli_call_loads_pipeline_and_uses_text_pair() -> None:
    scorer = PassageScorer("fake", mock_mode=False)
    calls: List[Dict[str, Any]] = []
    load_count = 0

    def fake_pipeline(
        payload: Dict[str, str],
        *,
        top_k: Any,
        truncation: str,
        max_length: int,
    ) -> List[Dict[str, Any]]:
        calls.append(
            {
                "payload": payload,
                "top_k": top_k,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        return [
            {"label": "neutral", "score": 0.1},
            {"label": "ENTAILMENT", "score": 0.83},
            {"label": "contradiction", "score": 0.07},
        ]

    def fake_load() -> None:
        nonlocal load_count
        load_count += 1
        scorer._nli_pipeline = fake_pipeline

    scorer._load_nli = fake_load  # type: ignore[method-assign]
    score = scorer._nli_entailment("Paris is in France.", "Paris is in France.")

    assert score == pytest.approx(0.83)
    assert load_count == 1
    assert calls == [
        {
            "payload": {
                "text": "Paris is in France.",
                "text_pair": "Paris is in France.",
            },
            "top_k": None,
            "truncation": "only_first",
            "max_length": 512,
        }
    ]


def test_direct_nli_call_fails_explicitly_when_loader_produces_nothing() -> None:
    scorer = PassageScorer("fake", mock_mode=False)
    scorer._load_nli = lambda: None  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="initialization"):
        scorer._nli_entailment("premise", "hypothesis")


def _write_predictions(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_judgments(path: Path, judgments: List[Dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": analyze_scorer.JUDGMENTS_SCHEMA,
                "independent_of_scorer": True,
                "label_source": "fixture/independent-human-labels",
                "label_source_revision": "a" * 40,
                "judgments": judgments,
            }
        ),
        encoding="utf-8",
    )


def test_scorer_auc_uses_independent_ids_not_score_derived_labels(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    judgments_path = tmp_path / "judgments.json"
    _write_predictions(
        predictions_path,
        [
            {
                "example_id": "example-1",
                "scored_passages": [
                    {"id": "independent-gold", "final_score": 0.1},
                    {"id": "non-relevant", "final_score": 0.9},
                ],
            }
        ],
    )
    _write_judgments(
        judgments_path,
        [{"example_id": "example-1", "relevant_passage_ids": ["independent-gold"]}],
    )

    predictions, _ = analyze_scorer._read_predictions(predictions_path)
    judgments, provenance = analyze_scorer._load_independent_judgments(judgments_path)
    scores, labels, count = analyze_scorer._collect_independently_labeled_scores(
        predictions, judgments, 10
    )
    metrics = analyze_scorer._compute_metrics(scores, labels)

    assert count == 1
    assert labels == [1, 0]
    assert scores == [0.1, 0.9]
    assert metrics["roc_auc"] == 0.0
    assert metrics["average_precision"] == 0.5
    assert provenance["judgments_sha256"] == hashlib.sha256(judgments_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "records,match",
    [
        (
            [{"example_id": "example-1", "trusted_passages": [{"id": "p", "final_score": 1.0}]}],
            "scored_passages",
        ),
        (
            [
                {
                    "example_id": "example-1",
                    "scored_passages": [
                        {"id": "gold", "final_score": float("nan")},
                        {"id": "negative", "final_score": 0.2},
                    ],
                }
            ],
            "finite",
        ),
    ],
)
def test_scorer_analysis_rejects_biased_or_invalid_candidates(
    records: List[Dict[str, Any]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        analyze_scorer._collect_independently_labeled_scores(records, {"example-1": {"gold"}}, 10)


def test_scorer_main_failure_writes_no_numeric_error_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_scorer.py", "--output", str(output_path)],
    )
    with pytest.raises(ValueError, match="requires both"):
        analyze_scorer.main()
    assert not output_path.exists()


def test_offline_gating_analysis_counts_loaded_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_dir = tmp_path / "full"
    closed_dir = tmp_path / "closed"
    full_dir.mkdir()
    closed_dir.mkdir()
    _write_predictions(
        full_dir / "predictions.jsonl",
        [
            {
                "example_id": "example-1",
                "input": "Question?",
                "reference": "correct",
                "answer": "correct",
                "retrieval_triggered": False,
            }
        ],
    )
    _write_predictions(
        closed_dir / "predictions.jsonl",
        [{"example_id": "example-1", "input": "Question?", "answer": "wrong"}],
    )
    output_path = tmp_path / "gating.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_gating.py",
            "--full-run",
            str(full_dir),
            "--closedbook-run",
            str(closed_dir),
            "--output",
            str(output_path),
        ],
    )

    analyze_gating.main()

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["metrics"]["n_queries"] == 1
    assert len(output["per_query"]) == 1


def test_offline_gating_analysis_rejects_empty_runs_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_dir = tmp_path / "full"
    closed_dir = tmp_path / "closed"
    full_dir.mkdir()
    closed_dir.mkdir()
    (full_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (closed_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    output_path = tmp_path / "gating.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_gating.py",
            "--full-run",
            str(full_dir),
            "--closedbook-run",
            str(closed_dir),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="No gating decisions"):
        analyze_gating.main()
    assert not output_path.exists()

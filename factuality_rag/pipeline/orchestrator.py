"""
factuality_rag.pipeline.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end RAG pipeline: gating → retrieval → scoring → generation.

Provides both a stateless ``run_pipeline()`` convenience function
and a ``Pipeline`` class that loads components once and reuses them
across queries (fixing the re-instantiation performance bug).

Example (mock-mode)::

    >>> answer, passages, provenance, tag = run_pipeline(
    ...     "What is Python?", mock_mode=True)
    >>> tag in ("high", "medium", "low")
    True

Example (pipeline class)::

    >>> pipe = Pipeline(mock_mode=True)
    >>> ans, ps, prov, tag = pipe.run("What is DNA?")
    >>> tag in ("high", "medium", "low")
    True
"""

from __future__ import annotations

import logging
import math
import numbers
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from factuality_rag.resources import (
    DEFAULT_PIPELINE_CONFIG,
    experiment_config_identity,
    read_experiment_config_bytes,
)

logger = logging.getLogger(__name__)

# ``None`` is an explicit sentinel for the package-distributed default.  It
# must never be resolved through the current working directory.
_DEFAULT_CONFIG: Optional[str] = None


def _canonicalize_retrieved_passages(passages: Any) -> List[Dict[str, Any]]:
    """Validate retriever records and return copies with canonical string IDs.

    Passage IDs are a provenance boundary: only non-empty trimmed strings and
    non-boolean integers are accepted.  Integer IDs are converted to their
    decimal string representation, and collisions are rejected after that
    conversion so downstream scoring, prompts, artifacts, and provenance use
    one identity consistently.
    """
    if not isinstance(passages, list):
        raise TypeError("retriever must return a list of passage mappings")

    canonical_passages: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, passage in enumerate(passages):
        if not isinstance(passage, Mapping):
            raise TypeError(f"retrieved passage {index} must be a mapping")

        raw_id = passage.get("id")
        if isinstance(raw_id, bool):
            raise ValueError(f"retrieved passage {index} has an invalid boolean id")
        if isinstance(raw_id, str):
            if not raw_id or raw_id != raw_id.strip():
                raise ValueError(f"retrieved passage {index} id must be non-empty and trimmed")
            passage_id = raw_id
        elif isinstance(raw_id, int):
            passage_id = str(raw_id)
        else:
            raise ValueError(
                f"retrieved passage {index} id must be a trimmed string or non-boolean integer"
            )

        if passage_id in seen_ids:
            raise ValueError(
                f"duplicate retrieved passage id after canonicalization: {passage_id!r}"
            )
        seen_ids.add(passage_id)

        canonical_passage = dict(passage)
        canonical_passage["id"] = passage_id
        canonical_passages.append(canonical_passage)
    return canonical_passages


def _scored_passage_artifact(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the minimal JSON-safe pre-threshold scorer artifact.

    Only stable passage identifiers and final scalar scores cross this seam.
    Retriever metadata, model features, passage text, and any prompt-like fields
    remain internal to the pipeline.
    """
    artifact: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, passage in enumerate(passages):
        passage_id = passage.get("id")
        if not isinstance(passage_id, str) or not passage_id or passage_id != passage_id.strip():
            raise ValueError(f"scored passage {index} has an invalid id")
        if passage_id in seen_ids:
            raise ValueError(f"duplicate scored passage id: {passage_id!r}")
        seen_ids.add(passage_id)
        score = passage.get("final_score")
        if isinstance(score, bool) or not isinstance(score, numbers.Real):
            raise ValueError(f"scored passage {passage_id!r} has a non-numeric final_score")
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(
                f"scored passage {passage_id!r} final_score must be finite and in [0, 1]"
            )
        artifact.append({"id": passage_id, "final_score": numeric_score})
    return artifact


def _load_configured_learned_scorer(scorer_cfg: Mapping[str, Any]) -> Any:
    """Load the exact configured learned artifact without silent method drift.

    Pickle loading is denied unless the configuration supplies both an
    independently anchored metadata digest and the literal boolean unsafe
    opt-in. Missing, mismatched, or absent artifacts propagate as hard errors;
    a run labelled ``use_learned`` must never fall back to hand-tuned weights.
    """
    from factuality_rag.scorer.learned_scorer import LearnedScorer

    model_path = scorer_cfg.get("learned_model_path", "models/learned_scorer_logreg")
    if not isinstance(model_path, str) or not model_path or model_path != model_path.strip():
        raise ValueError("scorer.learned_model_path must be a non-empty trimmed string")
    allow_unsafe_pickle = scorer_cfg.get("allow_unsafe_pickle", False)
    if allow_unsafe_pickle is not True:
        raise ValueError(
            "pickle deserialization is disabled; set scorer.allow_unsafe_pickle=true only "
            "for an independently hash-anchored learned artifact"
        )
    metadata_sha256 = scorer_cfg.get("learned_model_metadata_sha256")
    if not isinstance(metadata_sha256, str):
        raise ValueError("scorer.learned_model_metadata_sha256 must be a SHA-256 string")
    return LearnedScorer.load(
        model_path,
        expected_metadata_sha256=metadata_sha256,
        allow_unsafe_pickle=True,
    )


def _load_config(
    config_path: Optional[str],
    *,
    default_resource: str = DEFAULT_PIPELINE_CONFIG,
) -> Dict[str, Any]:
    """Load a mapping-valued YAML config from an explicit path or package.

    Args:
        config_path: Explicit path to a user-supplied YAML file. ``None``
            selects *default_resource* from package data without consulting
            the current working directory.
        default_resource: Packaged config name used only for the ``None``
            sentinel.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If an explicit path or packaged resource is absent.
        ValueError: If the YAML is not UTF-8 or its root is not a mapping.
    """
    if config_path is None:
        raw_bytes = read_experiment_config_bytes(default_resource)
        source = experiment_config_identity(default_resource)
    else:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"config file does not exist or is not a file: {path}")
        raw_bytes = path.read_bytes()
        source = str(path)

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"config must be UTF-8: {source}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"config is not valid YAML: {source}") from exc
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError(f"config root must be a non-empty mapping: {source}")
    return dict(loaded)


def run_pipeline(
    query: str,
    k: int = 10,
    gate: bool = True,
    score_threshold: float = 0.4,
    config_path: Optional[str] = _DEFAULT_CONFIG,
    seed: int = 42,
    mock_mode: bool = False,
    *,
    config: Optional[Dict[str, Any]] = None,
    probe: Optional[Any] = None,
    retriever: Optional[Any] = None,
    scorer: Optional[Any] = None,
    learned_scorer: Optional[Any] = None,
    generator: Optional[Any] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
    """Run the full Factuality-first RAG pipeline.

    Steps:
        1. (Optional) Gating probe decides if retrieval is needed.
        2. Hybrid retrieval (dense + BM25).
        3. Optionally score passages (NLI + overlap + retriever fusion).
        4. Filter scored passages, or pass all retrieved passages in the
           explicit scorer-bypass baseline mode.
        5. Generate answer from trusted passages.
        6. Assign confidence tag.

    Pre-built components can be passed via keyword arguments to
    avoid re-instantiation on every call (see :class:`Pipeline`).

    Args:
        query: User question.
        k: Number of passages to retrieve.
        gate: Whether to apply the gating probe.
        score_threshold: Minimum ``final_score`` to keep a passage.
        config_path: Explicit experiment YAML path. When omitted, use the
            package-distributed sample config independent of the current
            working directory.
        seed: Random seed for reproducibility.
        mock_mode: If ``True``, all components run in mock-mode
                   (no model downloads, deterministic outputs).
        config: Optional pre-loaded config dict.  When provided
                this takes priority over *config_path* so callers
                can pass overridden settings without touching disk.
        probe: Optional pre-built :class:`~factuality_rag.gating.probe.GatingProbe`.
        retriever: Optional pre-built :class:`~factuality_rag.retriever.hybrid.HybridRetriever`.
        scorer: Optional pre-built :class:`~factuality_rag.scorer.passage.PassageScorer`.
        learned_scorer: Optional pre-authenticated learned scorer. Stateful
            callers pass this to avoid re-reading executable artifacts.
        generator: Optional pre-built :class:`~factuality_rag.generator.wrapper.Generator`.
        info: Optional mutable dict populated with run metadata
              (``retrieval_triggered``, ``gating_enabled``, ``scorer_enabled``,
              and, when scoring is enabled, the minimal pre-threshold
              ``scored_passages`` artifact). Useful for
              experiment tracking without changing the return type.

    Returns:
        Tuple of ``(answer, trusted_passages, provenance, confidence_tag)``:

        - **answer** – generated answer string.
        - **trusted_passages** – list of passage dicts with
          ``final_score`` ≥ *score_threshold*.
        - **provenance** – mapping ``{claim_idx: [passage_ids]}``.
        - **confidence_tag** – ``'high'`` | ``'medium'`` | ``'low'``.

    Example::

        >>> ans, ps, prov, tag = run_pipeline("test?", mock_mode=True)
        >>> isinstance(ans, str)
        True
    """
    import numpy as np

    # ── Seed everything ───────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    cfg = config if config is not None else _load_config(config_path)

    scorer_cfg = cfg.get("scorer", {})
    gating_cfg = cfg.get("gating", {})
    ret_cfg = cfg.get("retriever", {})
    if not isinstance(scorer_cfg, Mapping):
        raise TypeError("config.scorer must be a mapping")
    if not isinstance(gating_cfg, Mapping):
        raise TypeError("config.gating must be a mapping")
    if not isinstance(ret_cfg, Mapping):
        raise TypeError("config.retriever must be a mapping")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-blank string")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or int(k) < 0:
        raise ValueError("k must be a non-negative integer")
    k = int(k)
    if type(gate) is not bool:
        raise TypeError("gate must be exactly bool")
    if type(mock_mode) is not bool:
        raise TypeError("mock_mode must be exactly bool")
    if isinstance(score_threshold, bool) or not isinstance(score_threshold, numbers.Real):
        raise TypeError("score_threshold must be a real number")
    score_threshold = float(score_threshold)
    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be finite and in [0, 1]")
    rerank = ret_cfg.get("rerank", True)
    if type(rerank) is not bool:
        raise TypeError("retriever.rerank must be exactly bool")

    scorer_enabled = scorer_cfg.get("enabled", True)
    if type(scorer_enabled) is not bool:
        raise TypeError("scorer.enabled must be exactly bool")
    if not scorer_enabled and scorer_cfg.get("use_learned"):
        raise ValueError("scorer.use_learned cannot be enabled when scorer.enabled is false")
    if not scorer_enabled and (scorer is not None or learned_scorer is not None):
        raise ValueError("scorer components were supplied while scorer.enabled is false")

    # Authenticate executable learned-model artifacts before gating, retrieval,
    # or generation.  Delaying this until passages exist would let an empty
    # retrieval silently bypass a configuration that explicitly claims to use
    # the learned scorer.
    _learned_scorer = learned_scorer
    if scorer_cfg.get("use_learned"):
        if _learned_scorer is None:
            _learned_scorer = _load_configured_learned_scorer(scorer_cfg)
    elif _learned_scorer is not None:
        raise ValueError("learned_scorer was supplied but scorer.use_learned is not enabled")

    # ── Component imports (lazy, inside function) ─────────────
    from factuality_rag.gating.probe import GatingProbe
    from factuality_rag.generator.wrapper import Generator
    from factuality_rag.retriever.hybrid import HybridRetriever
    from factuality_rag.scorer.passage import PassageScorer

    # ── 1. Gating ─────────────────────────────────────────────
    retrieval_needed = True
    if gate:
        _probe = probe or GatingProbe(
            generator_model_hf=cfg.get("models", {}).get(
                "generator", "mistralai/Mistral-7B-Instruct-v0.3"
            ),
            mock_mode=mock_mode,
            temp=gating_cfg.get("softmax_temperature", 1.0),
        )
        # Probe the same no-context instruction prompt that generation would
        # consume if retrieval were skipped; raw-query logits are a different
        # distribution and cannot support the configured thresholds.
        gating_prompt = Generator._format_prompt(query, "")
        retrieval_needed = _probe.should_retrieve(
            gating_prompt,
            probe_tokens=gating_cfg.get("probe_tokens", 1),
            entropy_thresh=gating_cfg.get("entropy_thresh", 1.2),
            logit_gap_thresh=gating_cfg.get("logit_gap_thresh", 2.0),
        )
        logger.info("Gating decision: %s", "RETRIEVE" if retrieval_needed else "SKIP")

    # ── 2. Retrieval ──────────────────────────────────────────
    passages: List[Dict[str, Any]] = []
    if retrieval_needed and k > 0:
        if retriever is not None:
            _retriever = retriever
        elif mock_mode:
            _retriever = HybridRetriever.build_mock(
                n_docs=max(k * 2, 20),
                seed=seed,
                alpha=ret_cfg.get("alpha", 0.6),
            )
        else:
            idx_cfg = cfg.get("index", {})
            _retriever = HybridRetriever(
                faiss_index_path=idx_cfg.get("faiss_out", "indexes/faiss.index"),
                pyserini_index_path=idx_cfg.get("pyserini_out", "indexes/pyserini_dir"),
                corpus_path=idx_cfg.get("corpus_path"),
                embed_model=cfg.get("models", {}).get(
                    "dense_embedder", "sentence-transformers/all-mpnet-base-v2"
                ),
                alpha=ret_cfg.get("alpha", 0.6),
                normalize=ret_cfg.get("normalize", True),
            )
        retrieved_passages = _retriever.retrieve(query, k=k, rerank=rerank)
        passages = _canonicalize_retrieved_passages(retrieved_passages)

    # ── 3. Scoring ────────────────────────────────────────────
    weights = scorer_cfg.get("weights", {})
    _scorer = None
    if scorer_enabled:
        _scorer = scorer or PassageScorer(
            nli_model_hf=cfg.get("models", {}).get(
                "nli_verifier",
                "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            ),
            overlap_metric=scorer_cfg.get("overlap_metric", "token"),
            device=scorer_cfg.get("device", "cpu"),
            mock_mode=mock_mode,
            w_nli=weights.get("w_nli", 0.5),
            w_overlap=weights.get("w_overlap", 0.2),
            w_ret=weights.get("w_ret", 0.3),
            nli_mode=scorer_cfg.get("nli_mode", "passage"),
            nli_batch_size=scorer_cfg.get("nli_batch_size", 8),
            cross_encoder_model=scorer_cfg.get("cross_encoder_model", None),
        )
    if passages and _scorer is not None:
        passages = _scorer.score_passages(query, passages)

    # ── 3b. Learned scorer (optional) ─────────────────────────
    if passages and _learned_scorer is not None:
        passages = _learned_scorer.score_passages(passages)
        # Replace final_score with learned_score for downstream filtering
        for p in passages:
            p["final_score"] = p.get("learned_score", p.get("final_score", 0))
        logger.info("Applied authenticated learned scorer")

    # Validate the complete pre-threshold scoring artifact before filtering or
    # generation.  A malformed scorer result must fail before an external model
    # or service can observe a generation request.
    scored_passages_artifact = _scored_passage_artifact(passages) if scorer_enabled else None

    # ── 4. Filter ─────────────────────────────────────────────
    trusted = (
        [p for p in passages if p.get("final_score", 0) >= score_threshold]
        if scorer_enabled
        else list(passages)
    )

    # ── 5. Generate ───────────────────────────────────────────
    _gen = generator or Generator(
        model_name=cfg.get("models", {}).get("generator", "mistralai/Mistral-7B-Instruct-v0.3"),
        mock_mode=mock_mode,
    )
    context = "\n\n".join(p["text"] for p in trusted) if trusted else ""
    answer = _gen.generate(query, context=context)

    # ── 6. Provenance & confidence ────────────────────────────
    provenance = _build_provenance(answer, trusted, _scorer) if _scorer is not None else {}

    confidence_tag = _compute_confidence(trusted, retrieval_needed, gate)

    logger.info("Pipeline done: %d trusted passages, confidence=%s", len(trusted), confidence_tag)

    # Populate info dict for experiment tracking
    if info is not None:
        info["retrieval_triggered"] = retrieval_needed and k > 0
        info["gating_enabled"] = gate
        info["scorer_enabled"] = scorer_enabled
        if scored_passages_artifact is not None:
            info["scored_passages"] = scored_passages_artifact

    return answer, trusted, provenance, confidence_tag


def _compute_confidence(
    trusted: List[Dict[str, Any]],
    retrieval_needed: bool,
    gating_enabled: bool,
) -> str:
    """Determine a confidence tag from the pipeline output.

    Args:
        trusted: List of trusted passages.
        retrieval_needed: Whether retrieval was triggered.
        gating_enabled: Whether the gating probe was used.

    Returns:
        ``'high'``, ``'medium'``, or ``'low'``.

    Example::

        >>> _compute_confidence([], True, True)
        'low'
    """
    if not retrieval_needed and gating_enabled:
        # Gating skipped retrieval → model was confident, but we cannot
        # verify the answer without passages, so cap at "medium".
        return "medium"
    if not trusted:
        return "low"
    avg_score = sum(p.get("final_score", 0) for p in trusted) / len(trusted)
    if avg_score >= 0.7:
        return "high"
    elif avg_score >= 0.45:
        return "medium"
    return "low"


def _build_provenance(
    answer: str,
    trusted: List[Dict[str, Any]],
    scorer: Any,
) -> Dict[str, Any]:
    """Build heuristic sentence-unit → passage evidence links.

    Uses :func:`~factuality_rag.eval.metrics.compute_nli_claim_support` to
    split the answer into heuristic sentence units and match each *supported* unit to
    its best-supporting passage via the scorer's NLI function.  A passage
    with the highest score is not an evidence link unless that score clears the
    entailment threshold.

    Args:
        answer: Generated answer string.
        trusted: List of trusted passage dicts.
        scorer: :class:`~factuality_rag.scorer.passage.PassageScorer`
                instance (used for its ``_nli_entailment`` method).

    Returns:
        Dict mapping ``{claim_index: [best_passage_id]}``.

    Example::

        >>> from factuality_rag.scorer.passage import PassageScorer
        >>> s = PassageScorer("mock", mock_mode=True)
        >>> prov = _build_provenance("Hello.", [{"id":"0","text":"hi"}], s)
        >>> isinstance(prov, dict)
        True
    """
    if not trusted or not answer:
        return {}

    from factuality_rag.eval.metrics import compute_nli_claim_support

    nli_batch_fn = getattr(scorer, "_batch_nli_entailment", None)
    if nli_batch_fn is not None:
        scorer_dict = getattr(scorer, "__dict__", {})
        single_owner = next(
            (cls for cls in type(scorer).__mro__ if "_nli_entailment" in cls.__dict__),
            None,
        )
        batch_owner = next(
            (cls for cls in type(scorer).__mro__ if "_batch_nli_entailment" in cls.__dict__),
            None,
        )
        single_is_customized_without_batch = (
            "_nli_entailment" in scorer_dict and "_batch_nli_entailment" not in scorer_dict
        ) or (
            single_owner is not None
            and batch_owner is not None
            and single_owner is not batch_owner
            and issubclass(single_owner, batch_owner)
        )
        if single_is_customized_without_batch:
            # A custom scorer may override only the long-standing single-pair
            # hook while inheriting PassageScorer's newer batch method. Using
            # that inherited method would silently bypass the customization.
            nli_batch_fn = None

    result = compute_nli_claim_support(
        answer,
        trusted,
        nli_fn=scorer._nli_entailment,
        nli_batch_fn=nli_batch_fn,
    )

    provenance: Dict[str, Any] = {}
    for i, detail in enumerate(result.get("details", [])):
        pid = detail.get("best_passage_id")
        provenance[str(i)] = [pid] if detail.get("supported") is True and pid is not None else []

    return provenance


# ── Pipeline class (reusable, loads components once) ─────────


class Pipeline:
    """Reusable pipeline that instantiates component wrappers once.

    Heavy models and indexes remain lazy-loaded and cached. Reusing the
    wrappers avoids recreating them for every query.

    Args:
        config_path: Path to the experiment YAML config.
        mock_mode: If ``True``, all components run in mock-mode.
        seed: Default random seed.

    Example::

        >>> pipe = Pipeline(mock_mode=True)
        >>> ans, ps, prov, tag = pipe.run("What is DNA?")
        >>> tag in ("high", "medium", "low")
        True
    """

    def __init__(
        self,
        config_path: Optional[str] = _DEFAULT_CONFIG,
        mock_mode: bool = False,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        from factuality_rag.gating.probe import GatingProbe
        from factuality_rag.generator.wrapper import Generator
        from factuality_rag.retriever.hybrid import HybridRetriever
        from factuality_rag.scorer.passage import PassageScorer

        self.cfg = config if config is not None else _load_config(config_path)
        self._config_path = (
            config_path
            if config_path is not None
            else experiment_config_identity(DEFAULT_PIPELINE_CONFIG)
        )
        self.mock_mode = mock_mode
        self.seed = seed
        if type(mock_mode) is not bool:
            raise TypeError("mock_mode must be exactly bool")

        models_cfg = self.cfg.get("models", {})
        gating_cfg = self.cfg.get("gating", {})
        ret_cfg = self.cfg.get("retriever", {})
        scorer_cfg = self.cfg.get("scorer", {})
        idx_cfg = self.cfg.get("index", {})
        for name, value in (
            ("models", models_cfg),
            ("gating", gating_cfg),
            ("retriever", ret_cfg),
            ("scorer", scorer_cfg),
            ("index", idx_cfg),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"config.{name} must be a mapping")
        gate_enabled = gating_cfg.get("enabled", True)
        if type(gate_enabled) is not bool:
            raise TypeError("gating.enabled must be exactly bool")
        self.gate_enabled = gate_enabled
        rerank = ret_cfg.get("rerank", True)
        if type(rerank) is not bool:
            raise TypeError("retriever.rerank must be exactly bool")
        scorer_enabled = scorer_cfg.get("enabled", True)
        if type(scorer_enabled) is not bool:
            raise TypeError("scorer.enabled must be exactly bool")
        if not scorer_enabled and scorer_cfg.get("use_learned"):
            raise ValueError("scorer.use_learned cannot be enabled when scorer.enabled is false")
        self.scorer_enabled = scorer_enabled
        weights = scorer_cfg.get("weights", {})
        if not isinstance(weights, Mapping):
            raise TypeError("config.scorer.weights must be a mapping")
        score_threshold = scorer_cfg.get("score_threshold", 0.4)
        if isinstance(score_threshold, bool) or not isinstance(score_threshold, numbers.Real):
            raise TypeError("scorer.score_threshold must be a real number")
        self.score_threshold = float(score_threshold)
        if not math.isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("scorer.score_threshold must be finite and in [0, 1]")

        # ── Build components once ─────────────────────────────
        generator_id = models_cfg.get("generator", "mistralai/Mistral-7B-Instruct-v0.3")

        self.probe = GatingProbe(
            generator_model_hf=generator_id,
            mock_mode=mock_mode,
            temp=gating_cfg.get("softmax_temperature", 1.0),
        )

        top_k = ret_cfg.get("top_k", 10)
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral) or int(top_k) < 0:
            raise ValueError("retriever.top_k must be a non-negative integer")
        top_k = int(top_k)
        self.k = top_k

        # Closed-book mode: skip retriever/scorer when top_k == 0
        if top_k == 0 and not mock_mode:
            self.retriever = None
            self.scorer = None
        else:
            if mock_mode:
                self.retriever = HybridRetriever.build_mock(
                    n_docs=max(top_k * 2, 20),
                    seed=seed,
                    alpha=ret_cfg.get("alpha", 0.6),
                )
            else:
                self.retriever = HybridRetriever(
                    faiss_index_path=idx_cfg.get("faiss_out", "indexes/faiss.index"),
                    pyserini_index_path=idx_cfg.get("pyserini_out", "indexes/pyserini_dir"),
                    corpus_path=idx_cfg.get("corpus_path"),
                    embed_model=models_cfg.get(
                        "dense_embedder",
                        "sentence-transformers/all-mpnet-base-v2",
                    ),
                    alpha=ret_cfg.get("alpha", 0.6),
                    normalize=ret_cfg.get("normalize", True),
                )

            if scorer_enabled:
                self.scorer = PassageScorer(
                    nli_model_hf=models_cfg.get(
                        "nli_verifier",
                        "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                    ),
                    overlap_metric=scorer_cfg.get("overlap_metric", "token"),
                    device=scorer_cfg.get("device", "cpu"),
                    mock_mode=mock_mode,
                    w_nli=weights.get("w_nli", 0.5),
                    w_overlap=weights.get("w_overlap", 0.2),
                    w_ret=weights.get("w_ret", 0.3),
                    nli_mode=scorer_cfg.get("nli_mode", "passage"),
                    nli_batch_size=scorer_cfg.get("nli_batch_size", 8),
                    cross_encoder_model=scorer_cfg.get("cross_encoder_model", None),
                )
            else:
                self.scorer = None

        # ── Optional learned scorer ───────────────────────────
        self.learned_scorer = None
        if scorer_cfg.get("use_learned"):
            self.learned_scorer = _load_configured_learned_scorer(scorer_cfg)
            logger.info("Loaded authenticated learned scorer")

        self.generator = Generator(
            model_name=generator_id,
            mock_mode=mock_mode,
        )

        self._gating_cfg = gating_cfg
        logger.info("Pipeline initialised (mock_mode=%s, top_k=%d).", mock_mode, top_k)

    def run(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        gate: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], str]:
        """Run the pipeline on a single query, reusing loaded components.

        Args:
            query: User question.
            k: Override for retrieval depth (default from config).
            gate: Whether to apply the gating probe. ``None`` uses the
                configuration's exact ``gating.enabled`` value.
            score_threshold: Override for minimum ``final_score``.
            seed: Override for random seed.
            info: Optional mutable dict populated with run metadata.

        Returns:
            ``(answer, trusted_passages, provenance, confidence_tag)``

        Example::

            >>> pipe = Pipeline(mock_mode=True)
            >>> ans, ps, prov, tag = pipe.run("test?")
            >>> isinstance(ans, str)
            True
        """
        return run_pipeline(
            query,
            k=k if k is not None else self.k,
            gate=self.gate_enabled if gate is None else gate,
            score_threshold=(
                score_threshold if score_threshold is not None else self.score_threshold
            ),
            config=self.cfg,
            seed=seed if seed is not None else self.seed,
            mock_mode=self.mock_mode,
            probe=self.probe,
            retriever=self.retriever,
            scorer=self.scorer,
            learned_scorer=self.learned_scorer,
            generator=self.generator,
            info=info,
        )

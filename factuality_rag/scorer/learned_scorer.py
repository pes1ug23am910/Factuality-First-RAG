"""
factuality_rag.scorer.learned_scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight learned passage relevance classifier.

Trains a logistic regression or small MLP over feature vectors
``(nli_score, overlap_score, retriever_score_norm)`` produced by
:class:`~factuality_rag.scorer.passage.PassageScorer`.

**Scope:** The default scorer uses fixed, not-yet-tuned weights
``0.5 * NLI + 0.2 * overlap + 0.3 * retriever``. This module provides
classifier mechanics for externally supplied feature vectors and binary
labels. It does not create independent labels, select a valid data split, or
establish cross-dataset generalisation; those are experiment-protocol duties.

Supported classifiers:

- ``"logreg"`` — L2-regularised logistic regression (scikit-learn)
- ``"mlp"`` — 1-hidden-layer MLP (16 units, ReLU, scikit-learn)

Both are small classifiers; their quality and runtime depend on the supplied
artifact and are not claimed by this module.

Usage::

    >>> from factuality_rag.scorer.learned_scorer import LearnedScorer
    >>> ls = LearnedScorer(classifier_type="logreg")
    >>> X = [[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]]
    >>> y = [1, 0]
    >>> ls.fit(X, y)
    >>> preds = ls.predict_proba([[0.8, 0.4, 0.7]])
    >>> 0 <= preds[0] <= 1
    True
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Union, cast

import numpy as np

logger = logging.getLogger(__name__)

_ARTIFACT_SCHEMA = "factuality-rag.learned-scorer.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MAX_METADATA_BYTES = 64 * 1024
_MAX_MODEL_BYTES = 64 * 1024 * 1024


def _require_sha256(value: object, field: str) -> str:
    """Validate a non-placeholder lowercase SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise ValueError(f"{field} must be a non-zero lowercase SHA-256 digest")
    return value


def _read_bounded(path: Path, limit: int, artifact_name: str) -> bytes:
    """Read one immutable byte snapshot while enforcing a conservative size cap."""
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{artifact_name} exceeds the {limit}-byte safety limit")
    return raw


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for the authenticated metadata."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _strict_json_object(raw: bytes) -> Dict[str, Any]:
    """Parse strict UTF-8 JSON, rejecting duplicate keys and non-finite constants."""

    def reject_duplicates(pairs: List[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"metadata contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"metadata contains non-finite JSON constant {value}")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("metadata.json must be valid UTF-8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("metadata.json must be valid strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("metadata.json must contain a JSON object")
    return payload


class LearnedScorer:
    """Lightweight learned scorer for passage relevance fusion.

    The classifier takes 3-dimensional feature vectors
    ``[nli_score, overlap_score, retriever_score_norm]`` and
    outputs a probability that the passage is relevant/faithful.

    Args:
        classifier_type: ``"logreg"`` or ``"mlp"``.
        random_state: Random seed for reproducibility.

    Example::

        >>> ls = LearnedScorer("logreg")
        >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
        >>> 0 <= ls.predict_proba([[0.7, 0.3, 0.6]])[0] <= 1
        True
    """

    FEATURE_NAMES = ["nli_score", "overlap_score", "retriever_score_norm"]

    def __init__(
        self,
        classifier_type: str = "logreg",
        random_state: int = 42,
    ) -> None:
        if classifier_type not in {"logreg", "mlp"}:
            raise ValueError("classifier_type must be 'logreg' or 'mlp'")
        if type(random_state) is not int:
            raise TypeError("random_state must be an integer")
        self.classifier_type = classifier_type
        self.random_state = random_state
        self._model: Any = None
        self._fitted = False

    def _build_model(self) -> Any:
        """Create the sklearn classifier.

        Returns:
            A scikit-learn estimator.
        """
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.neural_network import MLPClassifier  # type: ignore[import-untyped]

        if self.classifier_type == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                max_iter=500,
                random_state=self.random_state,
                early_stopping=False,
            )
        # Default: logistic regression
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: Union[List[List[float]], "np.ndarray"],
        y: Union[List[int], "np.ndarray"],
    ) -> "LearnedScorer":
        """Train the classifier on feature vectors and binary labels.

        Args:
            X: Feature matrix ``(n_samples, 3)`` — each row is
               ``[nli_score, overlap_score, retriever_score_norm]``.
            y: Binary labels ``(n_samples,)`` — 1 = relevant/supported,
               0 = irrelevant/refuted.

        Returns:
            Self, for chaining.

        Example::

            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8]], [1])
            LearnedScorer(classifier_type='logreg')
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)

        if X_arr.ndim != 2 or X_arr.shape[1] != 3:
            raise ValueError(f"Expected X shape (n, 3), got {X_arr.shape}")

        self._model = self._build_model()
        self._model.fit(X_arr, y_arr)
        self._fitted = True

        # Log learned weights for interpretability
        if hasattr(self._model, "coef_"):
            coefs = self._model.coef_[0]
            logger.info(
                "Learned weights: nli=%.3f  overlap=%.3f  ret=%.3f  (intercept=%.3f)",
                coefs[0],
                coefs[1],
                coefs[2],
                self._model.intercept_[0],
            )

        logger.info(
            "LearnedScorer trained (%s) on %d samples (pos=%.1f%%).",
            self.classifier_type,
            len(y_arr),
            100 * y_arr.mean(),
        )
        return self

    def predict_proba(
        self,
        X: Union[List[List[float]], "np.ndarray"],
    ) -> "np.ndarray":
        """Predict relevance probability for each sample.

        Args:
            X: Feature matrix ``(n_samples, 3)``.

        Returns:
            Array of probabilities ``(n_samples,)`` in [0, 1].

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Example::

            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> probs = ls.predict_proba([[0.8, 0.4, 0.7]])
            >>> len(probs) == 1
            True
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        proba = np.asarray(self._model.predict_proba(X_arr), dtype=np.float64)
        # Return probability of class 1 (relevant)
        return cast(np.ndarray, proba[:, 1])

    def score_passages(
        self,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply the learned scorer to pre-featurised passages.

        Expects each passage dict to already contain ``nli_score``,
        ``overlap_score``, and a normalised retriever score.
        Adds ``learned_score`` to each passage.

        Args:
            passages: List of passage dicts with existing scores.

        Returns:
            Same list with ``learned_score`` added.

        Example::

            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> ps = [{"nli_score": 0.8, "overlap_score": 0.3, "combined_score": 0.6}]
            >>> out = ls.score_passages(ps)
            >>> "learned_score" in out[0]
            True
        """
        if not passages:
            return passages

        # Build feature matrix from passage dicts
        ret_scores = [p.get("combined_score", 0.0) for p in passages]
        ret_min = min(ret_scores) if ret_scores else 0.0
        ret_max = max(ret_scores) if ret_scores else 1.0

        X = []
        for p in passages:
            nli = p.get("nli_score", 0.0)
            overlap = p.get("overlap_score", 0.0)
            ret_raw = p.get("combined_score", 0.0)
            ret_norm = (ret_raw - ret_min) / (ret_max - ret_min) if ret_max > ret_min else 0.5
            X.append([nli, overlap, ret_norm])

        probs = self.predict_proba(X)
        for p, prob in zip(passages, probs):
            p["learned_score"] = float(prob)

        return passages

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> str:
        """Save the trained model and return its metadata SHA-256 trust anchor.

        Args:
            path: Directory path. Creates ``model.pkl`` and
                   ``metadata.json`` inside.

        Returns:
            SHA-256 of the exact canonical ``metadata.json`` bytes. Preserve
            this digest in an independently controlled configuration or run
            ledger before loading the pickle artifact.

        Example::

            >>> import tempfile, os
            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> d = tempfile.mkdtemp()
            >>> metadata_sha256 = ls.save(d)
            >>> os.path.exists(os.path.join(d, "model.pkl"))
            True
            >>> len(metadata_sha256)
            64
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() before .save().")

        import sklearn  # type: ignore[import-untyped]

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        model_bytes = pickle.dumps(self._model, protocol=pickle.HIGHEST_PROTOCOL)
        if len(model_bytes) > _MAX_MODEL_BYTES:
            raise ValueError(f"model pickle exceeds the {_MAX_MODEL_BYTES}-byte safety limit")
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()

        meta: Dict[str, Any] = {
            "schema": _ARTIFACT_SCHEMA,
            "classifier_type": self.classifier_type,
            "random_state": self.random_state,
            "feature_names": self.FEATURE_NAMES,
            "model_file": "model.pkl",
            "model_sha256": model_sha256,
            "model_size_bytes": len(model_bytes),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
        # Store learned weights for interpretability
        if hasattr(self._model, "coef_"):
            meta["learned_weights"] = {
                name: float(w) for name, w in zip(self.FEATURE_NAMES, self._model.coef_[0])
            }
            meta["intercept"] = float(self._model.intercept_[0])

        metadata_bytes = _canonical_json_bytes(meta)
        metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()

        # Metadata is written last: an interrupted save leaves an artifact that
        # cannot be authenticated rather than a manifest pointing at partial bytes.
        (out / "model.pkl").write_bytes(model_bytes)
        (out / "metadata.json").write_bytes(metadata_bytes)

        logger.info(
            "Saved learned scorer → %s (metadata_sha256=%s, model_sha256=%s)",
            out,
            metadata_sha256,
            model_sha256,
        )
        return metadata_sha256

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        expected_metadata_sha256: str,
        allow_unsafe_pickle: bool = False,
    ) -> "LearnedScorer":
        """Load an explicitly trusted, externally hash-bound pickle artifact.

        Args:
            path: Directory containing ``model.pkl`` and ``metadata.json``.
            expected_metadata_sha256: Independently recorded SHA-256 of the
                exact metadata bytes. The authenticated metadata transitively
                binds the model pickle digest and byte length.
            allow_unsafe_pickle: Explicit acknowledgement that pickle can run
                arbitrary code. This must be the boolean ``True``; truthy
                substitutes are rejected.

        Returns:
            A fitted :class:`LearnedScorer`.

        Example::

            >>> import tempfile
            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> d = tempfile.mkdtemp()
            >>> digest = ls.save(d)
            >>> ls2 = LearnedScorer.load(
            ...     d,
            ...     expected_metadata_sha256=digest,
            ...     allow_unsafe_pickle=True,
            ... )
            >>> ls2._fitted
            True
        """
        if allow_unsafe_pickle is not True:
            raise ValueError(
                "pickle deserialization is disabled; pass allow_unsafe_pickle=True only "
                "for an artifact whose metadata SHA-256 was independently trusted"
            )
        expected_digest = _require_sha256(
            expected_metadata_sha256,
            "expected_metadata_sha256",
        )

        d = Path(path)
        metadata_bytes = _read_bounded(
            d / "metadata.json",
            _MAX_METADATA_BYTES,
            "learned-scorer metadata",
        )
        actual_metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
        if not secrets.compare_digest(expected_digest, actual_metadata_digest):
            raise ValueError("metadata.json SHA-256 does not match the trusted digest")

        meta = _strict_json_object(metadata_bytes)
        required_keys = {
            "schema",
            "classifier_type",
            "random_state",
            "feature_names",
            "model_file",
            "model_sha256",
            "model_size_bytes",
            "python_version",
            "numpy_version",
            "scikit_learn_version",
        }
        allowed_keys = required_keys | {"learned_weights", "intercept"}
        if not required_keys.issubset(meta) or not set(meta).issubset(allowed_keys):
            raise ValueError("metadata.json has an incomplete or unsupported schema")
        if meta["schema"] != _ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported learned-scorer artifact schema: {meta['schema']!r}")
        classifier_type = meta["classifier_type"]
        if classifier_type not in {"logreg", "mlp"}:
            raise ValueError("metadata classifier_type must be 'logreg' or 'mlp'")
        random_state = meta["random_state"]
        if type(random_state) is not int:
            raise ValueError("metadata random_state must be an integer")
        if meta["feature_names"] != cls.FEATURE_NAMES:
            raise ValueError("metadata feature_names do not match the supported feature contract")
        if meta["model_file"] != "model.pkl":
            raise ValueError("metadata model_file must be exactly 'model.pkl'")
        model_sha256 = _require_sha256(meta["model_sha256"], "metadata model_sha256")
        model_size = meta["model_size_bytes"]
        if type(model_size) is not int or not 0 < model_size <= _MAX_MODEL_BYTES:
            raise ValueError("metadata model_size_bytes is invalid")

        import sklearn  # type: ignore[import-untyped]

        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        expected_versions = {
            "python_version": current_python,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
        for field, current in expected_versions.items():
            recorded = meta[field]
            if not isinstance(recorded, str) or recorded != current:
                raise ValueError(
                    f"metadata {field}={recorded!r} is incompatible with current {current!r}"
                )

        model_bytes = _read_bounded(d / "model.pkl", _MAX_MODEL_BYTES, "learned-scorer model")
        if len(model_bytes) != model_size:
            raise ValueError("model.pkl byte length does not match authenticated metadata")
        actual_model_digest = hashlib.sha256(model_bytes).hexdigest()
        if not secrets.compare_digest(model_sha256, actual_model_digest):
            raise ValueError("model.pkl SHA-256 does not match authenticated metadata")

        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.neural_network import MLPClassifier  # type: ignore[import-untyped]

        # Pickle remains executable. Reaching this line requires both an
        # explicit unsafe opt-in and two verified byte snapshots rooted in an
        # externally supplied metadata digest.
        model = pickle.loads(model_bytes)  # noqa: S301
        expected_type = LogisticRegression if classifier_type == "logreg" else MLPClassifier
        if type(model) is not expected_type:
            raise ValueError(
                f"deserialized model type {type(model).__name__!r} does not match "
                f"classifier_type {classifier_type!r}"
            )
        if getattr(model, "n_features_in_", None) != len(cls.FEATURE_NAMES):
            raise ValueError("deserialized model does not use exactly three supported features")
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.shape != (2,) or not np.array_equal(classes, np.asarray([0, 1])):
            raise ValueError("deserialized model classes must be exactly [0, 1]")
        probe = np.asarray(model.predict_proba(np.zeros((1, len(cls.FEATURE_NAMES)))), dtype=float)
        if (
            probe.shape != (1, 2)
            or not np.isfinite(probe).all()
            or (probe < 0).any()
            or (probe > 1).any()
            or not np.allclose(probe.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("deserialized model failed the probability-output sanity check")

        scorer = cls(
            classifier_type=classifier_type,
            random_state=random_state,
        )
        scorer._model = model
        scorer._fitted = True

        logger.info(
            "Loaded trusted learned scorer from %s (metadata_sha256=%s)",
            d,
            actual_metadata_digest,
        )
        return scorer

    def evaluate(
        self,
        X: Union[List[List[float]], "np.ndarray"],
        y: Union[List[int], "np.ndarray"],
    ) -> Dict[str, float]:
        """Evaluate the model on a held-out set.

        Args:
            X: Feature matrix ``(n_samples, 3)``.
            y: True binary labels ``(n_samples,)``.

        Returns:
            Dict with ``accuracy``, ``auc_roc``, ``precision``,
            ``recall``, ``f1``.

        Example::

            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> m = ls.evaluate([[0.8, 0.4, 0.7], [0.2, 0.1, 0.1]], [1, 0])
            >>> "auc_roc" in m
            True
        """
        from sklearn.metrics import (  # type: ignore[import-untyped]
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_arr = np.asarray(y, dtype=np.int32)
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_arr, preds)),
            "precision": float(precision_score(y_arr, preds, zero_division=0)),
            "recall": float(recall_score(y_arr, preds, zero_division=0)),
            "f1": float(f1_score(y_arr, preds, zero_division=0)),
        }

        # AUC-ROC only if both classes present
        if len(np.unique(y_arr)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(y_arr, probs))
        else:
            metrics["auc_roc"] = float("nan")

        return metrics

    def __repr__(self) -> str:
        return f"LearnedScorer(classifier_type='{self.classifier_type}')"

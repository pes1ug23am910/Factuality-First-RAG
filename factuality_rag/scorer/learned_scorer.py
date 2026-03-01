"""
factuality_rag.scorer.learned_scorer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight learned passage relevance classifier.

Trains a logistic regression or small MLP over feature vectors
``(nli_score, overlap_score, retriever_score_norm)`` produced by
:class:`~factuality_rag.scorer.passage.PassageScorer`.

**Motivation:** The default scorer uses hand-tuned weights
``0.5 * NLI + 0.2 * overlap + 0.3 * retriever`` which work
reasonably but are not learned from data.  This module learns
the fusion weights from labelled (claim, evidence, label) triples
(e.g. FEVER train) and **evaluates cross-dataset** (e.g. on NQ-Open),
demonstrating generalisation.

Supported classifiers:

- ``"logreg"`` — L2-regularised logistic regression (scikit-learn)
- ``"mlp"`` — 1-hidden-layer MLP (16 units, ReLU, scikit-learn)

Both are tiny models (~50 parameters) that train in seconds.

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

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


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
        self.classifier_type = classifier_type
        self.random_state = random_state
        self._model: Any = None
        self._fitted = False

    def _build_model(self) -> Any:
        """Create the sklearn classifier.

        Returns:
            A scikit-learn estimator.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

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
            raise ValueError(
                f"Expected X shape (n, 3), got {X_arr.shape}"
            )

        self._model = self._build_model()
        self._model.fit(X_arr, y_arr)
        self._fitted = True

        # Log learned weights for interpretability
        if hasattr(self._model, "coef_"):
            coefs = self._model.coef_[0]
            logger.info(
                "Learned weights: nli=%.3f  overlap=%.3f  ret=%.3f  (intercept=%.3f)",
                coefs[0], coefs[1], coefs[2], self._model.intercept_[0],
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
        proba = self._model.predict_proba(X_arr)
        # Return probability of class 1 (relevant)
        return proba[:, 1]

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
            ret_norm = (
                (ret_raw - ret_min) / (ret_max - ret_min)
                if ret_max > ret_min
                else 0.5
            )
            X.append([nli, overlap, ret_norm])

        probs = self.predict_proba(X)
        for p, prob in zip(passages, probs):
            p["learned_score"] = float(prob)

        return passages

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk.

        Args:
            path: Directory path. Creates ``model.pkl`` and
                  ``metadata.json`` inside.

        Example::

            >>> import tempfile, os
            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> d = tempfile.mkdtemp()
            >>> ls.save(d)
            >>> os.path.exists(os.path.join(d, "model.pkl"))
            True
        """
        import pickle

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)

        meta = {
            "classifier_type": self.classifier_type,
            "random_state": self.random_state,
            "feature_names": self.FEATURE_NAMES,
        }
        # Store learned weights for interpretability
        if hasattr(self._model, "coef_"):
            meta["learned_weights"] = {
                name: float(w)
                for name, w in zip(self.FEATURE_NAMES, self._model.coef_[0])
            }
            meta["intercept"] = float(self._model.intercept_[0])

        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved learned scorer → %s", out)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LearnedScorer":
        """Load a saved model from disk.

        Args:
            path: Directory containing ``model.pkl`` and ``metadata.json``.

        Returns:
            A fitted :class:`LearnedScorer`.

        Example::

            >>> import tempfile
            >>> ls = LearnedScorer("logreg")
            >>> ls.fit([[0.9, 0.5, 0.8], [0.1, 0.1, 0.2]], [1, 0])
            LearnedScorer(classifier_type='logreg')
            >>> d = tempfile.mkdtemp()
            >>> ls.save(d)
            >>> ls2 = LearnedScorer.load(d)
            >>> ls2._fitted
            True
        """
        import pickle

        d = Path(path)
        with open(d / "metadata.json") as f:
            meta = json.load(f)

        scorer = cls(
            classifier_type=meta["classifier_type"],
            random_state=meta.get("random_state", 42),
        )

        with open(d / "model.pkl", "rb") as f:
            scorer._model = pickle.load(f)  # noqa: S301
        scorer._fitted = True

        logger.info("Loaded learned scorer from %s", d)
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
        from sklearn.metrics import (
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

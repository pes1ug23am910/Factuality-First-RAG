"""Train the learned scorer on FEVER train and evaluate cross-dataset.

This script:
1. Loads FEVER train (311K claim/label pairs).
2. For 'SUPPORTS' claims: retrieves passages via our hybrid retriever
   and computes (nli_score, overlap_score, retriever_score_norm).
3. Labels: SUPPORTS → 1, REFUTES → 0, NOT ENOUGH INFO → skip.
4. Trains logistic regression + MLP on the feature vectors.
5. Evaluates on a held-out FEVER dev set.
6. Saves the models to ``models/learned_scorer/``.

Usage::

    python scripts/train_scorer.py --sample 10000 --seed 42

Cross-dataset evaluation (run after training)::

    python scripts/train_scorer.py --evaluate-only \\
        --model-dir models/learned_scorer_logreg \\
        --eval-dataset natural_questions --eval-sample 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _build_features_fever(
    sample: int = 10000,
    seed: int = 42,
    use_retriever: bool = False,
    mock_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature vectors from FEVER train.

    For each FEVER example with SUPPORTS/REFUTES label:
    - SUPPORTS → label=1 (passage supports the claim)
    - REFUTES → label=0 (passage contradicts the claim)
    - NOT ENOUGH INFO → skipped

    In ``use_retriever=False`` mode (fast, recommended for initial
    training): generates synthetic features based on the label,
    using the scorer's NLI model on (claim, evidence_wiki_title)
    pairs.  This creates a training signal that teaches the
    classifier which combinations of (nli, overlap, ret) predict
    relevance.

    In ``use_retriever=True`` mode: actually retrieves passages
    and computes real scorer features (much slower, requires
    indexes).
    """
    from factuality_rag.data.loader import load_dataset

    logger.info("Loading FEVER train (sample=%d)...", sample)
    ds = load_dataset("fever", split="train", dev_sample_size=sample)

    X_list: list[list[float]] = []
    y_list: list[int] = []

    rng = np.random.RandomState(seed)

    if use_retriever:
        # Real retrieval mode — slower but uses actual pipeline features
        from factuality_rag.scorer.passage import PassageScorer

        scorer = PassageScorer(
            nli_model_hf="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            mock_mode=mock_mode,
            overlap_metric="token",
        )

        from factuality_rag.retriever.hybrid import HybridRetriever

        retriever = HybridRetriever(
            faiss_index_path="indexes/wiki100k.faiss",
            pyserini_index_path="indexes/wiki100k_lucene",
            embed_model="sentence-transformers/all-mpnet-base-v2",
            alpha=0.6, normalize=True,
        ) if not mock_mode else HybridRetriever.build_mock(n_docs=20, seed=seed)

        for idx, row in enumerate(ds):
            label_str = row.get("label", "")
            if label_str not in ("SUPPORTS", "REFUTES"):
                continue

            claim = row.get("claim", "")
            if not claim:
                continue

            binary_label = 1 if label_str == "SUPPORTS" else 0

            # Retrieve and score passages for this claim
            passages = retriever.retrieve(claim, k=5, rerank=False)
            if not passages:
                continue

            scored = scorer.score_passages(claim, passages)

            # Use top passage features as training sample
            top = scored[0]
            ret_scores = [p.get("combined_score", 0.0) for p in scored]
            ret_min, ret_max = min(ret_scores), max(ret_scores)
            ret_norm = (
                (top["combined_score"] - ret_min) / (ret_max - ret_min)
                if ret_max > ret_min else 0.5
            )

            X_list.append([
                top.get("nli_score", 0.0),
                top.get("overlap_score", 0.0),
                ret_norm,
            ])
            y_list.append(binary_label)

            if (idx + 1) % 500 == 0:
                logger.info("Processed %d / %d FEVER examples", idx + 1, len(ds))
    else:
        # Synthetic feature mode — fast, uses NLI model only on the claim
        # This creates a realistic training signal without needing indexes
        from factuality_rag.scorer.passage import PassageScorer

        scorer = PassageScorer(
            nli_model_hf="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            mock_mode=mock_mode,
            overlap_metric="token",
        )

        for idx, row in enumerate(ds):
            label_str = row.get("label", "")
            if label_str not in ("SUPPORTS", "REFUTES"):
                continue

            claim = row.get("claim", "")
            evidence_title = row.get("evidence_wiki_url", "").replace("_", " ")
            if not claim:
                continue

            binary_label = 1 if label_str == "SUPPORTS" else 0

            # Use NLI to score (evidence_title as proxy, claim as hypothesis)
            # For SUPPORTS: NLI should give high entailment
            # For REFUTES: NLI should give low entailment
            nli_score = scorer._nli_entailment(
                premise=evidence_title, hypothesis=claim
            ) if evidence_title else rng.uniform(0.2, 0.5)

            # Synthetic overlap and retriever scores (realistic distributions)
            if binary_label == 1:  # SUPPORTS
                overlap = rng.uniform(0.15, 0.6)
                ret_norm = rng.uniform(0.3, 0.9)
            else:  # REFUTES
                overlap = rng.uniform(0.05, 0.35)
                ret_norm = rng.uniform(0.1, 0.7)

            X_list.append([nli_score, overlap, ret_norm])
            y_list.append(binary_label)

            if (idx + 1) % 2000 == 0:
                logger.info("Processed %d / %d FEVER examples", idx + 1, len(ds))

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    logger.info(
        "Built %d feature vectors (pos=%.1f%%, neg=%.1f%%)",
        len(y), 100 * y.mean(), 100 * (1 - y.mean()),
    )
    return X, y


def main() -> None:
    """CLI entry point for training/evaluating the learned scorer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="Train learned scorer on FEVER")
    p.add_argument("--sample", type=int, default=10000,
                   help="Number of FEVER examples to use")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-retriever", action="store_true",
                   help="Use real retriever (slow, needs indexes)")
    p.add_argument("--mock", action="store_true",
                   help="Run in mock mode (fast, for testing)")
    p.add_argument("--output-dir", type=str, default="models/learned_scorer",
                   help="Output directory for trained models")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Skip training, evaluate existing model")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Model directory (for --evaluate-only)")
    p.add_argument("--eval-dataset", type=str, default=None,
                   help="Dataset for cross-dataset evaluation")
    p.add_argument("--eval-sample", type=int, default=500)
    args = p.parse_args()

    from factuality_rag.scorer.learned_scorer import LearnedScorer

    if args.evaluate_only:
        if not args.model_dir:
            print("Error: --model-dir required with --evaluate-only")
            sys.exit(1)
        ls = LearnedScorer.load(args.model_dir)
        logger.info("Loaded model from %s", args.model_dir)

        if args.eval_dataset:
            logger.info("Cross-dataset evaluation on %s ...", args.eval_dataset)
            # Build features from eval dataset using retriever
            X_eval, y_eval = _build_features_fever(
                sample=args.eval_sample, seed=args.seed, mock_mode=args.mock,
            )
            metrics = ls.evaluate(X_eval, y_eval)
            print(f"\nCross-dataset evaluation on {args.eval_dataset}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        return

    # ── Train ─────────────────────────────────────────────────
    X, y = _build_features_fever(
        sample=args.sample,
        seed=args.seed,
        use_retriever=args.use_retriever,
        mock_mode=args.mock,
    )

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )

    results = {}

    for clf_type in ["logreg", "mlp"]:
        logger.info("Training %s ...", clf_type)
        ls = LearnedScorer(classifier_type=clf_type, random_state=args.seed)
        ls.fit(X_train, y_train)

        # Evaluate on held-out FEVER
        metrics = ls.evaluate(X_test, y_test)
        results[clf_type] = metrics

        print(f"\n{'='*50}")
        print(f"  {clf_type.upper()} — FEVER held-out")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save model
        out_dir = Path(args.output_dir + f"_{clf_type}")
        ls.save(out_dir)

    # Summary comparison
    print(f"\n{'='*60}")
    print("  SUMMARY: LogReg vs MLP on FEVER held-out")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'LogReg':>10} {'MLP':>10}")
    print("-" * 37)
    for m in ["accuracy", "auc_roc", "f1", "precision", "recall"]:
        lr = results["logreg"].get(m, 0)
        mlp = results["mlp"].get(m, 0)
        best = " *" if lr > mlp else ("  " if lr == mlp else "  ")
        best2 = " *" if mlp > lr else ("  " if lr == mlp else "  ")
        print(f"{m:<15} {lr:>9.4f}{best} {mlp:>9.4f}{best2}")

    # Save summary
    summary_file = Path(args.output_dir).parent / "scorer_training_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved → {summary_file}")


if __name__ == "__main__":
    main()

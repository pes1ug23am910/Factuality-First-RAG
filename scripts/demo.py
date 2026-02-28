#!/usr/bin/env python
"""
scripts/demo.py
~~~~~~~~~~~~~~~~
Quick demo of the full Factuality-first RAG pipeline in mock-mode.

Usage::

    python scripts/demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factuality_rag.pipeline.orchestrator import run_pipeline
from factuality_rag.experiment_runner import run as experiment_run


def main() -> None:
    """Run a quick mock-mode demo."""
    print("=" * 60)
    print("  Factuality-first RAG – Mock Demo")
    print("=" * 60)

    queries = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "Explain photosynthesis briefly.",
    ]

    for q in queries:
        print(f"\n>>> Query: {q}")
        answer, trusted, provenance, confidence = run_pipeline(
            query=q,
            k=5,
            gate=True,
            mock_mode=True,
        )
        print(f"    Answer:       {answer}")
        print(f"    Confidence:   {confidence}")
        print(f"    Trusted:      {len(trusted)} passage(s)")
        print(f"    Provenance:   {json.dumps(provenance, indent=6)}")

    # Run full experiment
    print("\n" + "=" * 60)
    print("  Running experiment_runner ...")
    print("=" * 60)

    result = experiment_run(
        config={"seed": 42},
        queries=queries,
        mock_mode=True,
    )
    print(f"\n  Run ID:    {result['run_id']}")
    print(f"  Run dir:   {result['run_dir']}")
    print(f"  Metrics:   {json.dumps(result['metrics'], indent=4)}")
    print("\nDone!")


if __name__ == "__main__":
    main()

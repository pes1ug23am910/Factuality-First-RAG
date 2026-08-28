"""Build an immutable family-disjoint split manifest from normalized JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from factuality_rag.data.splits import build_group_disjoint_split, write_split_manifest


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=_reject_nonfinite_number,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            examples.append(value)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Normalized source JSONL")
    parser.add_argument("output", type=Path, help="New immutable manifest path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--tuning-ratio", type=float, required=True)
    parser.add_argument("--sealed-final-ratio", type=float, required=True)
    parser.add_argument("--strata-key", default=None)
    args = parser.parse_args()

    manifest = build_group_disjoint_split(
        _load_jsonl(args.input),
        ratios={
            "train": args.train_ratio,
            "tuning": args.tuning_ratio,
            "sealed_final": args.sealed_final_ratio,
        },
        seed=args.seed,
        strata_key=args.strata_key,
    )
    status = write_split_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "status": status,
                "manifest_sha256": manifest["manifest_sha256"],
                "source": manifest["source"],
                "partition_counts": {
                    partition: record["example_count"]
                    for partition, record in manifest["partitions"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

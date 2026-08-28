"""Deterministic, transitive group-disjoint dataset splits.

The builder operates on the complete normalized source snapshot.  Callers may
sample only after the manifest assigns whole connected components to a
partition.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from factuality_rag.reproducibility import (
    seal_manifest,
    sha256_json,
    verify_manifest,
    write_immutable_json,
)

PARTITIONS = ("train", "tuning", "sealed_final")
SPLIT_MANIFEST_SCHEMA = "factuality-rag.split-manifest.v1"
SPLIT_ALGORITHM = "transitive-family-greedy-balance-v1"

_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ZERO_SHA256 = "0" * 64
_PURPOSES = {"training", "tuning", "smoke", "final_evaluation"}
_TOP_LEVEL_FIELDS = {
    "schema",
    "algorithm",
    "seed",
    "ratios",
    "strata_key",
    "policy",
    "source",
    "partitions",
    "components",
}
_POLICY = {
    "split_complete_snapshot_before_sampling": True,
    "transitive_family_closure": True,
    "tie_breaker": "sha256",
    "sealed_final_development_access": "forbidden",
}
_SOURCE_FIELDS = {
    "example_count",
    "component_count",
    "source_snapshot_sha256",
    "normalized_examples_sha256",
}
_PARTITION_FIELDS = {
    "example_ids",
    "component_ids",
    "example_count",
    "component_count",
    "strata_counts",
}
_COMPONENT_FIELDS = {
    "example_ids",
    "family_ids",
    "example_count",
    "strata_counts",
    "partition",
}
PathLike = Union[str, Path]


class SplitValidationError(ValueError):
    """Raised when split inputs or a split manifest fail closed."""


def _is_sequence(value: object) -> bool:
    return isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray))


def _normalize_ratios(ratios: Mapping[str, float]) -> Dict[str, float]:
    if not isinstance(ratios, MappingABC):
        raise TypeError("ratios must be a mapping")
    if set(ratios) != set(PARTITIONS):
        raise SplitValidationError("ratios must contain exactly train, tuning, and sealed_final")
    normalized: Dict[str, float] = {}
    for partition in PARTITIONS:
        value = ratios[partition]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"ratio for {partition} must be a finite positive number")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise SplitValidationError(f"ratio for {partition} must be a finite positive number")
        normalized[partition] = number
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitValidationError("split ratios must sum to 1.0")
    return normalized


def _portable_text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SplitValidationError(f"{location} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise SplitValidationError(f"{location} must not contain control characters")
    return value


def _positive_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SplitValidationError(f"{location} must be a positive integer")
    return value


def _require_sha256(value: object, location: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value) or value == _ZERO_SHA256:
        raise SplitValidationError(f"{location} must be a non-placeholder lowercase SHA-256 digest")
    return value


def _string_list(value: object, location: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise SplitValidationError(f"{location} must be a non-empty list")
    normalized = [_portable_text(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        raise SplitValidationError(f"{location} must be sorted and duplicate-free")
    return normalized


def _strata_count_mapping(value: object, location: str) -> Dict[str, int]:
    if not isinstance(value, MappingABC):
        raise SplitValidationError(f"{location} must be a mapping")
    normalized: Dict[str, int] = {}
    for raw_stratum, raw_count in value.items():
        stratum = _portable_text(raw_stratum, f"{location} key")
        normalized[stratum] = _positive_integer(raw_count, f"{location}[{stratum!r}]")
    if len(normalized) != len(value):
        raise SplitValidationError(f"{location} contains duplicate normalized keys")
    return normalized


def _normalize_family_id(value: object, location: str) -> str:
    raw = _portable_text(value, location)
    namespace, separator, identifier = raw.partition(":")
    normalized_namespace = namespace.lower()
    if not separator or not identifier or not _NAMESPACE_RE.fullmatch(normalized_namespace):
        raise SplitValidationError(f"{location} must use a namespace:value identifier")
    return f"{normalized_namespace}:{identifier}"


def _structured_family_ids(example: Mapping[str, Any], index: int) -> List[str]:
    structured: List[str] = []
    list_fields = {
        "entity_ids": "entity",
        "document_family_ids": "document_family",
        "provenance_ids": "provenance",
    }
    singular_fields = {
        "entity_id": "entity",
        "document_family_id": "document_family",
        "provenance_id": "provenance",
        "near_duplicate_group": "near_duplicate",
    }
    for field, namespace in list_fields.items():
        if field not in example:
            continue
        values = example[field]
        if not _is_sequence(values):
            raise SplitValidationError(f"examples[{index}].{field} must be an ordered sequence")
        normalized = [
            f"{namespace}:{_portable_text(value, f'examples[{index}].{field}[{item_index}]')}"
            for item_index, value in enumerate(values)
        ]
        if len(set(normalized)) != len(normalized):
            raise SplitValidationError(f"examples[{index}].{field} contains a duplicate")
        structured.extend(normalized)
    for field, namespace in singular_fields.items():
        value = example.get(field)
        if value is not None:
            structured.append(f"{namespace}:{_portable_text(value, f'examples[{index}].{field}')}")
    return structured


def _normalize_examples(
    examples: Sequence[Mapping[str, Any]],
    strata_key: Optional[str],
) -> List[Dict[str, Any]]:
    if not _is_sequence(examples) or not examples:
        raise SplitValidationError("examples must be a non-empty ordered sequence")
    if strata_key is not None:
        _portable_text(strata_key, "strata_key")

    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for index, example in enumerate(examples):
        if not isinstance(example, MappingABC):
            raise TypeError(f"examples[{index}] must be a mapping")
        if "example_id" not in example or "family_ids" not in example:
            raise SplitValidationError(f"examples[{index}] requires example_id and family_ids")
        example_id = _portable_text(example["example_id"], f"examples[{index}].example_id")
        if example_id in seen_ids:
            raise SplitValidationError(f"duplicate example_id: {example_id!r}")
        seen_ids.add(example_id)

        raw_families = example["family_ids"]
        if not _is_sequence(raw_families) or not raw_families:
            raise SplitValidationError(
                f"examples[{index}].family_ids must be a non-empty ordered sequence"
            )
        families = [
            _normalize_family_id(value, f"examples[{index}].family_ids[{family_index}]")
            for family_index, value in enumerate(raw_families)
        ]
        if len(set(families)) != len(families):
            raise SplitValidationError(
                f"examples[{index}].family_ids contains a duplicate after normalization"
            )
        families.extend(_structured_family_ids(example, index))

        record: Dict[str, Any] = {
            "example_id": example_id,
            "family_ids": sorted(set(families)),
        }
        if strata_key is not None:
            if strata_key not in example:
                raise SplitValidationError(
                    f"examples[{index}] is missing strata key {strata_key!r}"
                )
            record["stratum"] = _portable_text(
                example[strata_key], f"examples[{index}].{strata_key}"
            )
        normalized.append(record)
    return sorted(normalized, key=lambda item: item["example_id"])


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        self.parent[upper] = lower


def _build_components(examples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ids = [str(example["example_id"]) for example in examples]
    union_find = _UnionFind(ids)
    family_owner: Dict[str, str] = {}
    for example in examples:
        example_id = str(example["example_id"])
        for family_id in example["family_ids"]:
            owner = family_owner.setdefault(str(family_id), example_id)
            union_find.union(example_id, owner)

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[union_find.find(str(example["example_id"]))].append(example)

    components: List[Dict[str, Any]] = []
    for members in grouped.values():
        example_ids = sorted(str(member["example_id"]) for member in members)
        family_ids = sorted(
            {str(family_id) for member in members for family_id in member["family_ids"]}
        )
        strata_counts = Counter(str(member["stratum"]) for member in members if "stratum" in member)
        component_id = sha256_json({"example_ids": example_ids, "family_ids": family_ids})
        components.append(
            {
                "component_id": component_id,
                "example_ids": example_ids,
                "family_ids": family_ids,
                "example_count": len(example_ids),
                "strata_counts": dict(sorted(strata_counts.items())),
            }
        )
    return components


def _seeded_digest(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed),) + parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assignment_penalty(
    candidate: str,
    component: Mapping[str, Any],
    counts: Mapping[str, int],
    stratum_counts: Mapping[str, Mapping[str, int]],
    target_counts: Mapping[str, float],
    target_strata: Mapping[str, Mapping[str, float]],
) -> float:
    overall_penalty = 0.0
    for partition in PARTITIONS:
        observed = counts[partition]
        if partition == candidate:
            observed += int(component["example_count"])
        target = target_counts[partition]
        overall_penalty += ((observed - target) ** 2) / max(target, 1.0)

    if not target_strata:
        return overall_penalty
    stratum_penalty = 0.0
    for stratum, targets in target_strata.items():
        component_count = int(component["strata_counts"].get(stratum, 0))
        for partition in PARTITIONS:
            observed = stratum_counts[partition].get(stratum, 0)
            if partition == candidate:
                observed += component_count
            target = targets[partition]
            stratum_penalty += ((observed - target) ** 2) / max(target, 1.0)
    return overall_penalty + stratum_penalty / len(target_strata)


def _assign_components(
    components: Sequence[Mapping[str, Any]],
    ratios: Mapping[str, float],
    seed: int,
) -> List[Dict[str, Any]]:
    counts = {partition: 0 for partition in PARTITIONS}
    stratum_counts: Dict[str, Dict[str, int]] = {partition: {} for partition in PARTITIONS}
    total = sum(int(component["example_count"]) for component in components)
    target_counts = {partition: total * ratios[partition] for partition in PARTITIONS}
    total_strata: Counter[str] = Counter()
    for component in components:
        total_strata.update(component["strata_counts"])
    target_strata = {
        stratum: {partition: count * ratios[partition] for partition in PARTITIONS}
        for stratum, count in sorted(total_strata.items())
    }

    ordered = sorted(
        components,
        key=lambda component: (
            -int(component["example_count"]),
            _seeded_digest(seed, str(component["component_id"])),
        ),
    )
    assigned: List[Dict[str, Any]] = []
    for component in ordered:
        choices = []
        for partition in PARTITIONS:
            penalty = _assignment_penalty(
                partition,
                component,
                counts,
                stratum_counts,
                target_counts,
                target_strata,
            )
            tie_breaker = _seeded_digest(seed, str(component["component_id"]), partition)
            choices.append((penalty, tie_breaker, partition))
        partition = min(choices)[2]
        record = dict(component)
        record["partition"] = partition
        assigned.append(record)
        counts[partition] += int(component["example_count"])
        for stratum, count in component["strata_counts"].items():
            stratum_counts[partition][stratum] = stratum_counts[partition].get(stratum, 0) + int(
                count
            )
    return sorted(assigned, key=lambda component: str(component["component_id"]))


def _audit_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, MappingABC):
        raise SplitValidationError("split manifest must be a mapping")
    keys = set(payload)
    unsealed_shape = _TOP_LEVEL_FIELDS
    sealed_shape = _TOP_LEVEL_FIELDS | {"audit", "manifest_sha256"}
    if keys != unsealed_shape and keys != sealed_shape:
        raise SplitValidationError("manifest top-level fields have an invalid schema")
    if payload["schema"] != SPLIT_MANIFEST_SCHEMA:
        raise SplitValidationError("unsupported split manifest schema")
    if payload["algorithm"] != SPLIT_ALGORITHM:
        raise SplitValidationError("unsupported split algorithm")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise SplitValidationError("manifest seed must be an integer")
    ratios = payload["ratios"]
    if not isinstance(ratios, MappingABC):
        raise SplitValidationError("manifest ratios must be a mapping")
    normalized_ratios = _normalize_ratios(cast(Mapping[str, float], ratios))

    strata_key = payload["strata_key"]
    if strata_key is not None:
        strata_key = _portable_text(strata_key, "manifest strata_key")

    policy = payload["policy"]
    if not isinstance(policy, MappingABC) or set(policy) != set(_POLICY):
        raise SplitValidationError("manifest policy has an invalid schema")
    for key, expected in _POLICY.items():
        actual = policy[key]
        if type(actual) is not type(expected) or actual != expected:
            raise SplitValidationError(f"manifest policy field {key!r} is invalid")

    source = payload["source"]
    if not isinstance(source, MappingABC) or set(source) != _SOURCE_FIELDS:
        raise SplitValidationError("manifest source has an invalid schema")
    source_example_count = _positive_integer(
        source["example_count"], "manifest source.example_count"
    )
    source_component_count = _positive_integer(
        source["component_count"], "manifest source.component_count"
    )
    if source_component_count > source_example_count:
        raise SplitValidationError("source component_count cannot exceed example_count")
    for field in ("source_snapshot_sha256", "normalized_examples_sha256"):
        _require_sha256(source[field], f"manifest source.{field}")

    partitions = payload["partitions"]
    components = payload["components"]
    if not isinstance(partitions, MappingABC) or set(partitions) != set(PARTITIONS):
        raise SplitValidationError("manifest partitions have an invalid schema")
    if not isinstance(components, MappingABC) or not components:
        raise SplitValidationError("manifest components must be a non-empty mapping")

    assigned_occurrences: Counter[str] = Counter()
    component_occurrences: Counter[str] = Counter()
    component_example_occurrences: Counter[str] = Counter()
    component_family_occurrences: Counter[str] = Counter()
    component_stratum_occurrences: Counter[str] = Counter()
    partition_examples: Dict[str, set] = {}
    partition_families: Dict[str, set] = {partition: set() for partition in PARTITIONS}
    partition_strata: Dict[str, Dict[str, int]] = {}
    validated_components: Dict[str, Dict[str, Any]] = {}
    component_integrity = True

    for component_id, component in components.items():
        _require_sha256(component_id, "component ID")
        if not isinstance(component, MappingABC) or set(component) != _COMPONENT_FIELDS:
            raise SplitValidationError(f"component {component_id!r} has an invalid schema")
        example_ids = _string_list(
            component["example_ids"], f"component {component_id!r}.example_ids"
        )
        family_ids = _string_list(component["family_ids"], f"component {component_id!r}.family_ids")
        for family_index, family_id in enumerate(family_ids):
            if (
                _normalize_family_id(
                    family_id, f"component {component_id!r}.family_ids[{family_index}]"
                )
                != family_id
            ):
                raise SplitValidationError("component family namespaces must be normalized")
        example_count = _positive_integer(
            component["example_count"], f"component {component_id!r}.example_count"
        )
        if example_count != len(example_ids):
            raise SplitValidationError(f"component {component_id!r} example_count is inconsistent")
        partition = component["partition"]
        if not isinstance(partition, str) or partition not in PARTITIONS:
            raise SplitValidationError(f"component {component_id!r} partition is invalid")
        strata_counts = _strata_count_mapping(
            component["strata_counts"], f"component {component_id!r}.strata_counts"
        )
        if strata_key is None and strata_counts:
            raise SplitValidationError("strata_counts must be empty when strata_key is null")
        if strata_key is not None and (
            not strata_counts or sum(strata_counts.values()) != example_count
        ):
            raise SplitValidationError(
                f"component {component_id!r} strata_counts must cover every example"
            )
        expected_component_id = sha256_json({"example_ids": example_ids, "family_ids": family_ids})
        if component_id != expected_component_id:
            component_integrity = False
        component_example_occurrences.update(example_ids)
        component_family_occurrences.update(family_ids)
        component_stratum_occurrences.update(strata_counts.keys())
        validated_components[component_id] = {
            "example_ids": example_ids,
            "family_ids": family_ids,
            "example_count": example_count,
            "strata_counts": strata_counts,
            "partition": partition,
        }

    unknown_component_ids = set()
    for partition in PARTITIONS:
        record = partitions[partition]
        if not isinstance(record, MappingABC) or set(record) != _PARTITION_FIELDS:
            raise SplitValidationError(f"partition {partition} has an invalid schema")
        example_ids = _string_list(record["example_ids"], f"partition {partition}.example_ids")
        component_ids = _string_list(
            record["component_ids"], f"partition {partition}.component_ids"
        )
        for component_index, component_id in enumerate(component_ids):
            _require_sha256(
                component_id,
                f"partition {partition}.component_ids[{component_index}]",
            )
        example_count = _positive_integer(
            record["example_count"], f"partition {partition}.example_count"
        )
        component_count = _positive_integer(
            record["component_count"], f"partition {partition}.component_count"
        )
        if example_count != len(example_ids):
            raise SplitValidationError(f"partition {partition} example_count is inconsistent")
        if component_count != len(component_ids):
            raise SplitValidationError(f"partition {partition} component_count is inconsistent")
        strata_counts = _strata_count_mapping(
            record["strata_counts"], f"partition {partition}.strata_counts"
        )
        if strata_key is None and strata_counts:
            raise SplitValidationError("strata_counts must be empty when strata_key is null")
        if strata_key is not None and (
            not strata_counts or sum(strata_counts.values()) != example_count
        ):
            raise SplitValidationError(
                f"partition {partition} strata_counts must cover every example"
            )

        partition_examples[partition] = set(example_ids)
        partition_strata[partition] = strata_counts
        assigned_occurrences.update(example_ids)
        component_occurrences.update(component_ids)
        derived_ids = set()
        derived_strata: Counter[str] = Counter()
        for component_id in component_ids:
            component = validated_components.get(component_id)
            if component is None:
                unknown_component_ids.add(component_id)
                component_integrity = False
                continue
            if component["partition"] != partition:
                component_integrity = False
            derived_ids.update(component["example_ids"])
            partition_families[partition].update(component["family_ids"])
            derived_strata.update(component["strata_counts"])
        if derived_ids != partition_examples[partition]:
            component_integrity = False
        if dict(sorted(derived_strata.items())) != dict(sorted(strata_counts.items())):
            component_integrity = False

    expected_ids = {
        example_id
        for component in validated_components.values()
        for example_id in component["example_ids"]
    }
    duplicate_assignments = sorted(
        example_id for example_id, count in assigned_occurrences.items() if count != 1
    )
    duplicate_component_examples = sorted(
        example_id for example_id, count in component_example_occurrences.items() if count != 1
    )
    duplicate_component_families = sorted(
        family_id for family_id, count in component_family_occurrences.items() if count != 1
    )
    missing_or_duplicate_components = sorted(
        component_id
        for component_id in validated_components
        if component_occurrences[component_id] != 1
    )
    unassigned = sorted(expected_ids - set(assigned_occurrences))
    unknown = sorted(set(assigned_occurrences) - expected_ids)
    family_overlaps: Dict[str, List[str]] = {}
    example_overlaps: Dict[str, List[str]] = {}
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            key = f"{left}__{right}"
            family_overlaps[key] = sorted(partition_families[left] & partition_families[right])
            example_overlaps[key] = sorted(partition_examples[left] & partition_examples[right])
    family_overlap_count = sum(len(values) for values in family_overlaps.values())
    example_overlap_count = sum(len(values) for values in example_overlaps.values())
    nonempty_partitions = all(partition_examples[partition] for partition in PARTITIONS)

    feasible_strata = sorted(
        stratum
        for stratum, count in component_stratum_occurrences.items()
        if count >= len(PARTITIONS)
    )
    feasible_strata_missing_partitions = {
        stratum: [
            partition
            for partition in PARTITIONS
            if partition_strata[partition].get(stratum, 0) == 0
        ]
        for stratum in feasible_strata
        if any(partition_strata[partition].get(stratum, 0) == 0 for partition in PARTITIONS)
    }
    strata_presence_passed = not feasible_strata_missing_partitions
    source_counts_match = (
        len(expected_ids) == source_example_count
        and len(validated_components) == source_component_count
    )
    replay_components = [
        {"component_id": component_id, **dict(component)}
        for component_id, component in sorted(validated_components.items())
    ]
    replayed_assignments = {
        str(component["component_id"]): component["partition"]
        for component in _assign_components(
            replay_components,
            normalized_ratios,
            payload["seed"],
        )
    }
    recorded_assignments = {
        component_id: component["partition"]
        for component_id, component in validated_components.items()
    }
    assignment_matches_algorithm = replayed_assignments == recorded_assignments
    passed = (
        component_integrity
        and assignment_matches_algorithm
        and nonempty_partitions
        and strata_presence_passed
        and source_counts_match
        and not duplicate_assignments
        and not duplicate_component_examples
        and not duplicate_component_families
        and not missing_or_duplicate_components
        and not unknown_component_ids
        and not unassigned
        and not unknown
        and family_overlap_count == 0
        and example_overlap_count == 0
    )
    return {
        "passed": passed,
        "component_integrity": component_integrity,
        "assignment_matches_declared_algorithm_seed_and_ratios": assignment_matches_algorithm,
        "source_counts_match": source_counts_match,
        "all_partitions_nonempty": nonempty_partitions,
        "all_examples_assigned_once": not duplicate_assignments and not unassigned and not unknown,
        "strata_presence_postcondition_passed": strata_presence_passed,
        "feasible_strata": feasible_strata,
        "feasible_strata_missing_partitions": feasible_strata_missing_partitions,
        "cross_partition_family_overlap_count": family_overlap_count,
        "cross_partition_example_overlap_count": example_overlap_count,
        "duplicate_or_multiply_assigned_example_ids": duplicate_assignments,
        "duplicate_component_example_ids": duplicate_component_examples,
        "duplicate_component_family_ids": duplicate_component_families,
        "missing_or_duplicate_component_ids": missing_or_duplicate_components,
        "unknown_component_ids": sorted(unknown_component_ids),
        "unassigned_example_ids": unassigned,
        "unknown_assigned_example_ids": unknown,
        "family_overlaps": family_overlaps,
        "example_overlaps": example_overlaps,
    }


def build_group_disjoint_split(
    examples: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float],
    seed: int,
    strata_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and seal a deterministic transitive family-disjoint split."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    normalized_ratios = _normalize_ratios(ratios)
    normalized_examples = _normalize_examples(examples, strata_key)
    raw_by_id = {
        _portable_text(example["example_id"], "example.example_id"): dict(example)
        for example in examples
    }
    ordered_source_snapshot = [raw_by_id[example["example_id"]] for example in normalized_examples]
    try:
        source_snapshot_sha256 = sha256_json(ordered_source_snapshot)
    except (TypeError, ValueError) as exc:
        raise SplitValidationError(
            "source examples must be finite, canonical-JSON-compatible records"
        ) from exc
    components = _assign_components(_build_components(normalized_examples), normalized_ratios, seed)

    component_map = {
        str(component["component_id"]): {
            key: value for key, value in component.items() if key != "component_id"
        }
        for component in components
    }
    partition_records: Dict[str, Dict[str, Any]] = {}
    for partition in PARTITIONS:
        partition_components = [
            component for component in components if component["partition"] == partition
        ]
        example_ids = sorted(
            example_id
            for component in partition_components
            for example_id in component["example_ids"]
        )
        strata_counts: Counter[str] = Counter()
        for component in partition_components:
            strata_counts.update(component["strata_counts"])
        partition_records[partition] = {
            "example_ids": example_ids,
            "component_ids": sorted(
                str(component["component_id"]) for component in partition_components
            ),
            "example_count": len(example_ids),
            "component_count": len(partition_components),
            "strata_counts": dict(sorted(strata_counts.items())),
        }

    payload: Dict[str, Any] = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "ratios": {partition: normalized_ratios[partition] for partition in PARTITIONS},
        "strata_key": strata_key,
        "policy": dict(_POLICY),
        "source": {
            "example_count": len(normalized_examples),
            "component_count": len(components),
            "source_snapshot_sha256": source_snapshot_sha256,
            "normalized_examples_sha256": sha256_json(normalized_examples),
        },
        "partitions": partition_records,
        "components": component_map,
    }
    payload["audit"] = _audit_payload(payload)
    if not payload["audit"]["passed"]:
        if not payload["audit"]["strata_presence_postcondition_passed"]:
            raise SplitValidationError("split failed feasible-stratum representation postcondition")
        raise SplitValidationError("internal split postcondition audit failed")
    return seal_manifest(payload)


def audit_split_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify a split seal and recompute all leakage postconditions."""

    if not verify_manifest(manifest):
        raise SplitValidationError("split manifest seal is invalid")
    if manifest.get("schema") != SPLIT_MANIFEST_SCHEMA:
        raise SplitValidationError("unsupported split manifest schema")
    computed = _audit_payload(manifest)
    stored_audit = manifest.get("audit")
    if not isinstance(stored_audit, MappingABC) or sha256_json(stored_audit) != sha256_json(
        computed
    ):
        raise SplitValidationError("stored split audit does not match recomputed audit")
    if not computed["passed"]:
        raise SplitValidationError("split manifest fails leakage postconditions")
    return computed


def write_split_manifest(path: PathLike, manifest: Mapping[str, Any]) -> str:
    """Write a valid split manifest once, allowing only byte-identical repeats."""

    audit_split_manifest(manifest)
    return write_immutable_json(path, manifest)


def partition_example_ids(
    manifest: Mapping[str, Any],
    partition: str,
    *,
    purpose: str,
) -> List[str]:
    """Return IDs only when the declared purpose respects the final firewall."""

    audit_split_manifest(manifest)
    if partition not in PARTITIONS:
        raise SplitValidationError(f"unknown partition: {partition!r}")
    if purpose not in _PURPOSES:
        raise SplitValidationError(f"unknown access purpose: {purpose!r}")
    if partition == "sealed_final" and purpose != "final_evaluation":
        raise PermissionError("sealed_final is inaccessible outside final_evaluation")
    if purpose == "training" and partition != "train":
        raise PermissionError("training may access only the train partition")
    if purpose == "final_evaluation" and partition != "sealed_final":
        raise PermissionError("final_evaluation may access only sealed_final")
    return list(manifest["partitions"][partition]["example_ids"])


def stable_sample_ids(
    manifest: Mapping[str, Any],
    partition: str,
    *,
    sample_size: int,
    seed: int,
    purpose: str,
) -> List[str]:
    """Select a stable post-split sample using SHA-256 ordering."""

    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size < 0:
        raise SplitValidationError("sample_size must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    example_ids = partition_example_ids(manifest, partition, purpose=purpose)
    if sample_size > len(example_ids):
        raise SplitValidationError(
            f"sample_size {sample_size} exceeds partition size {len(example_ids)}"
        )
    ordered = sorted(
        example_ids,
        key=lambda example_id: (
            _seeded_digest(seed, partition, example_id),
            example_id,
        ),
    )
    return ordered[:sample_size]


__all__ = [
    "PARTITIONS",
    "SPLIT_MANIFEST_SCHEMA",
    "SplitValidationError",
    "audit_split_manifest",
    "build_group_disjoint_split",
    "partition_example_ids",
    "stable_sample_ids",
    "write_split_manifest",
]

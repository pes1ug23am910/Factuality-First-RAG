"""Tests for deterministic transitive family-disjoint splits."""

from __future__ import annotations

import copy

import pytest

from factuality_rag.data.splits import (
    SplitValidationError,
    audit_split_manifest,
    build_group_disjoint_split,
    partition_example_ids,
    stable_sample_ids,
    write_split_manifest,
)
from factuality_rag.reproducibility import seal_manifest


RATIOS = {"train": 0.5, "tuning": 0.25, "sealed_final": 0.25}
ZERO_SHA256 = "0" * 64


def _examples():
    values = []
    for index in range(12):
        values.append(
            {
                "example_id": f"fixture:q{index:02d}",
                "family_ids": [f"example:q{index:02d}"],
                "bucket": "head" if index % 2 == 0 else "tail",
            }
        )
    return values


def test_split_is_independent_of_input_order_and_sealed() -> None:
    forward = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=17, strata_key="bucket")
    reverse = build_group_disjoint_split(
        list(reversed(_examples())), ratios=RATIOS, seed=17, strata_key="bucket"
    )
    assert forward == reverse
    assert audit_split_manifest(forward)["passed"] is True
    assert forward["manifest_sha256"] != ZERO_SHA256
    assert forward["source"]["source_snapshot_sha256"] != ZERO_SHA256
    assert forward["source"]["normalized_examples_sha256"] != ZERO_SHA256
    assert all(component_id != ZERO_SHA256 for component_id in forward["components"])


def test_transitive_family_links_stay_in_one_component() -> None:
    examples = _examples()
    examples[0]["family_ids"] = ["entity:a", "wiki:shared-1"]
    examples[1]["family_ids"] = ["wiki:shared-1", "entity:b"]
    examples[2]["family_ids"] = ["entity:b", "provenance:page-9"]
    manifest = build_group_disjoint_split(examples, ratios=RATIOS, seed=9)
    partitions = {
        example_id: partition
        for partition, record in manifest["partitions"].items()
        for example_id in record["example_ids"]
    }
    assert {partitions[f"fixture:q{index:02d}"] for index in range(3)} == {
        partitions["fixture:q00"]
    }
    component = next(
        record
        for record in manifest["components"].values()
        if "fixture:q00" in record["example_ids"]
    )
    assert component["example_ids"] == ["fixture:q00", "fixture:q01", "fixture:q02"]
    assert manifest["audit"]["cross_partition_family_overlap_count"] == 0


def test_structured_entity_ids_are_automatically_grouped() -> None:
    examples = _examples()
    examples[0]["entity_id"] = "shared"
    examples[1]["entity_id"] = "shared"
    manifest = build_group_disjoint_split(examples, ratios=RATIOS, seed=9)
    partitions = {
        example_id: partition
        for partition, record in manifest["partitions"].items()
        for example_id in record["example_ids"]
    }
    assert partitions["fixture:q00"] == partitions["fixture:q01"]


def test_source_content_changes_invalidate_the_split_fingerprint() -> None:
    examples = _examples()
    examples[0]["question"] = "original"
    original = build_group_disjoint_split(examples, ratios=RATIOS, seed=4)
    examples[0]["question"] = "changed"
    changed = build_group_disjoint_split(examples, ratios=RATIOS, seed=4)
    assert (
        original["source"]["normalized_examples_sha256"]
        == changed["source"]["normalized_examples_sha256"]
    )
    assert (
        original["source"]["source_snapshot_sha256"] != changed["source"]["source_snapshot_sha256"]
    )
    assert original["manifest_sha256"] != changed["manifest_sha256"]


def test_unusable_split_with_empty_partitions_fails_closed() -> None:
    with pytest.raises(SplitValidationError, match="non-empty list"):
        build_group_disjoint_split([_examples()[0]], ratios=RATIOS, seed=1)


def test_strata_and_counts_are_preserved() -> None:
    manifest = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=21, strata_key="bucket")
    assert sum(record["example_count"] for record in manifest["partitions"].values()) == 12
    assert (
        sum(record["strata_counts"].get("head", 0) for record in manifest["partitions"].values())
        == 6
    )
    assert (
        sum(record["strata_counts"].get("tail", 0) for record in manifest["partitions"].values())
        == 6
    )
    assert all(record["example_count"] > 0 for record in manifest["partitions"].values())


def test_feasible_strata_are_represented_or_the_build_fails_explicitly() -> None:
    ratios = {"train": 0.8, "tuning": 0.1, "sealed_final": 0.1}
    examples = [
        {
            "example_id": f"fixture:rare-{index:02d}",
            "family_ids": [f"example:rare-{index:02d}"],
            "bucket": "rare",
        }
        for index in range(3)
    ]
    examples.extend(
        {
            "example_id": f"fixture:common-{index:02d}",
            "family_ids": [f"example:common-{index:02d}"],
            "bucket": "common",
        }
        for index in range(27)
    )

    failed_seeds = []
    for seed in range(20):
        try:
            manifest = build_group_disjoint_split(
                examples, ratios=ratios, seed=seed, strata_key="bucket"
            )
        except SplitValidationError as exc:
            assert "feasible-stratum representation" in str(exc)
            failed_seeds.append(seed)
            continue
        assert all(
            manifest["partitions"][partition]["strata_counts"].get("rare", 0) > 0
            for partition in ("train", "tuning", "sealed_final")
        )
        assert manifest["audit"]["strata_presence_postcondition_passed"] is True
    assert failed_seeds, "fixture must exercise the explicit feasible-stratum failure path"


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            "duplicate example_id",
        ),
        (lambda rows: rows[0].pop("example_id"), "requires example_id"),
        (lambda rows: rows[0].__setitem__("family_ids", []), "non-empty ordered"),
        (
            lambda rows: rows[0].__setitem__("family_ids", ["not-namespaced"]),
            "namespace:value",
        ),
        (
            lambda rows: rows[0].__setitem__("family_ids", ["Entity:a", "entity:a"]),
            "duplicate after normalization",
        ),
        (lambda rows: rows[0].pop("bucket"), "missing strata key"),
    ],
)
def test_invalid_example_schema_fails_closed(mutation, message: str) -> None:
    examples = _examples()
    mutation(examples)
    with pytest.raises(SplitValidationError, match=message):
        build_group_disjoint_split(examples, ratios=RATIOS, seed=1, strata_key="bucket")


@pytest.mark.parametrize(
    "ratios, error_type, message",
    [
        ({"train": 0.8, "tuning": 0.2}, SplitValidationError, "exactly"),
        (
            {"train": 0.8, "tuning": 0.1, "sealed_final": 0.2},
            SplitValidationError,
            "sum to 1.0",
        ),
        (
            {"train": 1.0, "tuning": 0.0, "sealed_final": 0.0},
            SplitValidationError,
            "finite positive",
        ),
        (
            {"train": True, "tuning": 0.1, "sealed_final": 0.1},
            TypeError,
            "finite positive",
        ),
    ],
)
def test_invalid_ratios_fail_closed(ratios, error_type, message: str) -> None:
    with pytest.raises(error_type, match=message):
        build_group_disjoint_split(_examples(), ratios=ratios, seed=1)


def _reseal_with_mutation(manifest, mutation):
    plain = copy.deepcopy(manifest)
    plain.pop("manifest_sha256")
    mutation(plain)
    return seal_manifest(plain)


def _mutate_first_component(payload, values) -> None:
    next(iter(payload["components"].values())).update(values)


@pytest.mark.parametrize(
    "mutation,error_type,message",
    [
        (
            lambda payload: payload.update({"unexpected": "field"}),
            SplitValidationError,
            "top-level fields",
        ),
        (
            lambda payload: payload.update({"schema": "factuality-rag.split-manifest.v0"}),
            SplitValidationError,
            "schema",
        ),
        (
            lambda payload: payload.update({"algorithm": "unreviewed-algorithm"}),
            SplitValidationError,
            "algorithm",
        ),
        (
            lambda payload: payload.update({"seed": True}),
            SplitValidationError,
            "seed must be an integer",
        ),
        (
            lambda payload: payload["ratios"].pop("sealed_final"),
            SplitValidationError,
            "exactly",
        ),
        (
            lambda payload: payload["ratios"].update({"train": True}),
            TypeError,
            "finite positive",
        ),
        (
            lambda payload: payload["ratios"].update(
                {"train": 0.6, "tuning": 0.3, "sealed_final": 0.2}
            ),
            SplitValidationError,
            "sum to 1.0",
        ),
        (
            lambda payload: payload["policy"].update({"transitive_family_closure": 1}),
            SplitValidationError,
            "policy field",
        ),
        (
            lambda payload: payload["source"].update({"unexpected": 1}),
            SplitValidationError,
            "source has an invalid schema",
        ),
        (
            lambda payload: payload["source"].update(
                {"source_snapshot_sha256": payload["source"]["source_snapshot_sha256"].upper()}
            ),
            SplitValidationError,
            "lowercase SHA-256",
        ),
        (
            lambda payload: payload["source"].update({"source_snapshot_sha256": ZERO_SHA256}),
            SplitValidationError,
            "non-placeholder lowercase SHA-256",
        ),
        (
            lambda payload: payload["source"].update({"normalized_examples_sha256": ZERO_SHA256}),
            SplitValidationError,
            "non-placeholder lowercase SHA-256",
        ),
        (
            lambda payload: payload["components"].update(
                {ZERO_SHA256: payload["components"].pop(next(iter(payload["components"])))}
            ),
            SplitValidationError,
            "component ID must be a non-placeholder lowercase SHA-256",
        ),
        (
            lambda payload: next(iter(payload["partitions"].values()))["component_ids"].__setitem__(
                0, ZERO_SHA256
            ),
            SplitValidationError,
            "component_ids.*non-placeholder lowercase SHA-256",
        ),
        (
            lambda payload: payload["source"].update({"example_count": True}),
            SplitValidationError,
            "positive integer",
        ),
        (
            lambda payload: payload["partitions"]["train"].update({"unexpected": 1}),
            SplitValidationError,
            "partition train has an invalid schema",
        ),
        (
            lambda payload: payload["partitions"]["train"].update({"component_count": True}),
            SplitValidationError,
            "positive integer",
        ),
        (
            lambda payload: payload["partitions"]["train"].update(
                {"component_count": payload["partitions"]["train"]["component_count"] + 1}
            ),
            SplitValidationError,
            "component_count is inconsistent",
        ),
        (
            lambda payload: _mutate_first_component(payload, {"unexpected": 1}),
            SplitValidationError,
            "component .* invalid schema",
        ),
        (
            lambda payload: _mutate_first_component(payload, {"example_count": True}),
            SplitValidationError,
            "positive integer",
        ),
        (
            lambda payload: payload.update({"strata_key": None}),
            SplitValidationError,
            "strata_counts must be empty",
        ),
        (
            lambda payload: payload["audit"].update({"passed": 1}),
            SplitValidationError,
            "stored split audit",
        ),
    ],
)
def test_resealed_manifest_with_malformed_metadata_is_rejected(
    mutation, error_type, message: str
) -> None:
    manifest = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=5, strata_key="bucket")
    malformed = _reseal_with_mutation(manifest, mutation)
    with pytest.raises(error_type, match=message):
        audit_split_manifest(malformed)


def test_resealed_manifest_assignment_is_bound_to_declared_ratios() -> None:
    manifest = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=5, strata_key="bucket")
    mutated = _reseal_with_mutation(
        manifest,
        lambda payload: payload.update(
            {"ratios": {"train": 0.6, "tuning": 0.2, "sealed_final": 0.2}}
        ),
    )

    with pytest.raises(SplitValidationError):
        audit_split_manifest(mutated)


def test_manifest_tamper_and_conflicting_overwrite_are_rejected(tmp_path) -> None:
    manifest = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=5)
    destination = tmp_path / "split.json"
    assert write_split_manifest(destination, manifest) == "created"
    assert write_split_manifest(destination, manifest) == "unchanged"

    tampered = copy.deepcopy(manifest)
    tampered["seed"] = 6
    with pytest.raises(SplitValidationError, match="seal is invalid"):
        write_split_manifest(tmp_path / "tampered.json", tampered)

    zero_seal = copy.deepcopy(manifest)
    zero_seal["manifest_sha256"] = ZERO_SHA256
    with pytest.raises(SplitValidationError, match="seal is invalid"):
        audit_split_manifest(zero_seal)

    plain = copy.deepcopy(manifest)
    plain.pop("manifest_sha256")
    plain["partitions"]["train"]["example_ids"].append("fixture:unknown")
    resealed = seal_manifest(plain)
    with pytest.raises(SplitValidationError, match="example_count is inconsistent"):
        write_split_manifest(tmp_path / "resealed.json", resealed)

    duplicated_component = copy.deepcopy(manifest)
    duplicated_component.pop("manifest_sha256")
    component_ids = list(duplicated_component["components"])
    first = duplicated_component["components"][component_ids[0]]
    second = duplicated_component["components"][component_ids[1]]
    second["example_ids"] = list(first["example_ids"])
    second["example_count"] = first["example_count"]
    duplicated_component = seal_manifest(duplicated_component)
    with pytest.raises(SplitValidationError, match="stored split audit"):
        write_split_manifest(tmp_path / "duplicated-component.json", duplicated_component)

    other = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=6)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_split_manifest(destination, other)


def test_sealed_final_access_guard_and_stable_post_split_sampling() -> None:
    manifest = build_group_disjoint_split(_examples(), ratios=RATIOS, seed=13)
    with pytest.raises(PermissionError, match="sealed_final"):
        partition_example_ids(manifest, "sealed_final", purpose="tuning")
    with pytest.raises(PermissionError, match="sealed_final"):
        partition_example_ids(manifest, "sealed_final", purpose="training")
    final_ids = partition_example_ids(manifest, "sealed_final", purpose="final_evaluation")
    assert final_ids

    tuning_ids = partition_example_ids(manifest, "tuning", purpose="tuning")
    sample_size = min(2, len(tuning_ids))
    first = stable_sample_ids(
        manifest,
        "tuning",
        sample_size=sample_size,
        seed=99,
        purpose="tuning",
    )
    second = stable_sample_ids(
        manifest,
        "tuning",
        sample_size=sample_size,
        seed=99,
        purpose="tuning",
    )
    assert first == second
    assert set(first).issubset(tuning_ids)
    with pytest.raises(SplitValidationError, match="exceeds partition size"):
        stable_sample_ids(
            manifest,
            "tuning",
            sample_size=len(tuning_ids) + 1,
            seed=99,
            purpose="tuning",
        )

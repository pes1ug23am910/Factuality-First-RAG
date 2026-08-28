"""Package-distributed experiment configs and evaluator fixtures.

Resource names are logical identities, not working-directory paths. Relative
corpus, index, and output paths inside a config continue to resolve in the
caller's workspace; package resources are never used as write destinations.
"""

from __future__ import annotations

from importlib.abc import Traversable
from importlib.resources import files

DEFAULT_PIPELINE_CONFIG = "exp_sample.yaml"
DEFAULT_EXPERIMENT_CONFIG = "exp_full_pipeline.yaml"
EVALUATOR_SANITY_FIXTURE = "evaluator_sanity_v1.json"

EXPERIMENT_CONFIG_NAMES = (
    "exp_2wiki.yaml",
    "exp_b1_closed_book.yaml",
    "exp_b2_always_rag.yaml",
    "exp_b3_gate_only.yaml",
    "exp_b4_score_only.yaml",
    "exp_b5_learned_scorer.yaml",
    "exp_fever.yaml",
    "exp_full_pipeline.yaml",
    "exp_hagrid.yaml",
    "exp_popqa.yaml",
    "exp_sample.yaml",
)


def _portable_resource_name(name: str, *, suffix: str) -> str:
    """Validate one flat, portable resource name before traversal."""

    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or not name.endswith(suffix)
    ):
        raise ValueError(f"invalid packaged resource name: {name!r}")
    return name


def experiment_config_resource(name: str = DEFAULT_PIPELINE_CONFIG) -> Traversable:
    """Return an existing packaged experiment YAML resource."""

    resource_name = _portable_resource_name(name, suffix=".yaml")
    if resource_name not in EXPERIMENT_CONFIG_NAMES:
        raise FileNotFoundError(f"unknown packaged experiment config: {resource_name}")
    resource = files(__package__).joinpath("configs").joinpath(resource_name)
    if not resource.is_file():
        raise FileNotFoundError(f"packaged experiment config is unavailable: {resource_name}")
    return resource


def read_experiment_config_bytes(name: str = DEFAULT_PIPELINE_CONFIG) -> bytes:
    """Read exact packaged YAML bytes without consulting the current directory."""

    return experiment_config_resource(name).read_bytes()


def experiment_config_identity(name: str = DEFAULT_PIPELINE_CONFIG) -> str:
    """Return a stable logical identity suitable for run metadata."""

    resource_name = _portable_resource_name(name, suffix=".yaml")
    if resource_name not in EXPERIMENT_CONFIG_NAMES:
        raise FileNotFoundError(f"unknown packaged experiment config: {resource_name}")
    return f"package:factuality_rag.resources/configs/{resource_name}"


def evaluator_sanity_resource() -> Traversable:
    """Return the canonical packaged evaluator-sanity fixture."""

    resource = files(__package__).joinpath("data").joinpath(EVALUATOR_SANITY_FIXTURE)
    if not resource.is_file():
        raise FileNotFoundError("packaged evaluator sanity fixture is unavailable")
    return resource


def read_evaluator_sanity_bytes() -> bytes:
    """Read the exact canonical evaluator-sanity fixture bytes."""

    return evaluator_sanity_resource().read_bytes()


__all__ = [
    "DEFAULT_EXPERIMENT_CONFIG",
    "DEFAULT_PIPELINE_CONFIG",
    "EVALUATOR_SANITY_FIXTURE",
    "EXPERIMENT_CONFIG_NAMES",
    "evaluator_sanity_resource",
    "experiment_config_identity",
    "experiment_config_resource",
    "read_evaluator_sanity_bytes",
    "read_experiment_config_bytes",
]

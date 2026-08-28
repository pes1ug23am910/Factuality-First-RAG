"""Fail-closed tests for research paths that lack frozen evidence inputs."""

from __future__ import annotations

import pytest

from scripts import train_scorer, tune_scorer_weights


def test_train_scorer_refuses_unbound_feature_generation() -> None:
    with pytest.raises(RuntimeError, match="independently labeled feature artifact"):
        train_scorer.main()


def test_weight_tuner_refuses_unsealed_objective() -> None:
    with pytest.raises(RuntimeError, match="independent revision-bound judgments"):
        tune_scorer_weights.main()


def test_weight_grid_helper_is_deterministic_and_validated() -> None:
    grid = tune_scorer_weights.generate_weight_grid(0.5)

    assert grid == [
        (0.0, 0.0, 1.0),
        (0.0, 0.5, 0.5),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (1.0, 0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="finite and in"):
        tune_scorer_weights.generate_weight_grid(0.0)

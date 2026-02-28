"""
tests.test_gating
~~~~~~~~~~~~~~~~~~
Unit tests for GatingProbe in mock-mode.

Runs deterministic mock probes and asserts correct behaviour
without loading any HuggingFace models.
"""

from __future__ import annotations

import pytest

from factuality_rag.gating.probe import GatingProbe


class TestGatingProbeMock:
    """Test suite for mock-mode GatingProbe."""

    @pytest.fixture()
    def probe(self) -> GatingProbe:
        """Create a mock GatingProbe."""
        return GatingProbe(
            generator_model_hf="mistral-7b-instruct",
            mock_mode=True,
            temp=1.0,
        )

    def test_should_retrieve_returns_bool(self, probe: GatingProbe) -> None:
        """should_retrieve must return a boolean."""
        result = probe.should_retrieve("What is the capital of France?")
        assert isinstance(result, bool)

    def test_deterministic_output(self, probe: GatingProbe) -> None:
        """Same prompt should yield the same decision."""
        r1 = probe.should_retrieve("test prompt", entropy_thresh=1.2)
        r2 = probe.should_retrieve("test prompt", entropy_thresh=1.2)
        assert r1 == r2

    def test_different_prompts_can_differ(self, probe: GatingProbe) -> None:
        """Different prompts may (but aren't required to) yield different results.
        This test just ensures no errors are raised on different inputs.
        """
        probe.should_retrieve("short")
        probe.should_retrieve("A much longer prompt about quantum mechanics and physics")

    def test_calibrate_temperature_mock(self, probe: GatingProbe) -> None:
        """Mock calibration should return a reasonable temperature."""
        temp = probe.calibrate_temperature(
            dev_prompts=["hello", "world", "test"]
        )
        assert isinstance(temp, float)
        assert 0.1 <= temp <= 5.0

    def test_entropy_computation(self, probe: GatingProbe) -> None:
        """Entropy should be non-negative."""
        import numpy as np

        logits = np.array([1.0, 2.0, 3.0, 0.5])
        entropy = probe._compute_entropy(logits)
        assert entropy >= 0.0

    def test_logit_gap_computation(self, probe: GatingProbe) -> None:
        """Logit gap should be correct for known values."""
        import numpy as np

        logits = np.array([5.0, 2.0, 1.0, 3.0])
        gap = probe._compute_logit_gap(logits)
        assert gap == pytest.approx(2.0, abs=1e-6)

    def test_high_entropy_triggers_retrieval(self, probe: GatingProbe) -> None:
        """With a very low entropy threshold, retrieval should be triggered."""
        result = probe.should_retrieve(
            "any prompt",
            entropy_thresh=0.001,  # very low → almost any entropy exceeds it
            logit_gap_thresh=0.0,
        )
        assert result is True

    def test_low_entropy_skips_retrieval(self, probe: GatingProbe) -> None:
        """With very high thresholds, retrieval should be skipped
        (entropy below threshold and logit gap above threshold).
        """
        result = probe.should_retrieve(
            "any prompt",
            entropy_thresh=999.0,  # very high → entropy is below
            logit_gap_thresh=0.0,  # very low → logit gap is above
        )
        assert result is False

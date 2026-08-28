"""
tests.test_gating
~~~~~~~~~~~~~~~~~~
Unit tests for GatingProbe in mock-mode.

Runs deterministic mock probes and asserts correct behaviour
without loading any HuggingFace models.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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

    @pytest.mark.parametrize("targets", [None, ["target"]])
    def test_calibrate_temperature_fails_closed_without_supervised_protocol(
        self,
        probe: GatingProbe,
        targets: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The removed self-labelled calibration must not probe or mutate state."""

        def fail_if_probed(*args: object, **kwargs: object) -> None:
            raise AssertionError("fail-closed calibration reached model probing")

        original_temperature = probe.temp
        monkeypatch.setattr(probe, "_get_next_token_logits", fail_if_probed)

        with pytest.raises(
            NotImplementedError,
            match="explicit provenance-bound targets",
        ):
            probe.calibrate_temperature(
                dev_prompts=["hello"],
                targets=targets,  # type: ignore[arg-type]
            )

        assert probe.temp == original_temperature

    def test_entropy_computation(self, probe: GatingProbe) -> None:
        """Entropy should be non-negative."""
        import numpy as np

        logits = np.array([1.0, 2.0, 3.0, 0.5])
        entropy = probe._compute_entropy(logits)
        assert entropy >= 0.0

    def test_entropy_is_finite_for_extreme_float16_logits(self, probe: GatingProbe) -> None:
        """FP16 softmax underflow must not disable the entropy signal."""
        import numpy as np

        logits = np.full(32000, -20.0, dtype=np.float16)
        logits[0] = 0.0
        stable_logits = logits.astype(np.float64)
        stable_logits -= stable_logits.max()
        weights = np.exp(stable_logits)
        probabilities = weights / weights.sum()
        expected = float(-np.sum(probabilities * np.log(probabilities)))
        with np.errstate(all="raise"):
            entropy = probe._compute_entropy(logits)
        assert np.isfinite(entropy)
        assert entropy == pytest.approx(expected, rel=1e-4, abs=1e-8)
        assert 0.0 <= entropy <= np.log(logits.size)

    def test_fp16_entropy_branch_can_trigger_retrieval(
        self, probe: GatingProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrieval can be triggered by entropy while the gap condition is false."""
        import numpy as np

        logits = np.full(32000, -2.0, dtype=np.float16)
        logits[0] = 0.0
        entropy = probe._compute_entropy(logits)
        gap = probe._compute_logit_gap(logits)
        assert entropy > 1.2
        assert gap >= 2.0
        monkeypatch.setattr(probe, "_get_next_token_logits", lambda *_: logits)
        assert (
            probe.should_retrieve("entropy-only", entropy_thresh=1.2, logit_gap_thresh=2.0) is True
        )

    def test_nonfinite_signal_raises_by_default(
        self, probe: GatingProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline experiments must stop instead of silently skipping retrieval."""
        import numpy as np

        logits = np.array([0.0, np.nan], dtype=np.float32)
        monkeypatch.setattr(probe, "_get_next_token_logits", lambda *_: logits)
        with pytest.raises(FloatingPointError, match="Non-finite gating signal"):
            probe.should_retrieve("invalid")

    def test_nonfinite_signal_can_force_serving_retrieval(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Serving may opt into a logged conservative retrieval fallback."""
        import logging
        import numpy as np

        probe = GatingProbe("mock", mock_mode=True, nonfinite_policy="retrieve")
        logits = np.array([0.0, np.nan], dtype=np.float32)
        monkeypatch.setattr(probe, "_get_next_token_logits", lambda *_: logits)
        with caplog.at_level(logging.ERROR):
            assert probe.should_retrieve("invalid") is True
        assert "forcing retrieval" in caplog.text

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


class TestGatingProbeInjectedComponents:
    """Regression tests for partial injection and real-model device placement."""

    def test_model_only_injection_loads_only_tokenizer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import factuality_rag.model_registry as registry

        provided_model = object()
        loaded_tokenizer = object()

        def fail_if_model_loaded(*args: object, **kwargs: object) -> object:
            raise AssertionError("pre-supplied model was replaced")

        def load_tokenizer(model_id: str) -> object:
            assert model_id == "model-id"
            return loaded_tokenizer

        monkeypatch.setattr(registry, "get_model", fail_if_model_loaded)
        monkeypatch.setattr(registry, "get_tokenizer", load_tokenizer)
        probe = GatingProbe("model-id", model=provided_model)

        probe._load_model()

        assert probe._model is provided_model
        assert probe._tokenizer is loaded_tokenizer

    def test_tokenizer_only_injection_loads_only_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import factuality_rag.model_registry as registry

        loaded_model = object()
        provided_tokenizer = object()

        def load_model(model_id: str, *, device: str) -> object:
            assert model_id == "model-id"
            assert device == "cuda:1"
            return loaded_model

        def fail_if_tokenizer_loaded(*args: object, **kwargs: object) -> object:
            raise AssertionError("pre-supplied tokenizer was replaced")

        monkeypatch.setattr(registry, "get_model", load_model)
        monkeypatch.setattr(registry, "get_tokenizer", fail_if_tokenizer_loaded)
        probe = GatingProbe("model-id", device="cuda:1", tokenizer=provided_tokenizer)

        probe._load_model()

        assert probe._model is loaded_model
        assert probe._tokenizer is provided_tokenizer

    def test_inputs_follow_loaded_model_embedding_device(self) -> None:
        import numpy as np
        import torch

        actual_input_device = torch.device("cuda:1")
        moves: list[Any] = []

        class FakeBatch(dict[str, Any]):
            def to(self, device: Any) -> "FakeBatch":
                moves.append(device)
                return self

        class FakeTokenizer:
            def __call__(self, prompt: str, *, return_tensors: str) -> FakeBatch:
                assert prompt == "prompt"
                assert return_tensors == "pt"
                return FakeBatch({"input_ids": torch.tensor([[1]])})

        class FakeModel:
            device = torch.device("cpu")

            def get_input_embeddings(self) -> Any:
                return SimpleNamespace(weight=SimpleNamespace(device=actual_input_device))

            def __call__(self, **tokens: Any) -> Any:
                assert "input_ids" in tokens
                return SimpleNamespace(logits=torch.tensor([[[1.0, 2.0, 3.0]]]))

        probe = GatingProbe(
            "model-id",
            device="cpu",
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
        )

        logits = probe._get_next_token_logits("prompt")

        assert moves == [actual_input_device]
        assert np.array_equal(logits, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

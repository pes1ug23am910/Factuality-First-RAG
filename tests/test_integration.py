"""
tests.test_integration
~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for GPU-dependent features.

Marked with ``@pytest.mark.integration`` so they can be skipped
in CI (``pytest -m "not integration"``).
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestRealGenerator:
    """Integration test: real Mistral-7B generation."""

    def test_real_generator_produces_text(self) -> None:
        from factuality_rag.generator.wrapper import Generator

        gen = Generator(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            mock_mode=False,
        )
        answer = gen.generate("What is the capital of France?", context="Paris is the capital.")
        assert isinstance(answer, str)
        assert len(answer) > 0


@pytest.mark.integration
class TestRealGating:
    """Integration test: real logit probing on GPU."""

    def test_real_gating_decision(self) -> None:
        from factuality_rag.gating.probe import GatingProbe

        probe = GatingProbe(
            generator_model_hf="mistralai/Mistral-7B-Instruct-v0.3",
            mock_mode=False,
            device="cuda",
        )
        result = probe.should_retrieve("What is the capital of France?")
        assert isinstance(result, bool)

    def test_real_multi_token_probe(self) -> None:
        from factuality_rag.gating.probe import GatingProbe

        probe = GatingProbe(
            generator_model_hf="mistralai/Mistral-7B-Instruct-v0.3",
            mock_mode=False,
            device="cuda",
        )
        result = probe.should_retrieve(
            "What is DNA?",
            probe_tokens=3,
        )
        assert isinstance(result, bool)


@pytest.mark.integration
class TestRealNLI:
    """Integration test: real NLI scoring."""

    def test_real_nli_entailment(self) -> None:
        from factuality_rag.scorer.passage import PassageScorer

        scorer = PassageScorer(mock_mode=False)
        score = scorer._nli_entailment(
            premise="Paris is the capital of France.",
            hypothesis="The capital of France is Paris.",
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should clearly be entailment

    def test_real_sentence_level_nli(self) -> None:
        from factuality_rag.scorer.passage import PassageScorer

        scorer = PassageScorer(mock_mode=False, nli_mode="sentence")
        score = scorer._sentence_level_nli(
            query="What is the capital of France?",
            passage_text="France is a country in Europe. Paris is the capital of France. It has many landmarks.",
        )
        assert 0.0 <= score <= 1.0


@pytest.mark.integration
class TestRealPipeline:
    """Integration test: full pipeline on GPU."""

    def test_real_pipeline_end_to_end(self) -> None:
        from factuality_rag.pipeline.orchestrator import Pipeline

        pipe = Pipeline(mock_mode=False)
        answer, trusted, provenance, tag = pipe.run(
            "What is the capital of France?",
            gate=False,  # skip gating to avoid model load issues
        )
        assert isinstance(answer, str)
        assert tag in ("high", "medium", "low")


@pytest.mark.integration
class TestRealCalibration:
    """Integration test: temperature calibration with real logits."""

    def test_calibrate_on_dev_prompts(self) -> None:
        from factuality_rag.gating.probe import GatingProbe

        probe = GatingProbe(
            generator_model_hf="mistralai/Mistral-7B-Instruct-v0.3",
            mock_mode=False,
            device="cuda",
        )
        temp = probe.calibrate_temperature(
            dev_prompts=[
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What is photosynthesis?",
            ]
        )
        assert 0.1 <= temp <= 5.0

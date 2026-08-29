"""
tests.test_propensity
=======================
Shape, boundary-condition, and correctness tests for:
    * cognitive.hr_mcp_fusion (HRMCPFusion)
    * cognitive.propensity    (classify_propensity, PropensityClassifier)
    * agents.schemas          (Pydantic V2 validation behavior)
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from agents.schemas import EvidenceItem, ExplainedPrediction, VerdictLabel, VerifiableFact
from cognitive.hr_mcp_fusion import HRMCPFusion, HRMCPOutput
from cognitive.propensity import (
    CompositeMetrics,
    ImpactLevel,
    PropensityClassifier,
    TrustLevel,
    classify_propensity,
)


# ---------------------------------------------------------------------------
# cognitive.hr_mcp_fusion
# ---------------------------------------------------------------------------

class TestHRMCPFusion:
    def test_forward_output_shapes(self) -> None:
        model = HRMCPFusion(visual_embed_dim=512, text_embed_dim=512)
        model.eval()
        visual = torch.randn(4, 512)
        text = torch.randn(4, 512)
        with torch.no_grad():
            output = model(visual, text)

        assert isinstance(output, HRMCPOutput)
        assert output.enhanced_joint_embedding.shape == (4, 1024)
        assert output.aigc_likelihood.shape == (4, 1)
        assert output.belief.shape == (4, 1)
        assert output.dissemination_propensity.shape == (4, 1)

    def test_outputs_are_bounded_in_unit_interval(self) -> None:
        model = HRMCPFusion()
        model.eval()
        visual = torch.randn(8, 512) * 5.0  # exaggerate magnitude
        text = torch.randn(8, 512) * 5.0
        with torch.no_grad():
            output = model(visual, text)

        for tensor in (output.aigc_likelihood, output.belief, output.dissemination_propensity):
            assert torch.all(tensor >= 0.0)
            assert torch.all(tensor <= 1.0)

    def test_mismatched_visual_embed_dim_raises(self) -> None:
        model = HRMCPFusion(visual_embed_dim=512, text_embed_dim=512)
        visual = torch.randn(2, 256)  # wrong dim
        text = torch.randn(2, 512)
        with pytest.raises(ValueError):
            model(visual, text)

    def test_mismatched_text_embed_dim_raises(self) -> None:
        model = HRMCPFusion(visual_embed_dim=512, text_embed_dim=512)
        visual = torch.randn(2, 512)
        text = torch.randn(2, 128)  # wrong dim
        with pytest.raises(ValueError):
            model(visual, text)

    def test_supports_non_default_embed_dims(self) -> None:
        model = HRMCPFusion(visual_embed_dim=128, text_embed_dim=64)
        model.eval()
        visual = torch.randn(3, 128)
        text = torch.randn(3, 64)
        with torch.no_grad():
            output = model(visual, text)
        assert output.enhanced_joint_embedding.shape == (3, 192)


# ---------------------------------------------------------------------------
# cognitive.propensity
# ---------------------------------------------------------------------------

class TestClassifyPropensity:
    def test_trustworthiness_formula(self) -> None:
        aigc = torch.tensor([0.2])
        belief = torch.tensor([0.9])
        dissem = torch.tensor([0.1])
        metrics = classify_propensity(aigc, belief, dissem)
        expected_trust = 0.9 - 0.2
        assert metrics.trustworthiness.item() == pytest.approx(expected_trust, abs=1e-6)

    def test_impact_formula(self) -> None:
        aigc = torch.tensor([0.1])
        belief = torch.tensor([0.6])
        dissem = torch.tensor([0.3])
        metrics = classify_propensity(aigc, belief, dissem)
        expected_impact = 0.6 + 0.3
        assert metrics.impact.item() == pytest.approx(expected_impact, abs=1e-6)

    def test_trustworthiness_normalized_range(self) -> None:
        # Extremes: belief=1, aigc=0 -> trust=1 -> normalized=1
        metrics_max = classify_propensity(
            torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0])
        )
        assert metrics_max.trustworthiness_normalized.item() == pytest.approx(1.0)

        # belief=0, aigc=1 -> trust=-1 -> normalized=0
        metrics_min = classify_propensity(
            torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0])
        )
        assert metrics_min.trustworthiness_normalized.item() == pytest.approx(0.0)

    def test_impact_normalized_range(self) -> None:
        metrics_max = classify_propensity(
            torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1.0])
        )
        assert metrics_max.impact_normalized.item() == pytest.approx(1.0)

        metrics_min = classify_propensity(
            torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
        )
        assert metrics_min.impact_normalized.item() == pytest.approx(0.0)

    def test_suspicious_trust_bucket_triggers_on_high_aigc_and_belief(self) -> None:
        metrics = classify_propensity(
            aigc_likelihood=torch.tensor([0.9]),
            belief=torch.tensor([0.9]),
            dissemination_propensity=torch.tensor([0.1]),
        )
        assert metrics.trust_level[0] == TrustLevel.SUSPICIOUS

    def test_viral_risk_bucket_triggers_on_high_belief_and_dissemination(self) -> None:
        metrics = classify_propensity(
            aigc_likelihood=torch.tensor([0.1]),
            belief=torch.tensor([0.9]),
            dissemination_propensity=torch.tensor([0.9]),
        )
        assert metrics.impact_level[0] == ImpactLevel.VIRAL_RISK

    def test_batch_input_produces_matching_length_outputs(self) -> None:
        batch_size = 5
        metrics = classify_propensity(
            aigc_likelihood=torch.rand(batch_size, 1),
            belief=torch.rand(batch_size, 1),
            dissemination_propensity=torch.rand(batch_size, 1),
        )
        assert metrics.trustworthiness.shape == (batch_size,)
        assert len(metrics.trust_level) == batch_size
        assert len(metrics.impact_level) == batch_size

    def test_as_dict_has_expected_keys(self) -> None:
        metrics = classify_propensity(
            torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5])
        )
        d = metrics.as_dict()
        assert set(d.keys()) == {
            "trustworthiness", "trustworthiness_normalized",
            "impact", "impact_normalized", "trust_level", "impact_level",
        }


class TestPropensityClassifier:
    def test_classify_end_to_end_returns_composite_metrics(self) -> None:
        model = HRMCPFusion()
        classifier = PropensityClassifier(model)
        visual = torch.randn(2, 512)
        text = torch.randn(2, 512)
        result = classifier.classify(visual, text)

        assert isinstance(result, CompositeMetrics)
        assert result.trustworthiness.shape == (2,)
        assert len(result.trust_level) == 2

    def test_classify_puts_model_in_eval_mode(self) -> None:
        model = HRMCPFusion()
        model.train()  # start in training mode
        classifier = PropensityClassifier(model)
        classifier.classify(torch.randn(1, 512), torch.randn(1, 512))
        assert model.training is False


# ---------------------------------------------------------------------------
# agents.schemas
# ---------------------------------------------------------------------------

class TestEvidenceItem:
    def test_valid_construction(self) -> None:
        item = EvidenceItem(
            evidence_id="ev_1", snippet="Some evidence text.", source_url=None, retrieved_at_step=0
        )
        assert item.evidence_id == "ev_1"

    def test_blank_evidence_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem(evidence_id="   ", snippet="x", retrieved_at_step=0)

    def test_negative_step_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem(evidence_id="ev_1", snippet="x", retrieved_at_step=-1)


class TestVerifiableFact:
    def test_verifiable_claim_without_citations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerifiableFact(claim_key_point="Some claim", is_verifiable=True)

    def test_unverifiable_claim_with_citations_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerifiableFact(
                claim_key_point="Some claim",
                is_verifiable=False,
                supporting_evidence_ids=["ev_1"],
            )

    def test_verifiable_claim_with_citation_accepted(self) -> None:
        fact = VerifiableFact(
            claim_key_point="Some claim",
            is_verifiable=True,
            supporting_evidence_ids=["ev_1"],
        )
        assert fact.supporting_evidence_ids == ["ev_1"]

    def test_citations_are_grounded_true_case(self) -> None:
        fact = VerifiableFact(
            claim_key_point="claim", is_verifiable=True, supporting_evidence_ids=["ev_1", "ev_2"]
        )
        assert fact.citations_are_grounded({"ev_1", "ev_2", "ev_3"}) is True

    def test_citations_are_grounded_false_case(self) -> None:
        fact = VerifiableFact(
            claim_key_point="claim", is_verifiable=True, supporting_evidence_ids=["ev_999"]
        )
        assert fact.citations_are_grounded({"ev_1", "ev_2"}) is False


class TestExplainedPrediction:
    def test_rejects_hallucinated_citation_not_in_evidence_pool(self) -> None:
        fact = VerifiableFact(
            claim_key_point="claim", is_verifiable=True, supporting_evidence_ids=["ev_ghost"]
        )
        with pytest.raises(ValidationError):
            ExplainedPrediction(
                verdict=VerdictLabel.TRUE,
                confidence=0.8,
                reasons=["some reason"],
                verifiable_facts=[fact],
                evidence_pool=[],  # ev_ghost is not here -> should fail validation
            )

    def test_accepts_grounded_citation(self) -> None:
        evidence = EvidenceItem(
            evidence_id="ev_1", snippet="supporting text", retrieved_at_step=0
        )
        fact = VerifiableFact(
            claim_key_point="claim", is_verifiable=True, supporting_evidence_ids=["ev_1"]
        )
        prediction = ExplainedPrediction(
            verdict=VerdictLabel.TRUE,
            confidence=0.8,
            reasons=["grounded reason"],
            verifiable_facts=[fact],
            evidence_pool=[evidence],
        )
        assert prediction.verdict == VerdictLabel.TRUE

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExplainedPrediction(
                verdict=VerdictLabel.UNCERTAIN, confidence=1.5, reasons=["x"]
            )

    def test_empty_reasons_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExplainedPrediction(verdict=VerdictLabel.UNCERTAIN, confidence=0.5, reasons=[])

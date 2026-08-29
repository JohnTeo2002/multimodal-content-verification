"""
cognitive.propensity
=====================
Turns the three raw HR-MCP propensity scores (AIGC Likelihood, Belief,
Dissemination Propensity -- each in [0, 1]) into:
    1. Composite, normalised metrics (Trustworthiness, Impact) that are more
       directly actionable for a downstream moderation/ranking system than
       the three raw scores in isolation.
    2. Human-readable categorical classifications (e.g. "high risk / low
       trust") derived from those composite metrics via configurable
       thresholds.

COMPOSITE METRIC DEFINITIONS
    Trustworthiness = Belief - AIGC_Likelihood
        Intuition: content that people strongly believe (`Belief` -> 1)
        AND that is strongly perceived as AI-generated (`AIGC_Likelihood`
        -> 1) is contradictory/suspicious -- high believability *despite*
        looking synthetic should lower our confidence in that belief being
        well-founded. Trustworthiness therefore ranges over [-1, 1]:
            +1  => maximal belief, zero perceived synthesis (most trustworthy)
            -1  => zero belief, maximal perceived synthesis (least trustworthy)

    Impact = Belief + Dissemination_Propensity
        Intuition: content that is both strongly believed AND strongly
        likely to be shared has the highest potential real-world impact
        (true information spreading widely is high positive impact;
        false information spreading widely is high *negative* real-world
        impact -- Impact alone is directionless on truth value, which is
        why it must always be interpreted alongside Trustworthiness).
        Impact ranges over [0, 2].

    Both composite metrics are additionally exposed in a normalised [0, 1]
    form (`trustworthiness_normalized`, `impact_normalized`) via simple
    min-max rescaling of their known ranges, to make them easier to feed
    into downstream systems (e.g. a single moderation-priority score) that
    expect [0, 1]-ranged inputs uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class TrustLevel(str, Enum):
    """Categorical bucket for the Trustworthiness composite metric."""

    HIGH_TRUST = "high_trust"
    MODERATE_TRUST = "moderate_trust"
    LOW_TRUST = "low_trust"
    SUSPICIOUS = "suspicious"  # Strong belief + strong perceived synthesis.


class ImpactLevel(str, Enum):
    """Categorical bucket for the Impact composite metric."""

    LOW_IMPACT = "low_impact"
    MODERATE_IMPACT = "moderate_impact"
    HIGH_IMPACT = "high_impact"
    VIRAL_RISK = "viral_risk"  # Strong belief + strong dissemination propensity.


@dataclass
class CompositeMetrics:
    """
    Structured result of `classify_propensity`.

    Attributes:
        trustworthiness: Raw composite score in [-1, 1].
        trustworthiness_normalized: Min-max rescaled to [0, 1].
        impact: Raw composite score in [0, 2].
        impact_normalized: Min-max rescaled to [0, 1].
        trust_level: Categorical bucket for `trustworthiness`.
        impact_level: Categorical bucket for `impact`.
    """

    trustworthiness: torch.Tensor
    trustworthiness_normalized: torch.Tensor
    impact: torch.Tensor
    impact_normalized: torch.Tensor
    trust_level: list[TrustLevel]
    impact_level: list[ImpactLevel]

    def as_dict(self) -> dict:
        return {
            "trustworthiness": self.trustworthiness.tolist(),
            "trustworthiness_normalized": self.trustworthiness_normalized.tolist(),
            "impact": self.impact.tolist(),
            "impact_normalized": self.impact_normalized.tolist(),
            "trust_level": [t.value for t in self.trust_level],
            "impact_level": [i.value for i in self.impact_level],
        }


def _bucketize_trust(
    trustworthiness: torch.Tensor,
    aigc_likelihood: torch.Tensor,
    belief: torch.Tensor,
    suspicious_aigc_threshold: float,
    suspicious_belief_threshold: float,
    high_trust_threshold: float,
    low_trust_threshold: float,
) -> list[TrustLevel]:
    levels: list[TrustLevel] = []
    for trust_val, aigc_val, belief_val in zip(
        trustworthiness.flatten().tolist(),
        aigc_likelihood.flatten().tolist(),
        belief.flatten().tolist(),
    ):
        # "Suspicious" takes priority: high belief co-occurring with high
        # perceived synthesis is a specific, actionable red flag distinct
        # from a merely "low" trust score driven by low belief alone.
        if aigc_val >= suspicious_aigc_threshold and belief_val >= suspicious_belief_threshold:
            levels.append(TrustLevel.SUSPICIOUS)
        elif trust_val >= high_trust_threshold:
            levels.append(TrustLevel.HIGH_TRUST)
        elif trust_val >= low_trust_threshold:
            levels.append(TrustLevel.MODERATE_TRUST)
        else:
            levels.append(TrustLevel.LOW_TRUST)
    return levels


def _bucketize_impact(
    impact: torch.Tensor,
    belief: torch.Tensor,
    dissemination: torch.Tensor,
    viral_belief_threshold: float,
    viral_dissemination_threshold: float,
    high_impact_threshold: float,
    moderate_impact_threshold: float,
) -> list[ImpactLevel]:
    levels: list[ImpactLevel] = []
    for impact_val, belief_val, dissem_val in zip(
        impact.flatten().tolist(),
        belief.flatten().tolist(),
        dissemination.flatten().tolist(),
    ):
        if belief_val >= viral_belief_threshold and dissem_val >= viral_dissemination_threshold:
            levels.append(ImpactLevel.VIRAL_RISK)
        elif impact_val >= high_impact_threshold:
            levels.append(ImpactLevel.HIGH_IMPACT)
        elif impact_val >= moderate_impact_threshold:
            levels.append(ImpactLevel.MODERATE_IMPACT)
        else:
            levels.append(ImpactLevel.LOW_IMPACT)
    return levels


def classify_propensity(
    aigc_likelihood: torch.Tensor,
    belief: torch.Tensor,
    dissemination_propensity: torch.Tensor,
    *,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
    suspicious_aigc_threshold: float = 0.7,
    suspicious_belief_threshold: float = 0.7,
    high_trust_threshold: float = 0.5,
    low_trust_threshold: float = 0.0,
    viral_belief_threshold: float = 0.7,
    viral_dissemination_threshold: float = 0.7,
    high_impact_threshold: float = 1.3,
    moderate_impact_threshold: float = 0.7,
) -> CompositeMetrics:
    """
    Compute normalised composite metrics and categorical classifications
    from the three raw HR-MCP propensity scores.

    Args:
        aigc_likelihood: (batch, 1) or (batch,), values in [0, 1].
        belief: (batch, 1) or (batch,), values in [0, 1].
        dissemination_propensity: (batch, 1) or (batch,), values in [0, 1].
        clamp_min / clamp_max: Bounds applied to the raw `trustworthiness`
            score before normalisation (guards against any minor
            out-of-range float noise; Trustworthiness is mathematically
            already within [-1, 1] given [0, 1]-ranged inputs).
        suspicious_*_threshold: Thresholds for the SUSPICIOUS trust bucket.
        high_trust_threshold / low_trust_threshold: Cut points separating
            HIGH_TRUST / MODERATE_TRUST / LOW_TRUST.
        viral_*_threshold: Thresholds for the VIRAL_RISK impact bucket.
        high_impact_threshold / moderate_impact_threshold: Cut points
            separating HIGH_IMPACT / MODERATE_IMPACT / LOW_IMPACT.

    Returns:
        A populated `CompositeMetrics` dataclass.
    """
    # Flatten trailing singleton dims so this function works whether the
    # caller passes (batch, 1) tensors straight from HRMCPOutput or already
    # -squeezed (batch,) tensors.
    aigc = aigc_likelihood.reshape(-1)
    bel = belief.reshape(-1)
    dissem = dissemination_propensity.reshape(-1)

    # --- Trustworthiness = Belief - AIGC_Likelihood -------------------------
    trustworthiness = torch.clamp(bel - aigc, min=clamp_min, max=clamp_max)  # in [-1, 1]
    # Min-max rescale [-1, 1] -> [0, 1]: (x - min) / (max - min)
    trustworthiness_normalized = (trustworthiness - clamp_min) / (clamp_max - clamp_min)

    # --- Impact = Belief + Dissemination_Propensity -------------------------
    impact = torch.clamp(bel + dissem, min=0.0, max=2.0)  # in [0, 2]
    impact_normalized = impact / 2.0  # Min-max rescale [0, 2] -> [0, 1]

    trust_level = _bucketize_trust(
        trustworthiness, aigc, bel,
        suspicious_aigc_threshold, suspicious_belief_threshold,
        high_trust_threshold, low_trust_threshold,
    )
    impact_level = _bucketize_impact(
        impact, bel, dissem,
        viral_belief_threshold, viral_dissemination_threshold,
        high_impact_threshold, moderate_impact_threshold,
    )

    return CompositeMetrics(
        trustworthiness=trustworthiness,
        trustworthiness_normalized=trustworthiness_normalized,
        impact=impact,
        impact_normalized=impact_normalized,
        trust_level=trust_level,
        impact_level=impact_level,
    )


class PropensityClassifier:
    """
    Thin, stateless orchestration wrapper that combines an `HRMCPFusion`
    forward pass with `classify_propensity` in one call -- this is the
    object other layers (e.g. `agents/mcp_server.py`'s
    `get_human_perceptions` tool) should typically depend on, rather than
    reaching into `HRMCPFusion` and `classify_propensity` separately.

    Kept as a plain Python class (not an `nn.Module`) because it owns no
    learnable parameters of its own -- it simply composes an existing
    `nn.Module` with a pure post-processing function.
    """

    def __init__(self, hr_mcp_model: "torch.nn.Module", **classify_kwargs: float) -> None:
        self.hr_mcp_model = hr_mcp_model
        self.classify_kwargs = classify_kwargs

    @torch.no_grad()
    def classify(
        self, visual_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> CompositeMetrics:
        """
        Run HR-MCP fusion in inference mode (no gradient tracking) and
        return the resulting composite metrics + classifications.
        """
        self.hr_mcp_model.eval()
        output = self.hr_mcp_model(visual_embedding, text_embedding)
        return classify_propensity(
            aigc_likelihood=output.aigc_likelihood,
            belief=output.belief,
            dissemination_propensity=output.dissemination_propensity,
            **self.classify_kwargs,
        )

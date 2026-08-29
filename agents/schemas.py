"""
agents.schemas
===============
Pydantic V2 data models for AgentFact's auditable, structured outputs.

WHY STRUCTURED OUTPUTS AT ALL?
    A raw LLM/agent text response ("This looks fake because...") is not
    machine-checkable and cannot be safely logged, diffed, or fed into a
    downstream moderation pipeline. By constraining AgentFact's final
    answer to a validated Pydantic schema, we get:
        1. Guaranteed shape: every field the downstream consumer expects is
           either present and correctly typed, or a `ValidationError` is
           raised loudly at construction time (fail fast, not silently).
        2. Auditability: every claim (`VerifiableFact`) must cite specific
           `EvidenceItem` IDs that were actually returned by a search/tool
           call earlier in the ReAct loop -- see the
           `no_hallucinated_citations` validator on `VerifiableFact`, which
           is enforced at the orchestrator level (see
           `agents.agent_fact.AgentFact._validate_citations`).
        3. Serialisability: `.model_dump_json()` gives a stable, versioned
           JSON contract that other services (or a human reviewer UI) can
           consume without needing to understand any Python internals.

NO HALLUCINATED CITATIONS
    Pydantic's field-level validation can only check *shape* (e.g. "is this
    a non-empty list of strings?"), not *semantic grounding* (e.g. "does
    evidence ID 'ev_003' actually exist in the evidence pool the agent was
    given?"). The latter check requires cross-referencing against the live
    evidence pool at runtime, so it is implemented as an explicit method
    (`VerifiableFact.citations_are_grounded`) that the orchestrator calls
    with the actual evidence pool, rather than as a Pydantic
    `field_validator` (which only sees the single field's own value, not
    external context).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class VerdictLabel(str, Enum):
    """Final veracity verdict emitted by AgentFact."""

    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"
    UNVERIFIABLE = "unverifiable"
    UNCERTAIN = "uncertain"  # Confidence fell below the configured threshold.


class EvidenceItem(BaseModel):
    """
    A single piece of retrieved evidence (e.g. a search result snippet, a
    reverse-image-search match, or a knowledge-base lookup result).

    Attributes:
        evidence_id: Stable identifier unique within a single AgentFact run
            (e.g. "ev_001"). Referenced by `VerifiableFact.supporting_evidence_ids`.
        snippet: Short, human-readable excerpt of the evidence content.
            Kept intentionally short (see `max_length`) -- this schema
            stores *pointers and summaries* for audit purposes, not full
            scraped documents.
        source_url: Origin URL of the evidence, if it came from a live
            web/API lookup. `None` for closed-book "prior knowledge"
            evidence (see `agents.agent_fact.AgentMode.CLOSED_BOOK`).
        retrieved_at_step: Which ReAct loop step produced this evidence,
            for reconstructing the reasoning timeline during audit.
    """

    evidence_id: str = Field(..., min_length=1, max_length=64)
    snippet: str = Field(..., min_length=1, max_length=1000)
    source_url: Optional[str] = None
    retrieved_at_step: int = Field(..., ge=0)

    @field_validator("evidence_id")
    @classmethod
    def _validate_id_format(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_id must not be blank or whitespace-only")
        return value


class VerifiableFact(BaseModel):
    """
    A single, atomic factual claim extracted from the media under review,
    linked to the specific evidence that supports (or refutes) it.

    Attributes:
        claim_key_point: The atomic claim being assessed, phrased as a
            single, checkable statement (e.g. "The photo was taken in
            Paris in 2024").
        supporting_evidence_ids: IDs of `EvidenceItem`s that support this
            claim. MUST be non-empty for any claim that is not itself
            marked as unverifiable -- this is the core "no hallucinated
            citations" guarantee: the orchestrator must have actually
            retrieved *something* to point to.
        refuting_evidence_ids: IDs of `EvidenceItem`s that contradict this
            claim, if any.
        is_verifiable: Whether sufficient evidence exists (in either
            direction) to assess this claim at all. When False, both
            evidence-ID lists are expected to be empty (enforced below).
    """

    claim_key_point: str = Field(..., min_length=1, max_length=500)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    refuting_evidence_ids: List[str] = Field(default_factory=list)
    is_verifiable: bool = True

    @model_validator(mode="after")
    def _check_citation_consistency(self) -> "VerifiableFact":
        # A claim that IS verifiable must cite at least one piece of
        # evidence, in either direction -- an "empty-handed" verifiable
        # claim is, by definition, a hallucinated/unsupported assertion.
        if self.is_verifiable and not (self.supporting_evidence_ids or self.refuting_evidence_ids):
            raise ValueError(
                "A claim marked is_verifiable=True must cite at least one "
                "supporting or refuting evidence_id. Mark is_verifiable=False "
                "instead if no evidence could be retrieved for this claim."
            )
        # Conversely, an explicitly *unverifiable* claim should not carry
        # citations -- that would be a contradictory annotation.
        if not self.is_verifiable and (self.supporting_evidence_ids or self.refuting_evidence_ids):
            raise ValueError(
                "A claim marked is_verifiable=False must not cite any "
                "evidence_ids (that would contradict the unverifiable flag)."
            )
        return self

    def citations_are_grounded(self, known_evidence_ids: set[str]) -> bool:
        """
        Cross-reference this claim's cited evidence IDs against the live
        evidence pool actually collected during the current AgentFact run.

        This is the semantic ("does this ID *really exist*?") half of the
        "no hallucinated citations" guarantee; the structural half
        ("*some* ID must be cited") is enforced by `_check_citation_consistency`
        above at construction time.

        Args:
            known_evidence_ids: The set of `evidence_id` values present in
                the orchestrator's actual, retrieved `EvidenceItem` pool.

        Returns:
            True iff every cited ID (supporting and refuting) is present
            in `known_evidence_ids`.
        """
        cited = set(self.supporting_evidence_ids) | set(self.refuting_evidence_ids)
        return cited.issubset(known_evidence_ids)


class ExplainedPrediction(BaseModel):
    """
    The final, top-level auditable output of an AgentFact run.

    Attributes:
        verdict: The categorical veracity label.
        confidence: Calibrated-ish confidence in `verdict`, in [0, 1].
            Below `agents.agent_fact.AgentFact.confidence_threshold`, the
            orchestrator overrides `verdict` to `VerdictLabel.UNCERTAIN`
            regardless of what the raw prediction suggested.
        reasons: Ordered, human-readable list of the key reasons behind the
            verdict -- intended for direct display in a reviewer UI.
        verifiable_facts: The atomic claims assessed, each with grounded
            evidence citations (see `VerifiableFact`).
        evidence_pool: The full set of evidence collected during the run,
            for audit / re-verification purposes.
        post_hoc_explanation_json: A free-form (but still-a-dict) JSON
            blob capturing additional structured explanation detail (e.g.
            attention-weight summaries, tool-call traces) that doesn't fit
            neatly into the fixed fields above. Kept as `dict` rather than
            a loose string so it remains machine-parseable.
    """

    verdict: VerdictLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(..., min_length=1)
    verifiable_facts: List[VerifiableFact] = Field(default_factory=list)
    evidence_pool: List[EvidenceItem] = Field(default_factory=list)
    post_hoc_explanation_json: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_all_citations_grounded(self) -> "ExplainedPrediction":
        known_ids = {item.evidence_id for item in self.evidence_pool}
        for fact in self.verifiable_facts:
            if not fact.citations_are_grounded(known_ids):
                raise ValueError(
                    f"VerifiableFact {fact.claim_key_point!r} cites an "
                    f"evidence_id not present in evidence_pool -- this is "
                    f"exactly the 'hallucinated citation' failure mode this "
                    f"schema exists to prevent."
                )
        return self

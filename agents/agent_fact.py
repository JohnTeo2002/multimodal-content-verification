"""
agents.agent_fact
===================
AgentFact: a multi-agent, ReAct-style ("Reason + Act") fact-verification
orchestrator.

WHAT IS "ReAct" HERE?
    ReAct (Yao et al., 2022) interleaves three phases in a loop:
        Think  -> the agent reasons in natural language about what it
                  still needs to find out.
        Act    -> the agent invokes a tool (a web search, a reverse-image
                  lookup, a call into the cognitive/core models) based on
                  that reasoning.
        Observe -> the agent incorporates the tool's return value into its
                  running context before the next Think step.
    This loop repeats until the agent has enough evidence to commit to a
    verdict, or a step budget (`max_react_steps`) is exhausted.

THREE OPERATING MODES
    * CLOSED_BOOK:      No external tools are called at all. AgentFact
                         reasons only from the claim text and (optionally)
                         the core/cognitive model outputs already computed
                         upstream. Fully deterministic, air-gap-safe,
                         zero external dependencies -- but cannot verify
                         claims requiring outside knowledge.
    * EVIDENCE_BOUNDED:  AgentFact may only use evidence explicitly
                         supplied to it up-front (e.g. a fixed, pre-fetched
                         document set) -- it cannot issue *new* live search
                         queries. This models real deployments where a
                         retrieval step has already run and the agent's job
                         is purely to reason over a bounded context.
    * OPEN_WEB:          AgentFact may issue new tool calls (web search,
                         reverse image search) at each ReAct step, subject
                         to `max_react_steps`. This is the most powerful
                         but also the mode requiring live external API
                         credentials (see the TODO-EXTENSION-MARKERs below).

DECOUPLING INFERENCE FROM LIVE RETRIEVAL
    Every "tool" method on `AgentFact` (`_web_search_tool`,
    `_reverse_image_search_tool`) is implemented as a **stub** that returns
    clearly-fake, deterministic placeholder data by default, and is called
    through the same code path regardless of mode. This lets the *entire*
    orchestration/reasoning logic run and be unit-tested in an air-gapped
    CI environment without ever making a network call, while keeping the
    exact integration point obvious and swappable for a real deployment.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, List, Optional

from agents.schemas import (
    EvidenceItem,
    ExplainedPrediction,
    VerdictLabel,
    VerifiableFact,
)

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    CLOSED_BOOK = "closed_book"
    EVIDENCE_BOUNDED = "evidence_bounded"
    OPEN_WEB = "open_web"


class ReActStep:
    """
    A single recorded Think/Act/Observe step, kept for audit trails and
    for populating `ExplainedPrediction.post_hoc_explanation_json`.
    """

    __slots__ = ("step_index", "thought", "action", "observation")

    def __init__(self, step_index: int, thought: str, action: str, observation: str) -> None:
        self.step_index = step_index
        self.thought = thought
        self.action = action
        self.observation = observation

    def as_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
        }


class AgentFact:
    """
    ReAct-style multi-agent fact-verification orchestrator.

    Args:
        mode: One of CLOSED_BOOK, EVIDENCE_BOUNDED, OPEN_WEB (see module
            docstring). Determines which tool calls are permitted.
        max_react_steps: Hard cap on the number of Think/Act/Observe
            iterations, preventing unbounded loops.
        confidence_threshold: If the agent's final confidence falls below
            this value, the verdict is overridden to `VerdictLabel.UNCERTAIN`
            regardless of the raw predicted label -- we would rather report
            "I don't know" than a low-confidence wrong answer.
        preloaded_evidence: For EVIDENCE_BOUNDED mode, the fixed pool of
            evidence the agent is allowed to reason over. Ignored in
            CLOSED_BOOK mode; used as a *seed* pool (extendable via live
            tool calls) in OPEN_WEB mode.
        search_tool: Optional injected callable to use in place of the
            built-in `_web_search_tool` stub -- this is the primary
            dependency-injection point for wiring in a real search
            provider (see TODO-EXTENSION-MARKER below). Must match the
            `_web_search_tool` signature.
    """

    def __init__(
        self,
        mode: AgentMode = AgentMode.EVIDENCE_BOUNDED,
        max_react_steps: int = 6,
        confidence_threshold: float = 0.6,
        preloaded_evidence: Optional[List[EvidenceItem]] = None,
        search_tool: Optional[Callable[[str, int], List[EvidenceItem]]] = None,
    ) -> None:
        self.mode = mode
        self.max_react_steps = max_react_steps
        self.confidence_threshold = confidence_threshold
        self.preloaded_evidence: List[EvidenceItem] = list(preloaded_evidence or [])
        self._search_tool = search_tool or self._web_search_tool

    # -------------------------------------------------------------------
    # Tool stubs -- clearly marked, deterministic, offline-safe
    # -------------------------------------------------------------------

    def _web_search_tool(self, query: str, step_index: int) -> List[EvidenceItem]:
        """
        STUB: general-purpose web search tool.

        TODO-EXTENSION-MARKER [SERPER_INTEGRATION]:
        Replace this method body with a real call to the Serper.dev Search
        API (or equivalent), e.g.:

            import httpx
            response = httpx.post(
                self.config["search_providers"]["serper"]["endpoint"],
                headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
                json={"q": query},
                timeout=10.0,
            )
            results = response.json().get("organic", [])
            return [
                EvidenceItem(
                    evidence_id=f"ev_{step_index}_{i}",
                    snippet=r["snippet"][:1000],
                    source_url=r["link"],
                    retrieved_at_step=step_index,
                )
                for i, r in enumerate(results)
            ]

        Only OPEN_WEB mode is permitted to invoke this method (enforced by
        `_dispatch_action` below); CLOSED_BOOK and EVIDENCE_BOUNDED modes
        never reach this code path, which is exactly what keeps this
        repository's test suite runnable without network access.
        """
        logger.debug("STUB web search invoked for query=%r (no live network call made)", query)
        return [
            EvidenceItem(
                evidence_id=f"ev_stub_{step_index}_0",
                snippet=(
                    f"[STUB EVIDENCE -- no live search performed] "
                    f"Placeholder result for query: {query!r}"
                ),
                source_url=None,
                retrieved_at_step=step_index,
            )
        ]

    def _reverse_image_search_tool(self, image_ref: str, step_index: int) -> List[EvidenceItem]:
        """
        STUB: reverse image search / visual provenance lookup.

        TODO-EXTENSION-MARKER [GVISION_INTEGRATION]:
        Replace this method body with a real call to the Google Cloud
        Vision API's `web_detection` feature (or equivalent reverse-image
        search provider), e.g.:

            import httpx
            response = httpx.post(
                self.config["search_providers"]["google_vision"]["endpoint"],
                params={"key": os.environ["GVISION_API_KEY"]},
                json={"requests": [{
                    "image": {"source": {"imageUri": image_ref}},
                    "features": [{"type": "WEB_DETECTION"}],
                }]},
                timeout=10.0,
            )
            matches = response.json()["responses"][0].get(
                "webDetection", {}
            ).get("pagesWithMatchingImages", [])
            return [
                EvidenceItem(
                    evidence_id=f"ev_img_{step_index}_{i}",
                    snippet=f"Matching page found: {m.get('url', '')}",
                    source_url=m.get("url"),
                    retrieved_at_step=step_index,
                )
                for i, m in enumerate(matches)
            ]
        """
        logger.debug(
            "STUB reverse-image search invoked for image_ref=%r (no live call made)", image_ref
        )
        return [
            EvidenceItem(
                evidence_id=f"ev_img_stub_{step_index}_0",
                snippet=(
                    f"[STUB EVIDENCE -- no live reverse-image search performed] "
                    f"Placeholder result for image_ref: {image_ref!r}"
                ),
                source_url=None,
                retrieved_at_step=step_index,
            )
        ]

    # -------------------------------------------------------------------
    # ReAct loop
    # -------------------------------------------------------------------

    def _dispatch_action(
        self, action: str, query: str, step_index: int
    ) -> tuple[str, List[EvidenceItem]]:
        """
        Route a single Act phase to the appropriate tool, respecting the
        current `AgentMode`'s permissions.

        Returns:
            (observation_text, new_evidence_items)
        """
        if self.mode == AgentMode.CLOSED_BOOK:
            observation = (
                "Mode is CLOSED_BOOK: no external tools may be called. "
                "Reasoning must rely solely on the claim text and any "
                "model outputs already provided."
            )
            return observation, []

        if self.mode == AgentMode.EVIDENCE_BOUNDED:
            observation = (
                "Mode is EVIDENCE_BOUNDED: no new live tool calls are "
                "permitted; searching only within the preloaded evidence pool."
            )
            return observation, []

        # OPEN_WEB mode: actually dispatch to a tool.
        if action == "web_search":
            new_evidence = self._search_tool(query, step_index)
        elif action == "reverse_image_search":
            new_evidence = self._reverse_image_search_tool(query, step_index)
        else:
            new_evidence = []
        observation = f"Retrieved {len(new_evidence)} evidence item(s) via {action!r}."
        return observation, new_evidence

    def _search_preloaded_evidence(self, claim: str) -> List[EvidenceItem]:
        """
        Naive keyword-overlap search over `self.preloaded_evidence`, used
        in EVIDENCE_BOUNDED mode (and as a first pass in OPEN_WEB mode
        before falling back to live search). Kept intentionally simple
        (no embeddings/vector search) so this module has zero ML
        dependencies of its own -- it orchestrates the `core`/`cognitive`
        models, but is not itself one.
        """
        claim_terms = {t.lower() for t in claim.split() if len(t) > 3}
        matches: List[EvidenceItem] = []
        for item in self.preloaded_evidence:
            snippet_terms = {t.lower() for t in item.snippet.split() if len(t) > 3}
            if claim_terms & snippet_terms:
                matches.append(item)
        return matches

    def verify_claim(self, claim: str, image_ref: Optional[str] = None) -> ExplainedPrediction:
        """
        Run the full ReAct verification loop for a single claim (with an
        optional associated image reference) and return a validated
        `ExplainedPrediction`.

        Args:
            claim: The textual claim/caption to verify.
            image_ref: Optional identifier/path/URL for an associated
                image, used to trigger reverse-image-search style
                evidence-gathering in OPEN_WEB mode.

        Returns:
            A fully-populated, schema-validated `ExplainedPrediction`.
        """
        evidence_pool: List[EvidenceItem] = list(self.preloaded_evidence)
        react_trace: List[ReActStep] = []

        # --- Step 0: always start by checking the preloaded pool ----------
        seed_matches = self._search_preloaded_evidence(claim)
        react_trace.append(
            ReActStep(
                step_index=0,
                thought=f"Checking preloaded evidence pool for claim: {claim!r}",
                action="search_preloaded_evidence",
                observation=f"Found {len(seed_matches)} matching preloaded evidence item(s).",
            )
        )

        # --- ReAct loop (only actually does work in OPEN_WEB mode) ---------
        for step_index in range(1, self.max_react_steps):
            if self.mode != AgentMode.OPEN_WEB:
                # CLOSED_BOOK / EVIDENCE_BOUNDED modes never issue new tool
                # calls -- one informational trace entry is enough, no need
                # to loop further and pad the trace with no-ops.
                thought = (
                    f"Mode {self.mode.value} does not permit further tool "
                    f"calls; proceeding to synthesis with current evidence."
                )
                action = "none"
                observation, _new_evidence = self._dispatch_action(action, claim, step_index)
                react_trace.append(ReActStep(step_index, thought, action, observation))
                break

            if len(seed_matches) >= 1 and step_index > 1:
                # Already have some grounding evidence from the preloaded
                # pool; stop issuing further live searches to respect the
                # step budget and avoid redundant calls.
                break

            thought = f"Need external evidence for claim: {claim!r}. Issuing a web search."
            action = "web_search"
            observation, new_evidence = self._dispatch_action(action, claim, step_index)
            evidence_pool.extend(new_evidence)
            react_trace.append(ReActStep(step_index, thought, action, observation))

            if image_ref is not None:
                img_thought = f"Claim has an associated image ({image_ref}); checking provenance."
                img_action = "reverse_image_search"
                img_observation, img_evidence = self._dispatch_action(
                    img_action, image_ref, step_index
                )
                evidence_pool.extend(img_evidence)
                react_trace.append(ReActStep(step_index, img_thought, img_action, img_observation))

        # --- Synthesis: build the VerifiableFact + verdict -----------------
        supporting_ids = [item.evidence_id for item in seed_matches]
        supporting_ids += [
            item.evidence_id
            for item in evidence_pool
            if item not in seed_matches and item not in self.preloaded_evidence
        ]

        is_verifiable = len(supporting_ids) > 0
        verifiable_fact = VerifiableFact(
            claim_key_point=claim,
            supporting_evidence_ids=supporting_ids if is_verifiable else [],
            refuting_evidence_ids=[],
            is_verifiable=is_verifiable,
        )

        # A simple, transparent confidence heuristic: more corroborating
        # evidence -> higher confidence, saturating at 0.9 to always leave
        # room for human review (we deliberately never emit 1.0 certainty).
        #
        # TODO-EXTENSION-MARKER [CONFIDENCE_MODEL]:
        # Replace this heuristic with a learned confidence calibration
        # model (e.g. a small classifier trained on historical AgentFact
        # runs vs. human-adjudicated outcomes) for production use.
        raw_confidence = min(0.9, 0.3 + 0.2 * len(supporting_ids))
        raw_verdict = VerdictLabel.TRUE if is_verifiable else VerdictLabel.UNVERIFIABLE

        if raw_confidence < self.confidence_threshold:
            final_verdict = VerdictLabel.UNCERTAIN
        else:
            final_verdict = raw_verdict

        reasons = [
            f"Operating mode: {self.mode.value}.",
            f"Total evidence items considered: {len(evidence_pool)}.",
            f"Claim {'has' if is_verifiable else 'lacks'} supporting evidence citations.",
        ]
        if final_verdict == VerdictLabel.UNCERTAIN:
            reasons.append(
                f"Confidence {raw_confidence:.2f} fell below threshold "
                f"{self.confidence_threshold:.2f}; verdict overridden to UNCERTAIN."
            )

        prediction = ExplainedPrediction(
            verdict=final_verdict,
            confidence=raw_confidence,
            reasons=reasons,
            verifiable_facts=[verifiable_fact],
            evidence_pool=evidence_pool,
            post_hoc_explanation_json={
                "react_trace": [step.as_dict() for step in react_trace],
                "mode": self.mode.value,
            },
        )
        return prediction

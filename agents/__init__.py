"""
agents
======
The agentic orchestration layer: Pydantic schemas for auditable structured
outputs, the AgentFact multi-agent ReAct-style fact-verification
orchestrator, and an async Model Context Protocol (MCP) server exposing the
`core`/`cognitive` models as callable tools.

This layer is intentionally decoupled from `core`/`cognitive`: it never
imports `torch` directly in its planning logic, only calling into the
neural network layers through well-typed function boundaries (mirroring how
a production system would call out to a model-serving microservice).
"""

from .schemas import EvidenceItem, VerifiableFact, ExplainedPrediction, VerdictLabel
from .agent_fact import AgentFact, AgentMode

__all__ = [
    "EvidenceItem",
    "VerifiableFact",
    "ExplainedPrediction",
    "VerdictLabel",
    "AgentFact",
    "AgentMode",
]

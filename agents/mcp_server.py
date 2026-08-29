"""
agents.mcp_server
===================
An asynchronous mock server implementing (a simplified subset of)
Anthropic's Model Context Protocol (MCP), exposing the `core`/`cognitive`
models and the `AgentFact` orchestrator as plug-and-play, independently
callable **tools**.

WHY MCP?
    MCP standardises how a host application (e.g. an LLM-driven assistant)
    discovers and invokes external capabilities ("tools") over a uniform
    JSON-schema'd interface, regardless of what is running behind each
    tool. By exposing our verification/cognitive stack as MCP tools, *any*
    MCP-compatible client can call `verify_media_claims` or
    `get_human_perceptions` without knowing anything about PyTorch,
    Pydantic, or this repository's internals.

WHY "MOCK" / STUBBED TRANSPORT?
    A production MCP server negotiates a real transport (stdio pipe to a
    subprocess, or Server-Sent Events over HTTP) and a JSON-RPC-based
    handshake/capability-negotiation protocol. Implementing a fully
    spec-compliant transport is out of scope for this repository (and
    would add a hard dependency on a specific MCP SDK version). Instead,
    `MCPServer` here implements the **tool-registration and
    async-dispatch** pattern that sits *underneath* whatever transport is
    chosen, so swapping in a real `mcp` Python SDK's
    `Server`/`stdio_server` primitives later only requires changing
    `MCPServer.run` -- every tool function itself is already a plain,
    testable `async def` with a JSON-schema-describable signature.

    TODO-EXTENSION-MARKER [REAL_MCP_TRANSPORT]:
    To wire this up to the official `mcp` Python SDK:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        app = Server("multimodal-verifier")
        @app.list_tools()
        async def list_tools(): ...   # translate self._tools registry
        @app.call_tool()
        async def call_tool(name, arguments): ...  # dispatch into self._tools
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

import torch

from agents.agent_fact import AgentFact, AgentMode
from agents.schemas import EvidenceItem, ExplainedPrediction
from cognitive.hr_mcp_fusion import HRMCPFusion
from cognitive.propensity import CompositeMetrics, PropensityClassifier
from core.verifier_model import DualBranchVerifier, VerifierMetrics, compute_verifier_metrics

logger = logging.getLogger(__name__)

ToolFunc = Callable[..., Awaitable[Dict[str, Any]]]


@dataclass
class ToolSpec:
    """
    Metadata describing a single registered MCP tool -- mirrors the shape
    of an MCP `Tool` descriptor (name, human-readable description, and an
    input JSON-schema-like dict) closely enough that translating this into
    a real MCP SDK's `Tool` object is a mechanical, one-line-per-field
    mapping.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolFunc


class MCPServer:
    """
    Async tool-registry-and-dispatch server exposing the verification
    stack as callable MCP tools.

    Args:
        verifier_model: A `DualBranchVerifier` instance used by the
            `verify_media_claims` tool for image-authenticity scoring.
        hr_mcp_model: An `HRMCPFusion` instance used by the
            `get_human_perceptions` tool for human-response propensity
            scoring.
        agent_fact: An `AgentFact` orchestrator used by
            `verify_media_claims` to produce an auditable, evidence-linked
            verdict alongside the raw model score.
    """

    def __init__(
        self,
        verifier_model: Optional[DualBranchVerifier] = None,
        hr_mcp_model: Optional[HRMCPFusion] = None,
        agent_fact: Optional[AgentFact] = None,
    ) -> None:
        self.verifier_model = verifier_model or DualBranchVerifier()
        self.hr_mcp_model = hr_mcp_model or HRMCPFusion()
        self.agent_fact = agent_fact or AgentFact(mode=AgentMode.EVIDENCE_BOUNDED)
        self.propensity_classifier = PropensityClassifier(self.hr_mcp_model)

        self.verifier_model.eval()
        self.hr_mcp_model.eval()

        self._tools: Dict[str, ToolSpec] = {}
        self._register_default_tools()

    # -------------------------------------------------------------------
    # Tool registration
    # -------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: ToolFunc,
    ) -> None:
        """
        Register a new tool. Raises if `name` is already registered, to
        catch accidental duplicate registration early (fail fast).
        """
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"Tool handler for {name!r} must be an async function")
        self._tools[name] = ToolSpec(name, description, input_schema, handler)
        logger.info("Registered MCP tool: %s", name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP-style tool descriptors for client-side discovery."""
        return [
            {"name": spec.name, "description": spec.description, "inputSchema": spec.input_schema}
            for spec in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool call by name, mirroring MCP's `call_tool` RPC
        semantics: unknown tool names or handler exceptions are surfaced
        as a structured error dict rather than raising, so a malformed
        client request cannot crash the server process.
        """
        spec = self._tools.get(name)
        if spec is None:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name!r}"}]}
        try:
            result = await spec.handler(**arguments)
            return {"isError": False, "content": [{"type": "json", "json": result}]}
        except Exception as exc:  # noqa: BLE001 -- intentionally broad at the RPC boundary
            logger.exception("Tool %r raised an exception", name)
            return {"isError": True, "content": [{"type": "text", "text": str(exc)}]}

    # -------------------------------------------------------------------
    # Built-in tool implementations
    # -------------------------------------------------------------------

    def _register_default_tools(self) -> None:
        self.register_tool(
            name="verify_media_claims",
            description=(
                "Run the dual-branch synthetic-media verifier over an image "
                "tensor AND an auditable AgentFact ReAct verification pass "
                "over an associated textual claim, returning both a raw "
                "authenticity score and a structured, evidence-linked verdict."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "image_tensor_shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Shape (batch, channels, H, W) used to synthesize a random probe tensor in this stub; a real client would pass actual pixel data through a separate binary channel.",
                    },
                    "claim": {"type": "string", "description": "Textual claim/caption to fact-check."},
                    "image_ref": {"type": ["string", "null"], "description": "Optional reference/URL for reverse-image-search."},
                },
                "required": ["image_tensor_shape", "claim"],
            },
            handler=self.verify_media_claims,
        )
        self.register_tool(
            name="get_human_perceptions",
            description=(
                "Run the HR-MCP fusion + propensity classifier over a "
                "visual/text embedding pair, returning AIGC-likelihood, "
                "belief, dissemination propensity, and composite "
                "trustworthiness/impact metrics with categorical labels."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "visual_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "512-D visual embedding vector.",
                    },
                    "text_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "512-D text embedding vector.",
                    },
                },
                "required": ["visual_embedding", "text_embedding"],
            },
            handler=self.get_human_perceptions,
        )

    async def verify_media_claims(
        self,
        image_tensor_shape: List[int],
        claim: str,
        image_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        MCP tool: `verify_media_claims`.

        NOTE ON THE STUBBED IMAGE INPUT: a real deployment would receive
        actual pixel data (e.g. base64-encoded PNG bytes) as part of the
        MCP call and decode it into a tensor. To keep this server runnable
        without any image-codec dependency and fully deterministic for
        testing, this stub synthesizes a random tensor of the requested
        shape via `torch.randn` -- clearly marked below.

        TODO-EXTENSION-MARKER [IMAGE_DECODING]:
        Replace the `torch.randn(...)` line with real image decoding, e.g.
        `torchvision.io.decode_image(torch.frombuffer(image_bytes, dtype=torch.uint8))`
        followed by the appropriate resize/normalize transforms matching
        `config.vision_model.image_size`.

        This method offloads the CPU/GPU-bound forward pass to a thread
        via `asyncio.to_thread` so the async event loop is never blocked
        by synchronous PyTorch computation -- essential for an MCP server
        that must remain responsive to other concurrent tool calls.
        """
        # Run the (synchronous, CPU/GPU-bound) model forward pass off the
        # event loop thread, keeping this coroutine itself non-blocking.
        def _run_verifier() -> VerifierMetrics | None:
            probe_image = torch.randn(*image_tensor_shape)  # STUB input, see docstring.
            with torch.no_grad():
                logits = self.verifier_model(probe_image)
            # No ground-truth labels are available for a live inference
            # call, so we report the raw softmax score rather than metrics
            # that require targets (Accuracy/F1/ROC-AUC apply to *offline
            # evaluation*, not single-sample online inference).
            probs = torch.softmax(logits, dim=-1)
            return probs

        probs = await asyncio.to_thread(_run_verifier)
        synthetic_prob = probs[:, 1].mean().item()

        agent_prediction: ExplainedPrediction = await asyncio.to_thread(
            self.agent_fact.verify_claim, claim, image_ref
        )

        return {
            "synthetic_probability": synthetic_prob,
            "real_probability": 1.0 - synthetic_prob,
            "agent_verdict": agent_prediction.model_dump(mode="json"),
        }

    async def get_human_perceptions(
        self, visual_embedding: List[float], text_embedding: List[float]
    ) -> Dict[str, Any]:
        """
        MCP tool: `get_human_perceptions`.

        Offloads the HR-MCP forward pass + composite-metric computation to
        a worker thread for the same non-blocking-event-loop reasons as
        `verify_media_claims`.
        """

        def _run_hr_mcp() -> CompositeMetrics:
            visual_tensor = torch.tensor(visual_embedding, dtype=torch.float32).unsqueeze(0)
            text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0)
            return self.propensity_classifier.classify(visual_tensor, text_tensor)

        composite_metrics = await asyncio.to_thread(_run_hr_mcp)
        return composite_metrics.as_dict()

    # -------------------------------------------------------------------
    # Server lifecycle (stub transport)
    # -------------------------------------------------------------------

    async def run_stdio_stub(self) -> None:
        """
        Minimal stand-in "transport loop" for local testing without a real
        MCP client: logs the registered tool list and then idles,
        simulating a long-running server process. Real deployments should
        replace this with the actual MCP SDK transport (see the
        TODO-EXTENSION-MARKER in the module docstring).
        """
        logger.info("MCP server starting (stdio-stub transport). Tools available:")
        for tool in self.list_tools():
            logger.info("  - %s: %s", tool["name"], tool["description"])
        # In a real transport this would be `await app.run(read, write, ...)`;
        # here we just keep the coroutine alive so `main.py` can demonstrate
        # a clean async startup/shutdown lifecycle.
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("MCP server shutting down.")
            raise

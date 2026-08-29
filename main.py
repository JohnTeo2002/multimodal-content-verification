#!/usr/bin/env python3
"""
main.py
========
CLI entry point that wires together the full pipeline:

    1. Load configuration (config/default_config.yaml).
    2. Build the core `DualBranchVerifier` vision model and run it over a
       (randomly-synthesised, for this offline demo) probe image.
    3. Build the `HRMCPFusion` cognitive model and run it over
       (randomly-synthesised, for this offline demo) visual/text
       embeddings, then classify the composite propensity metrics.
    4. Run `AgentFact` over a sample textual claim to produce an auditable,
       schema-validated `ExplainedPrediction`.
    5. (Optionally) start the async MCP server exposing all of the above as
       tools, for `--serve` mode.

This script deliberately uses randomly-initialised model weights and
synthetic input tensors (no pretrained checkpoints, no real images) so that
`python main.py` runs end-to-end in any environment with just the Python
dependencies installed -- no external downloads, no GPU required, no
network access needed. It exists to demonstrate the wiring between modules,
not to produce meaningful predictions.

TODO-EXTENSION-MARKER [REAL_INPUTS_AND_WEIGHTS]:
    Swap the `torch.randn(...)` calls below for real image loading
    (`torchvision.io.read_image` + transforms) and real embeddings (from
    actual ViT/BERT encoders), and load trained checkpoints via
    `model.load_state_dict(torch.load(...))` before running inference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from agents.agent_fact import AgentFact, AgentMode
from agents.mcp_server import MCPServer
from agents.schemas import EvidenceItem
from cognitive.hr_mcp_fusion import HRMCPFusion
from cognitive.propensity import PropensityClassifier
from core.verifier_model import DualBranchVerifier

logger = logging.getLogger("multimodal_verifier")

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default_config.yaml"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> torch.device:
    """
    Resolve the requested device string to an actually-available
    `torch.device`, gracefully falling back to CPU if e.g. "cuda" was
    requested but no GPU is present -- this keeps `main.py` runnable in any
    environment (including CI) regardless of the config file's contents.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and not getattr(torch.backends, "mps", None):
        logger.warning("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def run_vision_pipeline(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Instantiate DualBranchVerifier from config and run one probe forward pass."""
    vm_cfg = config["vision_model"]
    model = DualBranchVerifier(
        in_channels=vm_cfg["in_channels"],
        branch_output_dim=vm_cfg["fusion"]["branch_output_dim"],
        fusion_strategy=vm_cfg["fusion"]["fusion_strategy"],
        weighted_sum_alpha=vm_cfg["fusion"]["weighted_sum_alpha"],
        classifier_hidden_dim=vm_cfg["classifier"]["hidden_dim"],
        num_classes=vm_cfg["classifier"]["num_classes"],
        dropout=vm_cfg["classifier"]["dropout"],
    ).to(device)
    model.eval()

    image_size = vm_cfg["image_size"]
    probe_image = torch.randn(1, vm_cfg["in_channels"], image_size, image_size, device=device)

    with torch.no_grad():
        logits = model(probe_image)
        probs = torch.softmax(logits, dim=-1)

    return {
        "real_probability": probs[0, 0].item(),
        "synthetic_probability": probs[0, 1].item(),
    }


def run_cognitive_pipeline(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Instantiate HRMCPFusion from config and classify a probe embedding pair."""
    hr_cfg = config["hr_mcp"]
    model = HRMCPFusion(
        visual_embed_dim=hr_cfg["visual_embed_dim"],
        text_embed_dim=hr_cfg["text_embed_dim"],
        sentiment_hidden_dim=hr_cfg["sentiment_mlp"]["hidden_dim"],
        sentiment_dropout=hr_cfg["sentiment_mlp"]["dropout"],
        propensity_hidden_dims=tuple(hr_cfg["propensity_heads"]["hidden_dims"]),
        propensity_dropout=hr_cfg["propensity_heads"]["dropout"],
    ).to(device)
    model.eval()

    classifier = PropensityClassifier(
        model,
        clamp_min=hr_cfg["composite_metrics"]["clamp_min"],
        clamp_max=hr_cfg["composite_metrics"]["clamp_max"],
    )

    probe_visual = torch.randn(1, hr_cfg["visual_embed_dim"], device=device)
    probe_text = torch.randn(1, hr_cfg["text_embed_dim"], device=device)

    composite_metrics = classifier.classify(probe_visual, probe_text)
    return composite_metrics.as_dict()


def run_agentic_pipeline(config: Dict[str, Any], claim: str) -> Dict[str, Any]:
    """Instantiate AgentFact from config and verify a sample textual claim."""
    agent_cfg = config["agents"]
    agent = AgentFact(
        mode=AgentMode(agent_cfg["mode"]),
        max_react_steps=agent_cfg["max_react_steps"],
        confidence_threshold=agent_cfg["confidence_threshold"],
        preloaded_evidence=[
            EvidenceItem(
                evidence_id="ev_seed_0",
                snippet=f"Contextual background information related to: {claim}",
                source_url=None,
                retrieved_at_step=0,
            )
        ],
    )
    prediction = agent.verify_claim(claim)
    return prediction.model_dump(mode="json")


async def serve_mcp() -> None:
    """Start the (stub-transport) async MCP server and run until interrupted."""
    server = MCPServer()
    logger.info("Starting MCP server with tools: %s", [t["name"] for t in server.list_tools()])
    await server.run_stdio_stub()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multimodal AI Verification System -- CLI pipeline runner."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML config file (defaults to config/default_config.yaml).",
    )
    parser.add_argument(
        "--claim", type=str,
        default="This image shows a real, unedited photograph taken yesterday.",
        help="Textual claim to run through the AgentFact verification pipeline.",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start the async MCP server instead of running the one-shot demo pipeline.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    logging.basicConfig(level=config["logging"]["level"])

    # Seed everything for reproducible demo output.
    torch.manual_seed(config["project"]["seed"])

    if args.serve:
        try:
            asyncio.run(serve_mcp())
        except KeyboardInterrupt:
            logger.info("MCP server stopped by user.")
        return

    device = resolve_device(config["project"]["device"])
    logger.info("Using device: %s", device)

    logger.info("--- Running vision verification pipeline ---")
    vision_result = run_vision_pipeline(config, device)
    print(json.dumps({"vision_verification": vision_result}, indent=2))

    logger.info("--- Running cognitive (HR-MCP) pipeline ---")
    cognitive_result = run_cognitive_pipeline(config, device)
    print(json.dumps({"human_response_propensity": cognitive_result}, indent=2))

    logger.info("--- Running agentic (AgentFact) pipeline ---")
    agentic_result = run_agentic_pipeline(config, args.claim)
    print(json.dumps({"agent_fact_verdict": agentic_result}, indent=2))


if __name__ == "__main__":
    main()

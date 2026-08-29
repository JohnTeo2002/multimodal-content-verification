"""
core
====
Low-level neural network building blocks for the Multimodal AI Verification
System: custom activations, mocked vision backbones (CoAtNet / PVTv2),
directional cross-modal attention, and the dual-branch verifier model.

Nothing in this package talks to the network, disk (beyond weights), or any
external API -- it is pure `torch.nn.Module` code, which is what makes it
unit-testable and safe to run in air-gapped CI environments.
"""

from .activations import CustomGELU, CustomELU
from .backbones import CoAtNetBackbone, PVTv2Backbone
from .cross_attention import DirectionalCrossAttention
from .verifier_model import DualBranchVerifier

__all__ = [
    "CustomGELU",
    "CustomELU",
    "CoAtNetBackbone",
    "PVTv2Backbone",
    "DirectionalCrossAttention",
    "DualBranchVerifier",
]

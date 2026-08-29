"""
cognitive
=========
Human-response modelling layer: given aligned visual + textual embeddings
for a piece of media, predicts how a human audience is likely to *perceive*
and *respond to* that media (independent of whether it is technically
AI-generated). This complements `core/` (which answers "is this synthetic?")
with "how will people react to this, and how much should we trust that
reaction?".
"""

from .hr_mcp_fusion import HRMCPFusion, HRMCPOutput
from .propensity import PropensityClassifier, CompositeMetrics, classify_propensity

__all__ = [
    "HRMCPFusion",
    "HRMCPOutput",
    "PropensityClassifier",
    "CompositeMetrics",
    "classify_propensity",
]

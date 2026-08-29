"""
cognitive.hr_mcp_fusion
========================
Human Response - Model Context Protocol (HR-MCP) fusion pipeline.

GOAL
    Given a piece of media already summarised as a 512-D visual embedding
    (e.g. from a ViT image encoder) and a 512-D text embedding (e.g. from a
    BERT-style encoder over a caption, headline, or surrounding context),
    predict three raw human-response propensities:
        1. AIGC Likelihood       -- how likely a human perceives the media
                                     as AI-generated (independent of ground
                                     truth -- this models *perception*, not
                                     the detector's own verdict from
                                     `core/verifier_model.py`).
        2. Belief (Veracity)     -- how likely a human is to believe the
                                     claim/media is authentic/true.
        3. Dissemination         -- how likely a human is to share/spread
                                     the media further.

PIPELINE (mirrors the module specification)
    visual_embedding (512-D)  --\\
                                  >-- concat --> joint (1024-D) [Semantics]
    text_embedding   (512-D)  --/
                                        |
                                        v
                        Sentiment MLP: 1024 -> 2048 -> 1024
                        (ReLU, Dropout p=0.5) -- captures emotional /
                        affective alignment between what is depicted and
                        what is said about it.
                                        |
                                        v
        Embedding Fusion: joint_semantics + sentiment_repr, then LayerNorm
        --> enhanced_joint_embedding (1024-D)
                                        |
              +-------------------------+-------------------------+
              |                         |                         |
              v                         v                         v
        AIGC-Likelihood Head     Belief Head            Dissemination Head
        (1024->256->128->1)      (1024->256->128->1)    (1024->256->128->1)
        ReLU + Dropout            ReLU + Dropout          ReLU + Dropout
              |                         |                         |
              v                         v                         v
           sigmoid                   sigmoid                   sigmoid
        (normalised to [0, 1])   (normalised to [0, 1])   (normalised to [0, 1])

    The three normalised scalars are then handed to
    `cognitive.propensity.classify_propensity` to compute composite metrics
    (Trustworthiness, Impact) and categorical labels.

WHY SUM (RATHER THAN CONCAT) FOR THE FUSION STEP?
    The Semantics representation and the Sentiment representation are both
    already 1024-D and live in the *same* representational space (the
    Sentiment MLP is a residual-style transformation of the Semantics
    output, not an independent embedding). Summing them is a lightweight
    residual combination -- exactly like the residual connections in
    `core.cross_attention` -- which lets the network learn "sentiment as a
    correction/adjustment on top of the base semantics" rather than forcing
    it to learn two independent 1024-D spaces that must be *concatenated
    and re-projected* to be compared. This also keeps the joint embedding
    at a fixed 1024-D width regardless of how many refinement stages are
    added later, simplifying the propensity heads that follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HRMCPOutput:
    """
    Structured output of the HR-MCP fusion network.

    Attributes:
        enhanced_joint_embedding: (batch, joint_dim) -- the fused
            semantics+sentiment representation, exposed for downstream
            explainability tooling or additional heads.
        aigc_likelihood: (batch, 1) -- P(perceived as AI-generated), in [0, 1].
        belief: (batch, 1) -- P(perceived as true/authentic), in [0, 1].
        dissemination_propensity: (batch, 1) -- P(will be shared), in [0, 1].
    """

    enhanced_joint_embedding: torch.Tensor
    aigc_likelihood: torch.Tensor
    belief: torch.Tensor
    dissemination_propensity: torch.Tensor


class _SentimentMLP(nn.Module):
    """
    3-layer MLP capturing emotional/affective alignment between the visual
    and textual modalities: 1024 -> 2048 -> 1024, ReLU activations, dropout
    p=0.5 (aggressive dropout is intentional here -- sentiment/affect
    signals are noisier and more subjective than raw semantic content, so
    stronger regularisation helps prevent overfitting to spurious
    correlations in training data).

    Shape contract:
        Input:  (batch, joint_dim)
        Output: (batch, joint_dim)   -- same width, ready to sum with the
                                          semantics representation.
    """

    def __init__(self, joint_dim: int = 1024, hidden_dim: int = 2048, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, joint_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, joint_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(joint_embedding)


class _PropensityHead(nn.Module):
    """
    A single 4-layer propensity-prediction MLP: joint_dim -> 256 -> 128 -> 1,
    with ReLU activations and dropout between layers, followed by a sigmoid
    to squash the raw scalar output into a [0, 1] "probability-like"
    propensity score.

    Three independent instances of this class (with separate, untied
    weights) are used for AIGC-Likelihood, Belief, and Dissemination
    Propensity respectively -- this is what "three parallel MLPs" means:
    they share the same *architecture* but not the same *parameters*, and
    they run concurrently (all three consume the same enhanced joint
    embedding as input, independently of one another).

    Shape contract:
        Input:  (batch, joint_dim)
        Output: (batch, 1), values in [0, 1]
    """

    def __init__(self, joint_dim: int = 1024, hidden_dims: tuple[int, int] = (256, 128), dropout: float = 0.3) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(joint_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
            # Sigmoid squashes the unbounded final linear output into
            # [0, 1] so the raw score can be interpreted as a normalised
            # propensity / pseudo-probability, and so composite metrics in
            # `cognitive.propensity` (which assume [0, 1]-ranged inputs)
            # remain well-defined.
            nn.Sigmoid(),
        )

    def forward(self, joint_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(joint_embedding)


class HRMCPFusion(nn.Module):
    """
    Full Human Response - Model Context Protocol fusion network.

    Args:
        visual_embed_dim: Dimensionality of the incoming visual embedding
            (expected to already be produced by an upstream ViT-style
            encoder; that encoder itself is out of scope for this module).
        text_embed_dim: Dimensionality of the incoming text embedding
            (expected from an upstream BERT-style encoder).
        sentiment_hidden_dim: Hidden width of the Sentiment MLP.
        sentiment_dropout: Dropout probability inside the Sentiment MLP.
        propensity_hidden_dims: (h1, h2) hidden widths for each propensity head.
        propensity_dropout: Dropout probability inside each propensity head.

    Shape contract:
        visual_embedding: (batch, visual_embed_dim)
        text_embedding:   (batch, text_embed_dim)
        -> HRMCPOutput with joint_dim = visual_embed_dim + text_embed_dim
    """

    def __init__(
        self,
        visual_embed_dim: int = 512,
        text_embed_dim: int = 512,
        sentiment_hidden_dim: int = 2048,
        sentiment_dropout: float = 0.5,
        propensity_hidden_dims: tuple[int, int] = (256, 128),
        propensity_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.visual_embed_dim = visual_embed_dim
        self.text_embed_dim = text_embed_dim
        self.joint_dim = visual_embed_dim + text_embed_dim

        # --- Semantics: simple concatenation of the two modalities --------
        # (No learnable parameters here -- concatenation itself carries no
        # weights; the "encoding" work was already done upstream by the ViT
        # / BERT encoders. This module's job starts at fusion.)

        # --- Sentiment: 3-layer MLP over the concatenated joint space ------
        self.sentiment_mlp = _SentimentMLP(
            joint_dim=self.joint_dim,
            hidden_dim=sentiment_hidden_dim,
            dropout=sentiment_dropout,
        )

        # --- Embedding Fusion: sum + LayerNorm ------------------------------
        self.fusion_norm = nn.LayerNorm(self.joint_dim)

        # --- Three parallel propensity heads --------------------------------
        self.aigc_head = _PropensityHead(
            joint_dim=self.joint_dim, hidden_dims=propensity_hidden_dims, dropout=propensity_dropout
        )
        self.belief_head = _PropensityHead(
            joint_dim=self.joint_dim, hidden_dims=propensity_hidden_dims, dropout=propensity_dropout
        )
        self.dissemination_head = _PropensityHead(
            joint_dim=self.joint_dim, hidden_dims=propensity_hidden_dims, dropout=propensity_dropout
        )

    def forward(
        self, visual_embedding: torch.Tensor, text_embedding: torch.Tensor
    ) -> HRMCPOutput:
        """
        Args:
            visual_embedding: (batch, visual_embed_dim)
            text_embedding: (batch, text_embed_dim)

        Returns:
            HRMCPOutput populated with the enhanced joint embedding and the
            three normalised propensity scores.
        """
        if visual_embedding.shape[-1] != self.visual_embed_dim:
            raise ValueError(
                f"Expected visual_embedding last dim {self.visual_embed_dim}, "
                f"got {visual_embedding.shape[-1]}"
            )
        if text_embedding.shape[-1] != self.text_embed_dim:
            raise ValueError(
                f"Expected text_embedding last dim {self.text_embed_dim}, "
                f"got {text_embedding.shape[-1]}"
            )

        # Step 1 (Semantics Encoder): concatenate visual + text embeddings
        # into a single joint representation.
        joint_semantics = torch.cat(
            [visual_embedding, text_embedding], dim=-1
        )  # -> (batch, joint_dim)

        # Step 2 (Sentiment Module): pass the joint representation through
        # the 3-layer sentiment MLP to capture emotional/affective signal.
        sentiment_repr = self.sentiment_mlp(joint_semantics)  # -> (batch, joint_dim)

        # Step 3 (Embedding Fusion): residual-style sum of the two
        # representations, followed by LayerNorm for training stability.
        fused = joint_semantics + sentiment_repr  # -> (batch, joint_dim)
        enhanced_joint_embedding = self.fusion_norm(fused)  # -> (batch, joint_dim)

        # Step 4 (Propensity Modules): three independent MLPs consume the
        # SAME enhanced joint embedding, each predicting a different
        # human-response propensity.
        aigc_likelihood = self.aigc_head(enhanced_joint_embedding)          # (batch, 1)
        belief = self.belief_head(enhanced_joint_embedding)                  # (batch, 1)
        dissemination_propensity = self.dissemination_head(enhanced_joint_embedding)  # (batch, 1)

        return HRMCPOutput(
            enhanced_joint_embedding=enhanced_joint_embedding,
            aigc_likelihood=aigc_likelihood,
            belief=belief,
            dissemination_propensity=dissemination_propensity,
        )

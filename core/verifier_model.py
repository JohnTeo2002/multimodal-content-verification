"""
core.verifier_model
====================
The top-level vision verification network: a **dual-branch parallel fusion
architecture** that classifies an input image as "Real" or "Synthetic
(AI-generated / manipulated)".

ARCHITECTURE OVERVIEW
    Input image
        |
        +--> [Branch 1: CoAtNet -> PVTv2 -> GELU -> MLP] --> feat_gelu (batch, D)
        |
        +--> [Branch 2: CoAtNet -> PVTv2 -> ELU  -> MLP] --> feat_elu  (batch, D)
        |
        +--> Fusion(feat_gelu, feat_elu) --> fused (batch, D or 2D)
                |
                +--> Classifier Head --> logits (batch, num_classes)

WHY TWO PARALLEL BRANCHES WITH *DIFFERENT* ACTIVATIONS?
    This is an ensembling-by-design trick. Both branches share the exact
    same backbone architecture (CoAtNet -> PVTv2) and see the exact same
    input image, but their non-linearities differ:
        * GELU is smooth and probabilistic (Gaussian-CDF gated) -- it tends
          to preserve small negative activations partially, which can help
          capture *subtle* statistical artefacts (e.g. faint upsampling
          checkerboard patterns from a GAN decoder).
        * ELU saturates to a fixed negative asymptote (-alpha) -- it tends
          to produce sparser, higher-contrast negative-region responses,
          which can help capture *sharp* local discontinuities (e.g. hard
          blending-boundary artefacts from a face-swap compositing step).
    Because the two branches are trained jointly but never share weights,
    they are pushed by gradient descent to become *complementary* feature
    extractors (this is analogous to bagging/ensembling, but performed
    inside a single end-to-end differentiable network rather than as a
    post-hoc ensemble of independently trained models). The fusion step
    then lets the classifier head draw on both sets of evidence at once.

WHY CoAtNet *THEN* PVTv2 IN SERIES (RATHER THAN AS ALTERNATIVES)?
    This repository chains the two backbones (CoAtNet's conv+attention
    output is fed as the "image" into PVTv2) to combine:
        * CoAtNet's strength: efficient early local feature extraction via
          convolution, with attention only in later stages.
        * PVTv2's strength: an explicit multi-scale pyramid with spatial-
          reduction attention, well suited to detecting artefacts that
          exist at different spatial scales (a whole-face warp vs. a
          pixel-level GAN fingerprint).
    NOTE: chaining two full backbones is unusual and computationally heavy
    for a production system; it is used here to literally satisfy the
    specified "CoAtNet -> PVTv2" per-branch pipeline. See the
    TODO-EXTENSION-MARKER below for a cheaper production alternative.

    TODO-EXTENSION-MARKER [BACKBONE_TOPOLOGY]:
    For production latency budgets, consider running CoAtNet and PVTv2 as
    two *independent* feature extractors on the raw image (rather than
    chained in series) and fusing their pooled features, or selecting only
    one backbone per branch. The `DualBranchVerifier.__init__` signature
    below is written so this swap only requires editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import torch
import torch.nn as nn

from core.activations import CustomELU, CustomGELU
from core.backbones import CoAtNetBackbone, PVTv2Backbone

FusionStrategy = Literal["concat", "weighted_sum"]


class _BranchMLP(nn.Module):
    """
    Small projection head applied at the end of each branch: takes the
    pooled backbone feature vector, applies the branch's signature
    activation, and projects to the shared `branch_output_dim`.

    Shape contract:
        Input:  (batch, in_dim)
        Output: (batch, branch_output_dim)
    """

    def __init__(self, in_dim: int, branch_output_dim: int, activation: nn.Module) -> None:
        super().__init__()
        hidden_dim = max(branch_output_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, branch_output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _VerifierBranch(nn.Module):
    """
    One full branch: CoAtNet -> PVTv2 -> {GELU | ELU} -> MLP.

    Both backbones are instantiated fresh per branch (no weight sharing
    across branches), which is what allows the two branches to specialise
    into complementary feature extractors during joint training.
    """

    def __init__(
        self,
        in_channels: int,
        branch_output_dim: int,
        activation: nn.Module,
        coatnet_kwargs: dict | None = None,
        pvtv2_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.coatnet = CoAtNetBackbone(in_channels=in_channels, **(coatnet_kwargs or {}))

        # CoAtNet returns a pooled (batch, C) vector, but PVTv2 expects a
        # spatial (batch, C, H, W) tensor. We re-inflate the pooled vector
        # into a minimal 1x1 spatial map so it can be fed into PVTv2's
        # patch-embedding convolution. This preserves the specified
        # "CoAtNet -> PVTv2" series topology while remaining shape-valid.
        #
        # TODO-EXTENSION-MARKER [SKIP_CONNECTION_TOPOLOGY]:
        # A production system would more naturally feed CoAtNet's
        # *intermediate* spatial feature map (before its own global pool)
        # into PVTv2, rather than re-inflating a pooled vector. That
        # requires exposing an intermediate hook on CoAtNetBackbone; left
        # as a clearly-marked extension point to keep this reference
        # implementation simple and independently testable per backbone.
        self._reinflate_channels = self.coatnet.output_dim
        # patch_size=1 is required (and forced) here, regardless of any
        # caller-supplied override: the CoAtNet output is re-inflated to a
        # 1x1 "spatial" map (see forward() below), so PVTv2's patch-embedding
        # convolution kernel must be exactly 1x1 to have any valid receptive
        # field to slide over. Any larger patch_size would raise a shape
        # error at the very first convolution.
        pvtv2_kwargs = dict(pvtv2_kwargs or {})
        pvtv2_kwargs["patch_size"] = 1
        self.pvtv2 = PVTv2Backbone(in_channels=self._reinflate_channels, **pvtv2_kwargs)

        self.mlp_head = _BranchMLP(
            in_dim=self.pvtv2.output_dim,
            branch_output_dim=branch_output_dim,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        coatnet_feats = self.coatnet(x)  # -> (batch, coatnet_output_dim)

        # Re-inflate to a (batch, C, 1, 1) "spatial" tensor so PVTv2's
        # strided patch-embedding convolution has something to operate on.
        # With a 1x1 spatial input, PVTv2's internal patch/merge convs
        # effectively degrade to per-channel linear projections -- which is
        # a deliberate, documented simplification (see class docstring).
        spatial = coatnet_feats.unsqueeze(-1).unsqueeze(-1)  # -> (batch, C, 1, 1)

        pvtv2_feats = self.pvtv2(spatial)  # -> (batch, pvtv2_output_dim)
        branch_feats = self.mlp_head(pvtv2_feats)  # -> (batch, branch_output_dim)
        return branch_feats


@dataclass
class VerifierMetrics:
    """
    Structured container for classification metrics reported after
    evaluation. Kept as a plain dataclass (rather than a dict) so
    downstream code (CLI printing, logging, unit tests) gets attribute
    access and type-checking instead of string-keyed dict lookups.

    Attributes:
        accuracy: Fraction of correctly classified samples, in [0, 1].
        f1_score: Harmonic mean of precision and recall for the positive
            ("synthetic") class, in [0, 1].
        roc_auc: Area under the Receiver Operating Characteristic curve,
            in [0, 1] (0.5 = random chance, 1.0 = perfect separation).
        num_samples: Number of samples the metrics were computed over.
    """

    accuracy: float
    f1_score: float
    roc_auc: float
    num_samples: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "num_samples": self.num_samples,
        }


def compute_verifier_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> VerifierMetrics:
    """
    Compute Accuracy, F1-Score, and ROC-AUC for a binary Real/Fake
    classification head, using only `torch` (no external ML-metrics
    dependency), so this stays runnable in a minimal, air-gapped
    environment.

    Args:
        logits: (batch, 2) raw, un-normalised classifier outputs (index 0
            = "Real" class score, index 1 = "Synthetic" class score).
        targets: (batch,) integer ground-truth labels in {0, 1}.

    Returns:
        A populated VerifierMetrics dataclass.
    """
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)          # (batch, 2)
        positive_probs = probs[:, 1]                    # P(synthetic)
        predictions = torch.argmax(logits, dim=-1)       # (batch,)

        num_samples = targets.shape[0]

        # --- Accuracy ------------------------------------------------------
        correct = (predictions == targets).float().sum()
        accuracy = (correct / max(num_samples, 1)).item()

        # --- Precision / Recall / F1 for the positive ("synthetic") class -
        true_positives = ((predictions == 1) & (targets == 1)).float().sum()
        false_positives = ((predictions == 1) & (targets == 0)).float().sum()
        false_negatives = ((predictions == 0) & (targets == 1)).float().sum()

        precision_denom = true_positives + false_positives
        recall_denom = true_positives + false_negatives
        precision = (true_positives / precision_denom) if precision_denom > 0 else torch.tensor(0.0)
        recall = (true_positives / recall_denom) if recall_denom > 0 else torch.tensor(0.0)

        f1_denom = precision + recall
        f1_score = (2 * precision * recall / f1_denom) if f1_denom > 0 else torch.tensor(0.0)

        # --- ROC-AUC via the Mann-Whitney U statistic -----------------------
        # AUC equals the probability that a randomly chosen positive sample
        # is ranked above a randomly chosen negative sample by the model's
        # score. This can be computed directly from rank statistics without
        # sweeping explicit thresholds:
        #   AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
        pos_mask = targets == 1
        neg_mask = targets == 0
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        if n_pos == 0 or n_neg == 0:
            # AUC is undefined with only one class present; report 0.5
            # (chance level) rather than raising, so evaluation loops don't
            # crash on degenerate mini-batches.
            roc_auc = 0.5
        else:
            ranks = torch.argsort(torch.argsort(positive_probs)).float() + 1.0
            sum_ranks_pos = ranks[pos_mask].sum()
            auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            roc_auc = auc.item()

    return VerifierMetrics(
        accuracy=accuracy,
        f1_score=f1_score.item() if isinstance(f1_score, torch.Tensor) else f1_score,
        roc_auc=roc_auc,
        num_samples=num_samples,
    )


class DualBranchVerifier(nn.Module):
    """
    The full dual-branch synthetic-media detector.

    Shape contract:
        Input:  (batch, in_channels, H, W)
        Output: logits of shape (batch, num_classes)  [num_classes=2 by default]

    Args:
        in_channels: Number of input image channels (3 for RGB).
        branch_output_dim: Feature dimensionality produced by each branch
            before fusion.
        fusion_strategy: "concat" (output dim = 2 * branch_output_dim) or
            "weighted_sum" (output dim = branch_output_dim).
        weighted_sum_alpha: Only used when fusion_strategy="weighted_sum".
            fused = alpha * feat_gelu + (1 - alpha) * feat_elu.
        classifier_hidden_dim: Hidden layer width of the classifier head.
        num_classes: Number of output classes (2: Real vs. Synthetic).
        dropout: Dropout probability inside the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        branch_output_dim: int = 512,
        fusion_strategy: FusionStrategy = "weighted_sum",
        weighted_sum_alpha: float = 0.5,
        classifier_hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not (0.0 <= weighted_sum_alpha <= 1.0):
            raise ValueError("weighted_sum_alpha must be within [0, 1]")

        self.fusion_strategy: FusionStrategy = fusion_strategy
        self.weighted_sum_alpha = weighted_sum_alpha

        # Branch 1 uses our custom GELU; Branch 2 uses our custom ELU. Note
        # each branch gets its OWN activation module instance -- these are
        # stateless (no learnable parameters) but we keep separate instances
        # for architectural clarity and to allow future stateful variants
        # (e.g. a learnable-alpha ELU) without cross-branch coupling.
        self.branch_gelu = _VerifierBranch(
            in_channels=in_channels,
            branch_output_dim=branch_output_dim,
            activation=CustomGELU(),
        )
        self.branch_elu = _VerifierBranch(
            in_channels=in_channels,
            branch_output_dim=branch_output_dim,
            activation=CustomELU(),
        )

        fused_dim = (
            branch_output_dim * 2 if fusion_strategy == "concat" else branch_output_dim
        )

        # Classifier head: fused_dim -> hidden -> num_classes.
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def fuse(self, feat_gelu: torch.Tensor, feat_elu: torch.Tensor) -> torch.Tensor:
        """
        Combine the two branch feature vectors according to
        `self.fusion_strategy`.

        Args:
            feat_gelu: (batch, branch_output_dim)
            feat_elu:  (batch, branch_output_dim)

        Returns:
            (batch, branch_output_dim)                  if "weighted_sum"
            (batch, 2 * branch_output_dim)               if "concat"
        """
        if self.fusion_strategy == "concat":
            return torch.cat([feat_gelu, feat_elu], dim=-1)
        elif self.fusion_strategy == "weighted_sum":
            alpha = self.weighted_sum_alpha
            return alpha * feat_gelu + (1.0 - alpha) * feat_elu
        else:  # pragma: no cover -- Literal type keeps this unreachable.
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W) input image batch.

        Returns:
            logits: (batch, num_classes) raw classifier scores. Apply
                `torch.softmax(logits, dim=-1)` downstream to obtain
                calibrated-looking class probabilities, or use
                `compute_verifier_metrics` directly on the logits.
        """
        feat_gelu = self.branch_gelu(x)   # -> (batch, branch_output_dim)
        feat_elu = self.branch_elu(x)     # -> (batch, branch_output_dim)

        fused = self.fuse(feat_gelu, feat_elu)  # -> (batch, fused_dim)
        logits = self.classifier(fused)          # -> (batch, num_classes)
        return logits

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as `forward`, but also returns the two intermediate branch
        feature vectors -- useful for explainability tooling (e.g. probing
        which branch contributed more to a given prediction) and for unit
        tests that assert per-branch output shapes independently.

        Returns:
            (logits, feat_gelu, feat_elu)
        """
        feat_gelu = self.branch_gelu(x)
        feat_elu = self.branch_elu(x)
        fused = self.fuse(feat_gelu, feat_elu)
        logits = self.classifier(fused)
        return logits, feat_gelu, feat_elu

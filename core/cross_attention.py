"""
core.cross_attention
=====================
Directional multi-head cross-modal attention.

WHAT IS "DIRECTIONAL" CROSS-ATTENTION?
    In *self*-attention, a single sequence attends to itself: Query, Key,
    and Value are all projections of the same input.

    In *cross*-attention, two different sequences (here: two different
    modalities, e.g. image tokens and text tokens) interact -- but we must
    choose which modality asks the questions (Query) and which modality
    supplies the answers (Key/Value). This choice is the "direction" of the
    attention:
        * direction = "A_to_B" means modality A's tokens are projected to
          Queries, and modality B's tokens are projected to Keys and
          Values. The output has the *same sequence length as A*, because
          every element of A produces exactly one attention-weighted
          output vector.

MATHEMATICAL FORMULATION
    Given Query-source tensor Q_src (shape: batch, seq_A, embed_dim) and
    Key/Value-source tensor KV_src (shape: batch, seq_B, embed_dim):

        Q = Q_src @ W_Q                      (batch, seq_A, embed_dim)
        K = KV_src @ W_K                     (batch, seq_B, embed_dim)
        V = KV_src @ W_V                     (batch, seq_B, embed_dim)

        AttentionScores = (Q @ K^T) / sqrt(d_k)     (batch, seq_A, seq_B)
        AttentionWeights = softmax(AttentionScores, dim=-1)
        AttentionOutput = AttentionWeights @ V       (batch, seq_A, embed_dim)

    where d_k = embed_dim / num_heads is the per-head dimensionality (the
    sqrt(d_k) division is the "scaled" part of "scaled dot-product
    attention" -- it prevents the softmax input from growing too large in
    magnitude as d_k increases, which would otherwise push the softmax into
    a saturated, near-one-hot regime with vanishing gradients).

    Finally, a residual connection adds this attention output back onto the
    *original* Query-source tensor (not a transformed version of it), and a
    LayerNorm stabilises the resulting distribution:

        Output = LayerNorm(Q_src + Dropout(AttentionOutput))

WHY THE RESIDUAL CONNECTION MATTERS (GRADIENT DEGRADATION)
    Stacking many attention/MLP layers without residual connections causes
    the gradient signal computed at the loss to shrink multiplicatively as
    it is backpropagated through each layer (vanishing gradients), because
    each layer's Jacobian gets multiplied in. A residual connection
    `y = x + f(x)` guarantees that the identity mapping is always a easy-to-
    learn special case (dy/dx includes a direct `+1` term from the skip
    connection, independent of f's Jacobian), so gradients can flow
    directly backward through the `+ x` path even if `f`'s gradient
    saturates. This is precisely why deep Transformer stacks (and deep
    CoAtNet/PVTv2-style hybrids) remain trainable at depth.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

Direction = Literal["image_to_text", "text_to_image"]


class DirectionalCrossAttention(nn.Module):
    """
    Multi-head cross-attention with a configurable Query/Key-Value
    direction, a residual skip connection to the Query-source modality, and
    a trailing LayerNorm.

    Shape contract:
        image_tokens: (batch, seq_img, embed_dim)
        text_tokens:  (batch, seq_txt, embed_dim)

        If direction == "image_to_text":
            Query source = image_tokens, Key/Value source = text_tokens
            Output shape = (batch, seq_img, embed_dim)   <- same as image_tokens

        If direction == "text_to_image":
            Query source = text_tokens, Key/Value source = image_tokens
            Output shape = (batch, seq_txt, embed_dim)   <- same as text_tokens

    Args:
        embed_dim: Shared embedding dimensionality of both modalities. Both
            modalities MUST already be projected into this common space
            upstream (e.g. by a linear projection after a ViT/BERT encoder)
            for cross-attention to be mathematically meaningful.
        num_heads: Number of parallel attention heads. embed_dim must be
            divisible by num_heads (this is enforced by
            `nn.MultiheadAttention` internally).
        dropout: Dropout probability applied to attention weights and to
            the attention output before the residual addition.
        direction: Which modality supplies the Query vs. the Key/Value.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        direction: Direction = "image_to_text",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.direction: Direction = direction
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # `nn.MultiheadAttention` internally owns the W_Q, W_K, W_V, and
        # output projection matrices, and performs the scaled dot-product
        # attention formula described in the module docstring across
        # `num_heads` parallel attention "sub-spaces". We use
        # `batch_first=True` so all our tensors follow the (batch, seq,
        # feature) convention used throughout this repository.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Dropout applied to the raw attention output before it is added
        # back to the residual stream -- a standard Transformer
        # regularisation trick ("post-attention dropout").
        self.output_dropout = nn.Dropout(dropout)

        # LayerNorm normalises each token's feature vector to zero mean /
        # unit variance (learned affine parameters gamma, beta then rescale
        # it), which keeps activation magnitudes stable across many
        # stacked cross-attention blocks.
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tokens: (batch, seq_img, embed_dim)
            text_tokens:  (batch, seq_txt, embed_dim)
            attn_mask: Optional attention mask forwarded to
                `nn.MultiheadAttention` (e.g. to mask out padding tokens in
                the Key/Value sequence).

        Returns:
            Tuple of:
                output: The residual-and-normed cross-attended tensor, with
                    the same shape as whichever modality supplied the Query
                    (see class docstring for the exact shape per direction).
                attn_weights: (batch, seq_Q, seq_KV) attention weight
                    matrix, useful for visualisation/explainability (e.g.
                    "which words did the model look at when judging this
                    image region as synthetic?").
        """
        if self.direction == "image_to_text":
            query_source = image_tokens   # (batch, seq_img, embed_dim)
            kv_source = text_tokens       # (batch, seq_txt, embed_dim)
        elif self.direction == "text_to_image":
            query_source = text_tokens    # (batch, seq_txt, embed_dim)
            kv_source = image_tokens      # (batch, seq_img, embed_dim)
        else:  # pragma: no cover -- Literal type keeps this unreachable.
            raise ValueError(f"Unknown direction: {self.direction!r}")

        # Step 1: Scaled dot-product multi-head attention.
        #   attn_output shape: (batch, seq_Q, embed_dim) -- same seq length
        #   as query_source, because every query token gets exactly one
        #   output vector (a weighted blend of the *value* vectors from the
        #   *other* modality).
        #   attn_weights shape: (batch, seq_Q, seq_KV) -- averaged across
        #   heads by default, showing how much each query token attended to
        #   each key/value token.
        attn_output, attn_weights = self.attention(
            query=query_source,
            key=kv_source,
            value=kv_source,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # Step 2: Residual connection back to the *original* query-source
        # tensor (not to any intermediate projection of it), so the model
        # can always fall back to "ignore the other modality entirely" as a
        # trivial, easy-to-learn solution if cross-modal evidence is
        # uninformative for a given example.
        attn_output = self.output_dropout(attn_output)
        residual_sum = query_source + attn_output  # (batch, seq_Q, embed_dim)

        # Step 3: LayerNorm stabilises the combined representation before
        # it is passed to downstream layers.
        output = self.layer_norm(residual_sum)  # (batch, seq_Q, embed_dim)

        return output, attn_weights

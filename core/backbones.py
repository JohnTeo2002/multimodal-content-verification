"""
core.backbones
==============
Mocked (lightweight, architecturally-faithful-but-simplified) implementations
of two well-known vision backbones:

    * CoAtNet  -- "Convolution + Attention Network" (Dai et al., 2021).
                  Interleaves convolutional MBConv stages (good local
                  inductive bias, cheap) with Transformer self-attention
                  stages (good global receptive field) in a single network.
    * PVTv2    -- "Pyramid Vision Transformer v2" (Wang et al., 2022).
                  A hierarchical, pyramid-shaped pure-transformer backbone
                  that progressively down-samples spatial resolution while
                  increasing channel width, similar to a CNN feature
                  pyramid, using Spatial-Reduction Attention (SRA) to keep
                  attention affordable at high resolution.

WHY "MOCKED"?
    Real CoAtNet / PVTv2 implementations (as in the `timm` library) are
    thousands of lines long and depend on pretrained weight checkpoints
    that are not available in an air-gapped test environment. To keep this
    repository:
        (a) fully runnable offline / in CI without internet access, and
        (b) pedagogically clear about *where the two backbones sit* in the
            larger pipeline,
    we implement architecturally-representative-but-compact stand-ins that
    preserve the key design ideas (staged channel growth, conv + attention
    mixing, spatial-reduction attention) while using a fraction of the
    parameters of the real networks.

    TODO-EXTENSION-MARKER [REAL_BACKBONE_WEIGHTS]:
    To swap in production-grade backbones, replace the bodies of
    `CoAtNetBackbone.forward` / `PVTv2Backbone.forward` with calls to a
    pretrained `timm.create_model("coatnet_2_rw_224", pretrained=True)` (or
    equivalent) and adapt `self.output_dim` to match the real backbone's
    final feature dimension. The rest of the pipeline (cross-attention,
    dual-branch fusion, cognitive layer) is agnostic to this swap because it
    only depends on the `(batch, output_dim)` contract documented below.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class _MBConvBlock(nn.Module):
    """
    A simplified Mobile Inverted Bottleneck Convolution block (the
    convolutional workhorse of CoAtNet's early stages).

    Structure: 1x1 expand -> depthwise 3x3 conv -> 1x1 project, with a
    residual connection when input/output channel counts match.
    """

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels

        self.block = nn.Sequential(
            # 1x1 pointwise "expand" convolution: grows channel capacity
            # before the more expensive depthwise conv, mirroring the
            # inverted-residual design from MobileNetV2.
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # Depthwise 3x3 convolution: captures local spatial patterns
            # (edges, textures) cheaply -- one filter per channel.
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1,
                groups=hidden_dim, bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            # 1x1 pointwise "project" convolution: compresses back down to
            # `out_channels`.
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x  # Residual skip connection stabilises gradients.
        return out


class CoAtNetBackbone(nn.Module):
    """
    Mocked CoAtNet feature extractor.

    Design: a convolutional "stem" downsamples the raw image, followed by a
    stack of MBConv stages (mirroring CoAtNet's early convolutional
    stages), then a lightweight self-attention stage over the flattened
    spatial tokens (mirroring CoAtNet's later transformer stages), and
    finally global average pooling to a single feature vector per image.

    Shape contract:
        Input:  (batch, in_channels, H, W)
        Output: (batch, output_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        stage_channels: List[int] | None = None,
        stage_depths: List[int] | None = None,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        stage_channels = stage_channels or [96, 192, 384, 768]
        stage_depths = stage_depths or [2, 2, 6, 2]
        assert len(stage_channels) == len(stage_depths), (
            "stage_channels and stage_depths must have equal length"
        )

        # --- Stem: standard strided conv to quickly reduce H x W ---------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        # --- Convolutional (MBConv) stages --------------------------------
        conv_stages: List[nn.Module] = []
        prev_channels = stem_channels
        # We treat the first half of the stages as "convolutional" (C) and
        # keep them purely conv-based, mirroring CoAtNet's "C-C-T-T" layout
        # at a small scale.
        num_conv_stages = max(1, len(stage_channels) // 2)
        for stage_idx in range(num_conv_stages):
            out_channels = stage_channels[stage_idx]
            blocks = [_MBConvBlock(prev_channels, out_channels)]
            for _ in range(stage_depths[stage_idx] - 1):
                blocks.append(_MBConvBlock(out_channels, out_channels))
            conv_stages.append(nn.Sequential(*blocks))
            # Downsample between conv stages via strided conv.
            conv_stages.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            prev_channels = out_channels
        self.conv_stages = nn.Sequential(*conv_stages)

        # --- Transformer (attention) stages --------------------------------
        # The remaining stages are treated as "T" (transformer) stages: we
        # flatten spatial tokens and run standard multi-head self-attention,
        # which gives the backbone a global receptive field -- something
        # pure convolutions struggle to achieve without very deep stacks.
        self.final_channels = stage_channels[-1]
        self.channel_proj = nn.Conv2d(prev_channels, self.final_channels, kernel_size=1)
        self.attn_norm = nn.LayerNorm(self.final_channels)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.final_channels, num_heads=attn_heads, batch_first=True
        )

        self.output_dim: int = self.final_channels
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        x = self.stem(x)               # -> (batch, stem_channels, H/2, W/2)
        x = self.conv_stages(x)        # -> (batch, C_last, H', W') after MBConv + downsampling
        x = self.channel_proj(x)       # -> (batch, final_channels, H', W')

        batch, channels, height, width = x.shape
        # Flatten spatial grid into a sequence of tokens for self-attention:
        # (batch, channels, H', W') -> (batch, H'*W', channels)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.attn_norm(tokens)
        # Standard scaled dot-product self-attention: each spatial location
        # attends to every other spatial location, giving CoAtNet its
        # global context even in early feature maps.
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = tokens + attn_out  # Residual connection around attention.

        # Reshape back to spatial grid then global-average-pool to a single
        # per-image feature vector.
        tokens = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        pooled = self.pool(tokens).flatten(1)  # -> (batch, output_dim)
        return pooled


class _SpatialReductionAttention(nn.Module):
    """
    Spatial-Reduction Attention (SRA), the key efficiency trick in PVTv2.

    Standard self-attention over an H x W feature map costs O((H*W)^2)
    because every token attends to every other token. SRA reduces the
    Key/Value sequence length by a `sr_ratio` factor using a strided
    convolution *before* computing attention, cutting the quadratic cost
    down to O(H*W * (H*W)/sr_ratio^2) while the Query sequence stays at
    full resolution -- preserving fine-grained output detail.
    """

    def __init__(self, embed_dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.sr_ratio = sr_ratio
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        if sr_ratio > 1:
            self.sr_conv = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)
        else:
            self.sr_conv = None
            self.sr_norm = None

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # x: (batch, seq_len=H*W, embed_dim) -- the Query sequence, always
        # at full resolution.
        batch, seq_len, channels = x.shape
        query = x

        # Only apply spatial reduction if the current H x W grid is at
        # least as large as the reduction kernel/stride; otherwise (e.g. a
        # degenerate 1x1 "token grid" re-inflated from another backbone's
        # pooled output) skip reduction and attend over the full-resolution
        # Key/Value sequence instead of raising a shape error.
        if self.sr_conv is not None and height >= self.sr_ratio and width >= self.sr_ratio:
            # Reshape the sequence back into a spatial grid so we can apply
            # a strided convolution that shrinks H*W by sr_ratio^2, then
            # flatten again -- this becomes our (reduced-length) Key/Value
            # source.
            spatial = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.sr_conv(spatial)  # -> (batch, C, H/sr, W/sr)
            reduced = reduced.flatten(2).transpose(1, 2)  # -> (batch, H'*W', C)
            key_value_source = self.sr_norm(reduced)
        else:
            key_value_source = x

        attn_out, _ = self.attn(query, key_value_source, key_value_source)
        return attn_out


class PVTv2Backbone(nn.Module):
    """
    Mocked Pyramid Vision Transformer v2 feature extractor.

    Design: patch-embeds the input into non-overlapping patches, then runs a
    sequence of pyramid stages, each of which (a) merges patches to reduce
    resolution and grow channel width, and (b) applies Spatial-Reduction
    Attention. Ends with global average pooling to a single feature vector.

    Shape contract:
        Input:  (batch, in_channels, H, W)
        Output: (batch, output_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dims: List[int] | None = None,
        num_heads: List[int] | None = None,
        sr_ratios: List[int] | None = None,
    ) -> None:
        super().__init__()
        embed_dims = embed_dims or [64, 128, 320, 512]
        num_heads = num_heads or [1, 2, 5, 8]
        sr_ratios = sr_ratios or [8, 4, 2, 1]
        assert len(embed_dims) == len(num_heads) == len(sr_ratios), (
            "embed_dims, num_heads, and sr_ratios must have equal length"
        )

        # --- Initial patch embedding ---------------------------------------
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dims[0], kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(embed_dims[0])

        # --- Pyramid stages: (merge -> SRA) per stage ----------------------
        self.stage_merges = nn.ModuleList()
        # `stage_merges_fallback` mirrors `stage_merges` 1:1, but uses a
        # stride-1, kernel-1 (i.e. purely channel-projecting) convolution
        # instead of a spatially-downsampling one. This exists so the
        # backbone remains numerically valid even when it is fed an
        # already-tiny spatial map (e.g. a single 1x1 "token" re-inflated
        # from another backbone's pooled output, as `core.verifier_model`
        # does when chaining CoAtNet -> PVTv2). A real H x W image input
        # will always be large enough to use the normal strided merge; the
        # fallback path only activates for degenerate spatial sizes, see
        # the size check in `forward` below.
        self.stage_merges_fallback = nn.ModuleList()
        self.stage_attns = nn.ModuleList()
        self.stage_norms = nn.ModuleList()

        prev_dim = embed_dims[0]
        for stage_idx, (dim, heads, sr) in enumerate(zip(embed_dims, num_heads, sr_ratios)):
            if stage_idx == 0:
                # First stage re-uses the initial patch embedding's channel
                # count; no merge needed before it.
                self.stage_merges.append(nn.Identity())
                self.stage_merges_fallback.append(nn.Identity())
            else:
                # Patch-merging: a strided conv that halves H/W and doubles
                # (or otherwise grows) channel depth, exactly like a CNN's
                # pooling + channel-expansion step.
                self.stage_merges.append(
                    nn.Conv2d(prev_dim, dim, kernel_size=2, stride=2)
                )
                self.stage_merges_fallback.append(
                    nn.Conv2d(prev_dim, dim, kernel_size=1, stride=1)
                )
            self.stage_attns.append(_SpatialReductionAttention(dim, heads, sr))
            self.stage_norms.append(nn.LayerNorm(dim))
            prev_dim = dim

        self.output_dim: int = embed_dims[-1]
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        x = self.patch_embed(x)  # -> (batch, embed_dims[0], H/patch, W/patch)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # -> (batch, H*W, C)
        tokens = self.patch_norm(tokens)

        for merge, merge_fallback, attn, norm in zip(
            self.stage_merges, self.stage_merges_fallback, self.stage_attns, self.stage_norms
        ):
            if not isinstance(merge, nn.Identity):
                # Reshape to spatial grid, merge patches (halves H, W),
                # flatten back to a (shorter) token sequence.
                spatial = tokens.transpose(1, 2).reshape(batch, channels, height, width)
                if height >= 2 and width >= 2:
                    spatial = merge(spatial)
                else:
                    # Degenerate spatial size (e.g. a single re-inflated
                    # 1x1 token) -- a 2x2-strided merge has no valid
                    # receptive field here, so fall back to a 1x1
                    # channel-only projection that preserves H=W=1.
                    spatial = merge_fallback(spatial)
                batch, channels, height, width = spatial.shape
                tokens = spatial.flatten(2).transpose(1, 2)

            attn_out = attn(tokens, height, width)
            tokens = norm(tokens + attn_out)  # Residual + LayerNorm.

        spatial_out = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        pooled = self.pool(spatial_out).flatten(1)  # -> (batch, output_dim)
        return pooled

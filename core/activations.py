"""
core.activations
=================
Custom, from-scratch implementations of the GELU and ELU activation
functions used by the two parallel branches of the DualBranchVerifier.

WHY IMPLEMENT THESE OURSELVES INSTEAD OF USING `torch.nn.GELU` / `torch.nn.ELU`?
    1. Pedagogical: the inline math comments below let a beginner see exactly
       what tensor operation is being applied, element-wise, to every
       activation value.
    2. Extensibility: because these are plain `nn.Module` subclasses with the
       math spelled out, it is trivial to swap in a variant (e.g. a
       "tanh-approximate GELU" vs. the exact erf-based GELU) by editing a
       single method, without touching any other file.
    3. Numerical parity: PyTorch's built-in ops are used *internally* here
       (torch.erf, torch.exp) for speed and numerical stability -- we are not
       reinventing floating point arithmetic, just making the formula
       explicit and swappable.

Both classes are drop-in replacements for `nn.GELU()` / `nn.ELU()`, i.e. they
accept a tensor of any shape and return a tensor of the identical shape.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn


class CustomGELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU).

    Mathematical definition (exact form):
        GELU(x) = x * Phi(x)
    where Phi(x) is the standard Gaussian cumulative distribution function:
        Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

    Intuition for beginners:
        GELU can be read as "multiply the input by the probability that a
        standard normal random variable is less than x". For large positive
        x, Phi(x) -> 1, so GELU(x) -> x (behaves like identity / ReLU).
        For large negative x, Phi(x) -> 0, so GELU(x) -> 0 (it gets
        "gated off"). Unlike ReLU, GELU is smooth everywhere (no kink at
        x=0), which empirically improves gradient flow in deep transformer
        and CNN-hybrid architectures such as CoAtNet.

    Shape:
        Input:  (*) -- any shape.
        Output: (*) -- identical shape to input.
    """

    #: sqrt(2), pre-computed once at class-definition time to avoid
    #: recomputing this constant on every forward pass.
    _SQRT_2: Final[float] = math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: compute the argument to the Gaussian CDF, x / sqrt(2).
        scaled_x = x / self._SQRT_2

        # Step 2: evaluate the Gaussian CDF itself.
        #   Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        # torch.erf is the (numerically stable, vectorised) error function.
        gaussian_cdf = 0.5 * (1.0 + torch.erf(scaled_x))

        # Step 3: element-wise multiply the original input by its own CDF
        # value. This is the defining "gating" operation of GELU.
        return x * gaussian_cdf


class CustomELU(nn.Module):
    """
    Exponential Linear Unit (ELU).

    Mathematical definition:
        ELU(x) = x                          if x > 0
        ELU(x) = alpha * (exp(x) - 1)       if x <= 0

    Intuition for beginners:
        For positive inputs, ELU behaves exactly like the identity function
        (same as ReLU's positive branch). For negative inputs, instead of
        clamping to zero (as ReLU does), ELU smoothly saturates towards
        `-alpha` as x -> -infinity. This means:
          (a) ELU can output negative values, which keeps the mean
              activation of a layer closer to zero ("bias shift"
              reduction), and
          (b) the curve is smooth (though not twice-differentiable at 0),
              which tends to produce more stable gradients than the
              hard zero of ReLU for very negative activations.

    Args:
        alpha: Scale for the negative-input saturation asymptote. Higher
            alpha allows a "more negative" saturation value.

    Shape:
        Input:  (*) -- any shape.
        Output: (*) -- identical shape to input.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha: float = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.where lets us apply a different formula element-wise
        # depending on a boolean condition tensor, without any Python-level
        # loop over the tensor's elements (which would be extremely slow).
        positive_branch = x
        # clamp(max=0) before exponentiating on the whole tensor keeps the
        # exponential numerically bounded (exp of a very large positive
        # number would overflow); we only actually use this branch's value
        # where x <= 0 anyway, thanks to torch.where below.
        negative_branch = self.alpha * (torch.exp(torch.clamp(x, max=0.0)) - 1.0)

        return torch.where(x > 0, positive_branch, negative_branch)

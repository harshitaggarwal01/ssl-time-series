"""Finite, scale-robust preprocessing for raw challenge windows."""

from __future__ import annotations

import torch

from .constants import CHANNELS, CONTEXT


def sanitize(x: torch.Tensor) -> torch.Tensor:
    """Replace non-finite values without changing finite values."""
    finite = torch.isfinite(x)
    return torch.where(finite, x, torch.zeros_like(x))


def normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalize each window independently over its time axis."""
    x = sanitize(x)
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    scale = torch.sqrt(torch.clamp(variance, min=eps * eps))
    return centered / scale


def real_points(x: torch.Tensor) -> torch.Tensor:
    """Where the series is, as opposed to its left padding: False on the leading run of the first value.

    The last point of that run counts as real, because it is the first value of the series itself.
    """
    x = sanitize(x)
    moved = (x != x[..., :1]).float().cumsum(dim=-1) > 0
    return torch.cat((moved[..., 1:], torch.ones_like(moved[..., :1])), dim=-1) | moved


def normalize_over(x: torch.Tensor, real: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Like ``normalize``, with the mean and scale taken over the points where ``real`` is True."""
    x = sanitize(x)
    weight = real.to(x.dtype)
    count = weight.sum(dim=-1, keepdim=True)
    centered = x - (x * weight).sum(dim=-1, keepdim=True) / count
    variance = (centered * centered * weight).sum(dim=-1, keepdim=True) / count
    return centered / torch.sqrt(torch.clamp(variance, min=eps * eps))


def derived_channels(x: torch.Tensor, real: torch.Tensor | None = None) -> torch.Tensor:
    """Return normalized level, difference, and absolute-difference channels.

    With ``real`` (from ``real_points``) the statistics come from the series alone, not from its padding.
    """
    if x.ndim != 3 or x.shape[1] != CHANNELS or x.shape[2] != CONTEXT:
        raise ValueError(f"expected [B, {CHANNELS}, {CONTEXT}], got {tuple(x.shape)}")
    scale = normalize if real is None else (lambda values: normalize_over(values, real))
    level = scale(x)
    difference = x[..., 1:] - x[..., :-1]
    difference = torch.cat((torch.zeros_like(x[..., :1]), difference), dim=-1)
    difference = scale(difference)
    magnitude = scale(difference.abs())
    return torch.cat((level, difference, magnitude), dim=1)


def pad_left(values, context: int = CONTEXT):
    """Pad a one-dimensional array on the left using its first finite value."""
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"expected a one-dimensional series, got {values.shape}")
    if len(values) == 0:
        return np.zeros(context, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) >= context:
        return values[-context:].astype(np.float32, copy=False)
    return np.pad(values, (context - len(values), 0), mode="edge")

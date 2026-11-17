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


def derived_channels(x: torch.Tensor) -> torch.Tensor:
    """Return normalized level, difference, and absolute-difference channels."""
    if x.ndim != 3 or x.shape[1] != CHANNELS or x.shape[2] != CONTEXT:
        raise ValueError(f"expected [B, {CHANNELS}, {CONTEXT}], got {tuple(x.shape)}")
    level = normalize(x)
    difference = x[..., 1:] - x[..., :-1]
    difference = torch.cat((torch.zeros_like(x[..., :1]), difference), dim=-1)
    difference = normalize(difference)
    magnitude = normalize(difference.abs())
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

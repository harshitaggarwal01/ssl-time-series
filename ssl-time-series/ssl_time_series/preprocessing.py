"""Finite, scale-robust preprocessing for raw challenge windows."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .constants import CHANNELS, CONTEXT


FEATURE_GROUP_CHANNELS = {
    "multiscale": 4,
    "log_returns": 4,
    "volatility": 2,
    "momentum": 3,
    "scale": 1,
}


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


def _rolling_mean(values: torch.Tensor, window: int) -> torch.Tensor:
    """Causal rolling mean with left-edge replication and unchanged length."""
    padded = F.pad(values, (window - 1, 0), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1)


def feature_channel_count(feature_groups: tuple[str, ...] | list[str] = ()) -> int:
    """Number of channels produced by the baseline plus selected feature groups."""
    unknown = set(feature_groups) - FEATURE_GROUP_CHANNELS.keys()
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    return 3 + sum(FEATURE_GROUP_CHANNELS[name] for name in set(feature_groups))


def derived_channels(
    x: torch.Tensor,
    feature_groups: tuple[str, ...] | list[str] = (),
) -> torch.Tensor:
    """Return baseline channels plus optional finance-inspired feature groups.

    Optional channels are scaled relative to the window's level or return scale,
    so they remain useful for arbitrary units and stable on constant windows.
    """
    if x.ndim != 3 or x.shape[1] != CHANNELS or x.shape[2] != CONTEXT:
        raise ValueError(f"expected [B, {CHANNELS}, {CONTEXT}], got {tuple(x.shape)}")
    level = normalize(x)
    difference = x[..., 1:] - x[..., :-1]
    difference = torch.cat((torch.zeros_like(x[..., :1]), difference), dim=-1)
    difference = normalize(difference)
    magnitude = normalize(difference.abs())
    channels = [level, difference, magnitude]

    raw = sanitize(x)
    level_scale = torch.sqrt(((raw - raw.mean(dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True))
    safe_level_scale = level_scale.clamp_min(1e-5)
    raw_difference = torch.cat((torch.zeros_like(raw[..., :1]), raw[..., 1:] - raw[..., :-1]), dim=-1)

    if "multiscale" in feature_groups:
        # Horizon changes, expressed in units of the window's level scale.
        for lag in (2, 4, 8, 16):
            lagged = F.pad(raw[..., :-lag], (lag, 0), mode="replicate")
            history = F.pad(torch.ones_like(raw[..., lag:]), (lag, 0)) > 0.5
            change = ((raw - lagged) / safe_level_scale).clamp(-10.0, 10.0)
            channels.append(torch.where(history, change, torch.zeros_like(change)))

    if "log_returns" in feature_groups:
        # Log differences are defined only where both observed levels are positive.
        positive = raw > 0
        safe_log = torch.log(raw.clamp_min(1e-8))
        for lag in (1, 4, 16, 64):
            lagged_log = F.pad(safe_log[..., :-lag], (lag, 0), mode="replicate")
            lagged_positive = F.pad(positive[..., :-lag].to(raw.dtype), (lag, 0), mode="replicate") > 0.5
            history = F.pad(torch.ones_like(raw[..., lag:]), (lag, 0)) > 0.5
            valid = positive & lagged_positive & history
            log_return = (safe_log - lagged_log).clamp(-10.0, 10.0)
            channels.append(torch.where(valid, log_return, torch.zeros_like(log_return)))

    if "volatility" in feature_groups:
        # Local RMS returns divided by the window's global RMS return.
        global_rms = torch.sqrt((raw_difference.square()).mean(dim=-1, keepdim=True)).clamp_min(1e-5)
        for window in (16, 64):
            local_rms = torch.sqrt(_rolling_mean(raw_difference.square(), window).clamp_min(1e-10))
            channels.append(torch.log1p(local_rms / global_rms).clamp(0.0, 5.0))

    if "momentum" in feature_groups:
        # Smoothed signed movement at two horizons and their short/long spread.
        short = _rolling_mean(raw_difference, 16) * (16.0 / safe_level_scale)
        long = _rolling_mean(raw_difference, 64) * (64.0 / safe_level_scale)
        short = short.clamp(-10.0, 10.0)
        long = long.clamp(-10.0, 10.0)
        channels.extend((short, long, (short - long).clamp(-10.0, 10.0)))

    if "scale" in feature_groups:
        # Broadcast one robust scale summary over time for the convolutional stem.
        log_scale = torch.log1p(level_scale).clamp(0.0, 20.0)
        channels.append(log_scale.expand_as(raw))

    return torch.cat(channels, dim=1)


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

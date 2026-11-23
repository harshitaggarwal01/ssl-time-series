"""Contrastive views, scale-aware augmentations and the TS2Vec-style hierarchical loss (Track 4).

All views keep the convention of the corpus windows: real points are right-aligned and the left padding repeats
the first real value, so ``preprocessing.real_points`` still finds the padding after any augmentation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _restore_padding(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Set every point before the first valid one to that point's value. ``values`` [B, T], ``valid`` [B, T]."""
    first = valid.float().argmax(dim=1, keepdim=True)
    position = torch.arange(values.shape[1], device=values.device)[None, :]
    return torch.where(position < first, values.gather(1, first), values)


def prefix_view(x: torch.Tensor, valid: torch.Tensor, low: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop a random prefix of the real part, keeping ``low`` to 100% of it; the end stays in place.

    Two such views of one window are aligned point for point where both are real, which the timestep-level
    (temporal) contrast needs. Returns the view [B, 1, T] and its validity [B, T].
    """
    batch, _, length = x.shape
    device = x.device
    first = valid.float().argmax(dim=1)
    real = (length - first).clamp(min=1)
    keep = ((low + (1 - low) * torch.rand(batch, device=device)) * real).floor().long().clamp(min=1)
    keep = torch.minimum(keep, real)
    kept = torch.arange(length, device=device)[None, :] >= (length - keep)[:, None]
    view_valid = valid & kept
    return _restore_padding(x[:, 0], kept)[:, None, :], view_valid


def augment(x: torch.Tensor, valid: torch.Tensor, jitter: float = 0.0, warp: float = 0.0, knots: int = 6) -> torch.Tensor:
    """Scale-aware jitter and magnitude warping of the real part of each window.

    The model normalizes each window, so raw scaling or offsets would be no-ops and raw-unit noise would depend on
    the series' scale. Here the noise is ``jitter`` times the window's own standard deviation, and the warp
    multiplies the deviation from the window mean by a smooth random curve around 1 (spread ``warp``).
    """
    if not (jitter or warp):
        return x
    values = x[:, 0]
    weight = valid.float()
    count = weight.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (values * weight).sum(dim=1, keepdim=True) / count
    std = ((values - mean).pow(2) * weight).sum(dim=1, keepdim=True).div(count).sqrt().clamp(min=1e-5)
    if warp:
        curve = 1 + warp * torch.randn(values.shape[0], 1, knots, device=x.device)
        curve = F.interpolate(curve, size=values.shape[1], mode="linear", align_corners=True)[:, 0]
        values = mean + (values - mean) * curve
    if jitter:
        values = values + jitter * std * torch.randn_like(values)
    return _restore_padding(values, valid)[:, None, :]


def pair_loss(z1: torch.Tensor, z2: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """InfoNCE within each of G groups of 2N items: item i of ``z1`` matches item i of ``z2``.

    ``z1``, ``z2`` [G, N, D]; ``valid`` [G, N] marks the items that take part as anchors and negatives. Dot-product
    similarity without a temperature, as in TS2Vec.
    """
    n = z1.shape[1]
    z = torch.cat((z1, z2), dim=1)
    mask = torch.cat((valid, valid), dim=1)
    similarity = z @ z.transpose(1, 2)
    index = torch.arange(2 * n, device=z.device)
    partner = (index + n) % (2 * n)
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    positive = torch.zeros_like(eye)
    positive[index, partner] = True
    blocked = (~mask[:, None, :] | eye) & ~positive  # the positive always stays, so no row is all -inf
    loss = -similarity.masked_fill(blocked, float("-inf")).log_softmax(dim=-1)[:, index, partner]
    weight = mask.float()
    return (loss * weight).sum() / weight.sum().clamp(min=1)


def hierarchical_contrast(
    a: torch.Tensor,
    b: torch.Tensor,
    overlap: torch.Tensor,
    alpha: float = 0.5,
    instance_steps: int = 32,
    temporal_steps: int = 64,
) -> torch.Tensor:
    """TS2Vec's hierarchical contrastive loss on two aligned views.

    ``a``, ``b`` [B, D, T] representations; ``overlap`` [B, T] where both views are real. At each level the
    instance term contrasts samples at the same timestep and the temporal term contrasts timesteps of the same
    sample; then both are max-pooled by 2 over time, down to one step. Timesteps are subsampled to bound memory.
    """
    a = a.float().transpose(1, 2)
    b = b.float().transpose(1, 2)
    total, levels = a.new_zeros(()), 0
    while True:
        steps = a.shape[1]
        if alpha:
            pick = torch.randperm(steps, device=a.device)[:instance_steps]
            total = total + alpha * pair_loss(a[:, pick].transpose(0, 1), b[:, pick].transpose(0, 1), overlap[:, pick].T)
        if steps > 1 and alpha < 1:
            pick = torch.randperm(steps, device=a.device)[:temporal_steps].sort().values
            total = total + (1 - alpha) * pair_loss(a[:, pick], b[:, pick], overlap[:, pick])
        levels += 1
        if steps == 1:
            return total / levels
        fill = ~overlap[..., None]
        a = F.max_pool1d(a.masked_fill(fill, -1e4).transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
        b = F.max_pool1d(b.masked_fill(fill, -1e4).transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
        overlap = F.max_pool1d(overlap[:, None].float(), 2, ceil_mode=True)[:, 0] > 0
        a = a.masked_fill(~overlap[..., None], 0.0)
        b = b.masked_fill(~overlap[..., None], 0.0)

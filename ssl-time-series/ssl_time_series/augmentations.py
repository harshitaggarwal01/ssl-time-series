"""Label-free augmentations that preserve useful temporal structure."""

from __future__ import annotations

import torch


def random_view(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    noise_std: float = 0.01,
    mask_probability: float = 0.08,
) -> torch.Tensor:
    """Create a scale/offset/noise/masking view of a batch of windows."""
    if x.ndim != 3:
        raise ValueError(f"expected [B, C, T], got {tuple(x.shape)}")
    device = x.device
    scale = torch.exp(torch.randn((x.shape[0], 1, 1), generator=generator, device=device) * 0.15)
    offset = torch.randn((x.shape[0], 1, 1), generator=generator, device=device) * 0.05
    noise = torch.randn(x.shape, generator=generator, device=device) * noise_std
    mask = torch.rand(x.shape, generator=generator, device=device) < mask_probability
    return torch.where(mask, torch.zeros_like(x), x * scale + offset + noise)


def mask_spans(
    x: torch.Tensor,
    patch_size: int,
    probability: float = 0.25,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask whole temporal patches and return the masked tensor and mask."""
    if x.ndim != 3 or x.shape[-1] % patch_size:
        raise ValueError("window length must be divisible by patch_size")
    patches = x.shape[-1] // patch_size
    mask = torch.rand((x.shape[0], patches), generator=generator, device=x.device) < probability
    expanded = mask.repeat_interleave(patch_size, dim=1).unsqueeze(1)
    return torch.where(expanded, torch.zeros_like(x), x), mask

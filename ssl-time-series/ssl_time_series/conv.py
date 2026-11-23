"""A dilated convolutional encoder (as in TS2Vec) and its training objective.

The encoder works at a quarter of the time resolution; the objective masks spans after the input projection,
reconstructs the masked points, and contrasts the pooled vectors of two random views. The heads are training-only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contrastive import augment, hierarchical_contrast, prefix_view
from .mae import crop_view, sample_mask
from .model import POOLINGS
from .preprocessing import derived_channels, feature_channel_count, real_points

STRIDE = 4  # points per position of the representation; full resolution does not fit a 12 GB GPU at batch 512
MASK_UNIT = 16  # points per masking unit
OBJECTIVES = ("instance", "hierarchical")
# Segment pooling (Track 4): the mean of each of N equal time segments plus the spread over the whole window, so the
# embedding keeps *where* in the window things happen. Conv encoder only; N * D + D dimensions.
SEGMENT_POOLINGS = ("seg2_std", "seg4_std")
CONV_POOLINGS = POOLINGS + SEGMENT_POOLINGS


class Block(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.first = nn.Conv1d(width, width, kernel_size=3, dilation=dilation, padding=dilation)
        self.second = nn.Conv1d(width, width, kernel_size=3, dilation=dilation, padding=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.second(F.gelu(self.first(F.gelu(x))))


class ConvEncoder(nn.Module):
    """[B, 1, T] raw windows -> [B, D] (mean over time) or [B, 3D] (mean, max and spread over time)."""

    def __init__(
        self, embed_dim: int = 320, hidden: int = 96, depth: int = 10, pooling: str = "mean", real_only: bool = False,
        feature_groups: tuple[str, ...] | list[str] = (),
    ):
        super().__init__()
        if pooling not in CONV_POOLINGS:
            raise ValueError(f"pooling must be one of {CONV_POOLINGS}")
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.real_only = real_only  # normalize and pool over the series alone, not over its left padding
        self.feature_groups = tuple(feature_groups)  # Track 2 channels, appended after the 3 base ones
        self.input = nn.Conv1d(feature_channel_count(self.feature_groups), hidden, kernel_size=STRIDE, stride=STRIDE)
        self.blocks = nn.ModuleList(Block(hidden, 2**index) for index in range(depth))
        self.output = nn.Conv1d(hidden, embed_dim, kernel_size=1)

    def features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Representation [B, D, T / STRIDE]; ``mask`` [B, T / STRIDE] hides positions from the layers above the input."""
        hidden = self.input(self.channels(x))
        if mask is not None:
            hidden = hidden.masked_fill(mask[:, None, :], 0.0)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)

    def channels(self, x: torch.Tensor) -> torch.Tensor:
        return derived_channels(x, real_points(x) if self.real_only else None, self.feature_groups)

    def pooled(self, features: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean, maximum and spread of ``features`` over time, [B, D] each."""
        if self.real_only:
            # Only the positions that hold at least one point of the series take part.
            real = real_points(x).reshape(x.shape[0], 1, -1, STRIDE).any(dim=-1)
            weight = real.to(features.dtype)
            count = weight.sum(dim=-1)
            mean = (features * weight).sum(dim=-1) / count
            spread = ((features - mean[:, :, None]).pow(2) * weight).sum(dim=-1).div(count).sqrt()
            return mean, features.masked_fill(~real, -1e9).amax(dim=-1), spread
        mean = features.mean(dim=-1)
        return mean, features.amax(dim=-1), (features - mean[:, :, None]).pow(2).mean(dim=-1).sqrt()

    def segment_means(self, features: torch.Tensor, x: torch.Tensor, segments: int) -> torch.Tensor:
        """Mean of ``features`` over each of ``segments`` equal stretches of the window, [B, segments * D].

        Segments are fixed positions in the window (series are right-aligned), so the same part of an aligned series
        always lands in the same segment. With ``real_only`` a segment averages only its real positions; a segment
        that is all padding gives zeros.
        """
        batch, width, steps = features.shape
        if self.real_only:
            weight = real_points(x).reshape(batch, 1, -1, STRIDE).any(dim=-1).to(features.dtype)
        else:
            weight = torch.ones_like(features[:, :1])
        features = (features * weight).reshape(batch, width, segments, steps // segments).sum(dim=-1)
        count = weight.reshape(batch, 1, segments, steps // segments).sum(dim=-1).clamp(min=1.0)
        return (features / count).transpose(1, 2).reshape(batch, segments * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        if self.pooling in SEGMENT_POOLINGS:
            segments = int(self.pooling[3])
            return torch.cat((self.segment_means(features, x, segments), self.pooled(features, x)[2]), dim=1)
        mean, peak, spread = self.pooled(features, x)
        parts = {"mean": (mean,), "mean_std": (mean, spread), "mean_max_std": (mean, peak, spread)}
        return torch.cat(parts[self.pooling], dim=1)


class ConvObjective(nn.Module):
    """Masked reconstruction of the normalized level plus a contrast between two views of each window.

    The defaults are the Track 3 recipe. Track 4 options: ``reconstruction`` weights the reconstruction loss (0 =
    contrastive only; the masking then acts as an augmentation, as in TS2Vec); ``temperature`` is the InfoNCE
    temperature; ``jitter`` and ``warp`` add scale-aware augmentations to each view; ``objective`` is
    ``"instance"`` (InfoNCE on the pooled mean and max of two crops) or ``"hierarchical"`` (TS2Vec: instance and
    temporal contrast at every time scale, on two aligned prefix-dropped views).
    """

    teacher = None  # the training loop updates a teacher where there is one

    def __init__(
        self,
        encoder: ConvEncoder,
        ratio: float = 0.4,
        span: int = 2,
        contrast: float = 1.0,
        resize: bool = False,
        reconstruction: float = 1.0,
        temperature: float = 0.1,
        jitter: float = 0.0,
        warp: float = 0.0,
        objective: str = "instance",
        crop_low: float = 0.8,
    ):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.encoder = encoder
        self.ratio = ratio
        self.span = span
        self.contrast = contrast
        self.resize = resize
        self.reconstruction = reconstruction
        self.temperature = temperature
        self.jitter = jitter
        self.warp = warp
        self.objective = objective
        self.crop_low = crop_low
        self.reconstruction_head = nn.Conv1d(encoder.embed_dim, STRIDE, kernel_size=1)
        self.contrast_head = nn.Sequential(nn.LayerNorm(2 * encoder.embed_dim), nn.Linear(2 * encoder.embed_dim, 128))

    def view(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.objective == "hierarchical":
            view, view_valid = prefix_view(x, valid, self.crop_low)
        else:
            view, view_valid = crop_view(x, valid, self.resize, self.crop_low)
        return augment(view, view_valid, self.jitter, self.warp), view_valid

    def view_loss(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruction loss of one view and its representation [B, D, T / STRIDE]."""
        batch, _, length = x.shape
        units = sample_mask(batch, length // MASK_UNIT, self.ratio, self.span, device=x.device)
        features = self.encoder.features(x, units.repeat_interleave(MASK_UNIT // STRIDE, dim=1))
        if not self.reconstruction:
            return features.new_zeros((), dtype=torch.float32), features
        prediction = self.reconstruction_head(features).float().transpose(1, 2).reshape(batch, length)
        weight = units.repeat_interleave(MASK_UNIT, dim=1) & valid
        error = F.smooth_l1_loss(prediction, self.encoder.channels(x)[:, 0], reduction="none")
        return (error * weight).sum() / weight.sum().clamp(min=1), features

    def loss(self, x: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x`` is [B, 1, T] raw windows; ``valid`` is [B, T], True at real, finite points."""
        (va, ma), (vb, mb) = self.view(x, valid), self.view(x, valid)
        (first, fa), (second, fb) = self.view_loss(va, ma), self.view_loss(vb, mb)
        if self.objective == "hierarchical":
            real = lambda m: m.reshape(m.shape[0], -1, STRIDE).any(dim=-1)  # noqa: E731
            contrast = hierarchical_contrast(fa, fb, real(ma) & real(mb))
        else:
            a = torch.cat(self.encoder.pooled(fa, va)[:2], dim=1)
            b = torch.cat(self.encoder.pooled(fb, vb)[:2], dim=1)
            a = F.normalize(self.contrast_head(a).float(), dim=-1)
            b = F.normalize(self.contrast_head(b).float(), dim=-1)
            logits = a @ b.T / self.temperature
            labels = torch.arange(len(x), device=x.device)
            contrast = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        reconstruction = (first + second) / 2
        return {
            "loss": self.reconstruction * reconstruction + self.contrast * contrast,
            "reconstruction": reconstruction,
            "contrast": contrast,
        }

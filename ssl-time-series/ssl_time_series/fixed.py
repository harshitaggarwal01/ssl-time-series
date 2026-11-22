"""Untrained features appended to the exported embedding.

They are not trained on anything, so they cannot overfit, and they carry what the trained encoder is weak at or
discards: the magnitude spectrum and random dilated convolutions (as in ROCKET and Hydra) of the series and of
its first difference, and the raw scale of the window.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .constants import CONTEXT
from .preprocessing import sanitize

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 127)
KERNEL = 9
GROUP = 8  # kernels that compete with each other
FREQUENCIES = 512


class FixedFeatures(nn.Module):
    """[B, 1, T] raw windows -> [B, 2 * (bands + kernels) + kernels + 5]."""

    def __init__(self, bands: int = 64, kernels: int = 384, seed: int = 0):
        super().__init__()
        if FREQUENCIES % bands or kernels % (GROUP * len(DILATIONS)):
            raise ValueError("bands must divide 512 and kernels must be a multiple of 64")
        self.bands = bands
        self.dim = 2 * (bands + kernels) + kernels + 5
        # The discrete Fourier transform as a matrix product, which every ONNX runtime supports.
        phase = (torch.arange(CONTEXT)[:, None] * torch.arange(FREQUENCIES)[None, :]) % CONTEXT
        angle = phase.double() * (2 * math.pi / CONTEXT)
        self.register_buffer("fourier", torch.cat((angle.cos(), angle.sin()), dim=1).float())
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn((kernels, 1, KERNEL), generator=generator)
        self.register_buffer("weight", weight - weight.mean(dim=-1, keepdim=True))
        self.register_buffer("bias", torch.rand(kernels, generator=generator) * 2 - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        raw = sanitize(x)[:, 0]
        # A short series arrives left-padded with its first value. Statistics and pooling use the real part only,
        # so the features do not depend on how much padding a series happens to have.
        moved = (raw != raw[:, :1]).float().cumsum(dim=1) > 0
        real = (torch.cat((moved[:, 1:], torch.ones_like(moved[:, :1])), dim=1) | moved).float()
        count = real.sum(dim=1, keepdim=True)

        def normalize(values: torch.Tensor) -> torch.Tensor:
            centered = values - (values * real).sum(dim=1, keepdim=True) / count
            variance = (centered * centered * real).sum(dim=1, keepdim=True) / count
            return centered / torch.sqrt(torch.clamp(variance, min=1e-10))

        # The normalized level and first difference, each treated as a series of its own (as in MultiRocket).
        difference = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, 1:] - raw[:, :-1]), dim=1)
        signal = torch.stack((normalize(raw), normalize(difference)), dim=1).reshape(2 * batch, 1, -1)
        weight = real.repeat_interleave(2, dim=0)[:, None, :]
        total = weight.sum(dim=-1)
        cosine, sine = (signal[:, 0] @ self.fourier).chunk(2, dim=1)
        magnitude = torch.log1p(torch.sqrt(cosine * cosine + sine * sine))
        parts = [magnitude.reshape(2 * batch, self.bands, -1).mean(dim=-1)]
        per = len(self.bias) // len(DILATIONS)
        winners = []
        for index, dilation in enumerate(DILATIONS):
            group = slice(index * per, (index + 1) * per)
            padding = (KERNEL // 2) * dilation
            response = F.conv1d(signal, self.weight[group], dilation=dilation, padding=padding)
            # Share of positive responses only, as in MiniRocket: the maximum response made the probe worse.
            positive = (response + self.bias[group][None, :, None] > 0).float()
            parts.append((positive * weight).sum(dim=-1) / total)
            # Competing kernels, as in Hydra, on the level: how often each kernel of a group of GROUP responds
            # most strongly. A different statistic of the same responses, and complementary to the share above.
            grouped = response[0::2].reshape(batch, per // GROUP, GROUP, -1)
            first = (grouped == grouped.amax(dim=2, keepdim=True)).float()
            winners.append(((first * real[:, None, None, :]).sum(dim=-1) / count[:, :, None]).reshape(batch, per))
        features = [torch.cat(parts, dim=1).reshape(batch, -1), torch.cat(winners, dim=1)]

        # Scale: statistics of the log-compressed raw values, finite for any finite float32 input.
        compressed = torch.sign(raw) * torch.log1p(raw.abs())
        low, high = compressed.amin(dim=-1), compressed.amax(dim=-1)
        features.append(torch.stack((compressed.mean(dim=-1), compressed.std(dim=-1), low, high, high - low), dim=1))
        return torch.cat(features, dim=1)


PARTS = ("spectrum", "ppv", "hydra", "scale")


class FixedSubset(nn.Module):
    """Only some parts of FixedFeatures: ``spectrum`` (bands per signal), ``ppv`` (MiniRocket share of positive
    responses), ``hydra`` (competing-kernel counts) and ``scale`` (5 raw-scale statistics).

    Why: the probe standardizes every dimension, so a block that helps a few hidden tasks can still cost others by
    sheer size (leaderboard, 22 Nov: the full 1,285 dimensions cost tasks 1, 3, 4 and 12).
    """

    def __init__(self, parts, bands: int = 64, kernels: int = 384, seed: int = 0):
        super().__init__()
        unknown = set(parts) - set(PARTS)
        if unknown or not parts:
            raise ValueError(f"parts must be a non-empty subset of {PARTS}")
        self.fixed = FixedFeatures(bands, kernels, seed)
        columns = []
        for signal in range(2):  # level, then first difference
            start = signal * (bands + kernels)
            if "spectrum" in parts:
                columns += range(start, start + bands)
            if "ppv" in parts:
                columns += range(start + bands, start + bands + kernels)
        if "hydra" in parts:
            columns += range(2 * (bands + kernels), 2 * (bands + kernels) + kernels)
        if "scale" in parts:
            columns += range(self.fixed.dim - 5, self.fixed.dim)
        self.dim = len(columns)
        self.register_buffer("columns", torch.tensor(sorted(columns), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fixed(x).index_select(1, self.columns)


class HybridEncoder(nn.Module):
    """The exported model: the trained encoder's embedding followed by the untrained features."""

    def __init__(self, encoder: nn.Module, fixed: FixedFeatures):
        super().__init__()
        self.encoder = encoder
        self.fixed = fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.encoder(x), self.fixed(x)), dim=1)

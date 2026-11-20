"""Compact ONNX-friendly multiscale patch transformer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .constants import CHANNELS, CONTEXT
from .preprocessing import derived_channels, real_points


POOLINGS = ("mean", "mean_max_std", "mean_std")
STEM_CHANNELS = 32
STEM_DILATIONS = (1, 2, 3)


class SeriesEncoder(nn.Module):
    """Encode raw windows independently into fixed-width embeddings."""

    def __init__(
        self,
        embed_dim: int = 320,
        patch_size: int = 32,
        depth: int = 4,
        num_heads: int = 5,
        dropout: float = 0.0,
        pooling: str = "mean",
        stem: bool = False,
        real_only: bool = False,
    ):
        super().__init__()
        self.real_only = real_only  # normalize and pool over the series alone, not over its left padding
        if CONTEXT % patch_size:
            raise ValueError("patch_size must divide the context length")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.embed_dim = embed_dim
        self.pooling = pooling
        self.patch_size = patch_size
        self.patch_count = CONTEXT // patch_size
        self.patch = nn.Conv1d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.stem = None
        if stem:
            self.stem = nn.ModuleList(
                nn.Conv1d(3, STEM_CHANNELS, kernel_size=9, dilation=d, padding=4 * d) for d in STEM_DILATIONS
            )
            self.stem_project = nn.Linear(2 * STEM_CHANNELS * len(STEM_DILATIONS), embed_dim)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def stem_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """[B, 3, T] -> [B, N, D]: dilated convolutions pooled over each patch.

        Every patch is convolved on its own, so a token never sees values from another patch and masked patches
        cannot leak into visible tokens.
        """
        batch = features.shape[0]
        local = features.reshape(batch, 3, self.patch_count, self.patch_size).transpose(1, 2)
        local = local.reshape(batch * self.patch_count, 3, self.patch_size)
        maps = torch.cat([F.gelu(conv(local)) for conv in self.stem], dim=1)
        pooled = torch.cat((maps.mean(dim=-1), maps.amax(dim=-1)), dim=1)
        return self.stem_project(pooled).reshape(batch, self.patch_count, self.embed_dim)

    def channels(self, x: torch.Tensor) -> torch.Tensor:
        return derived_channels(x, real_points(x) if self.real_only else None)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        features = self.channels(x)
        tokens = self.patch(features).transpose(1, 2)
        if self.stem is not None:
            tokens = tokens + self.stem_tokens(features)
        return tokens + self.position

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.embed(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens(x)
        if self.real_only:
            # Only the tokens that hold at least one point of the series take part in the pooling.
            real = real_points(x).reshape(x.shape[0], self.patch_count, self.patch_size).any(dim=-1)[:, :, None]
            weight = real.to(tokens.dtype)
            count = weight.sum(dim=1)
            mean = (tokens * weight).sum(dim=1) / count
            if self.pooling == "mean":
                return self.norm(mean)
            spread = ((tokens - mean[:, None, :]).pow(2) * weight).sum(dim=1).div(count).sqrt()
            if self.pooling == "mean_std":
                return torch.cat((mean, spread), dim=1)
            return torch.cat((mean, tokens.masked_fill(~real, -1e9).amax(dim=1), spread), dim=1)
        mean = tokens.mean(dim=1)
        if self.pooling == "mean":
            return self.norm(mean)
        # Padding dilutes the mean of short series; the maximum and the spread over tokens keep what it loses.
        spread = (tokens - mean[:, None, :]).pow(2).mean(dim=1).sqrt()
        if self.pooling == "mean_std":  # enough next to the untrained features, and 320 dimensions fewer
            return torch.cat((mean, spread), dim=1)
        return torch.cat((mean, tokens.amax(dim=1), spread), dim=1)

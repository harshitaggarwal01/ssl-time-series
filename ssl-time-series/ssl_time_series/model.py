"""Compact ONNX-friendly multiscale patch transformer."""

from __future__ import annotations

import torch
from torch import nn

from .constants import CHANNELS, CONTEXT
from .preprocessing import derived_channels


class SeriesEncoder(nn.Module):
    """Encode raw windows independently into fixed-width embeddings."""

    def __init__(
        self,
        embed_dim: int = 320,
        patch_size: int = 32,
        depth: int = 4,
        num_heads: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if CONTEXT % patch_size:
            raise ValueError("patch_size must divide the context length")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_count = CONTEXT // patch_size
        self.patch = nn.Conv1d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
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

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        features = derived_channels(x)
        tokens = self.patch(features).transpose(1, 2)
        return self.transformer(tokens + self.position)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tokens(x).mean(dim=1))

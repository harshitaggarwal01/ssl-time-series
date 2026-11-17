"""Training heads and label-free objectives; heads are not exported."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .augmentations import mask_spans, random_view
from .preprocessing import normalize


class SSLModule(nn.Module):
    """Wrap an encoder with temporary reconstruction and probe heads."""

    def __init__(self, encoder, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        d = encoder.embed_dim
        self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))
        self.direction = nn.Linear(d, num_classes)
        self.reconstruction = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, encoder.patch_size))

    def loss(self, x: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        view_a = random_view(x, generator=generator)
        view_b = random_view(x, generator=generator)
        za = F.normalize(self.projector(self.encoder(view_a)), dim=-1)
        zb = F.normalize(self.projector(self.encoder(view_b)), dim=-1)
        logits = za @ zb.transpose(0, 1) / 0.1
        labels = torch.arange(x.shape[0], device=x.device)
        contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

        masked, mask = mask_spans(x, self.encoder.patch_size, generator=generator)
        tokens = self.encoder.tokens(masked)
        prediction = self.reconstruction(tokens)
        target = normalize(x).reshape(x.shape[0], 1, -1, self.encoder.patch_size).squeeze(1)
        reconstruction = F.smooth_l1_loss(prediction[mask], target[mask]) if mask.any() else prediction.sum() * 0

        embedding = self.encoder(x)
        direction_target = torch.sign(x[:, 0, -1] - x[:, 0, -2]).long() + 1
        direction = F.cross_entropy(self.direction(embedding), direction_target)
        total = 0.45 * reconstruction + 0.40 * contrastive + 0.15 * direction
        return {
            "loss": total,
            "reconstruction": reconstruction.detach(),
            "contrastive": contrastive.detach(),
            "direction": direction.detach(),
        }

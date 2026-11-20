"""Masked-autoencoder objective; the decoder and head are training-only and are not exported."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


TARGETS = {
    "level": ("level",),
    "difference": ("difference",),
    "both": ("level", "difference"),
    "spectrum": ("spectrum",),
    "level_spectrum": ("level", "spectrum"),
    "latent": ("latent",),
    "level_latent": ("level", "latent"),
}
# "only": the decoder sees nothing but the mean of the visible tokens; "both": that loss is added to the usual one.
BOTTLENECKS = ("off", "both", "only")


def sample_mask(batch: int, patches: int, ratio: float, span: int = 1, device=None) -> torch.Tensor:
    """Mask exactly ``round(ratio * patches)`` patches per row, in runs of about ``span`` patches.

    ``span=1`` is random patch masking; ``span >= patches`` is one contiguous (circular) block.
    """
    count = round(ratio * patches)
    shift = torch.randint(patches, (batch, 1), device=device)
    position = (torch.arange(patches, device=device) + shift) % patches
    # Runs are masked in random order; within a run patches go in order, so a partly masked run stays contiguous.
    order = torch.rand((batch, -(-patches // span)), device=device).argsort(dim=1).argsort(dim=1)
    key = order.gather(1, position // span) * span + position % span
    return key.argsort(dim=1).argsort(dim=1) < count


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, C, T] -> [B, T / patch_size, C * patch_size]."""
    batch, channels, _ = x.shape
    return x.reshape(batch, channels, -1, patch_size).transpose(1, 2).reshape(batch, -1, channels * patch_size)


def crop_view(x: torch.Tensor, valid: torch.Tensor, resize: bool, low: float = 0.8):
    """A random view of each window: a piece covering ``low`` to 100% of its real part.

    With ``resize`` the piece is stretched to the whole window by linear interpolation; without, it keeps its
    length and is left-padded with its first value, as short series are. Returns the view and its validity.
    """
    batch, _, length = x.shape
    device = x.device
    first = valid.float().argmax(dim=1)  # real points are right-aligned
    real = (length - first).float()
    crop = ((low + (1 - low) * torch.rand(batch, device=device)) * real).clamp(min=2.0)
    start = first + torch.rand(batch, device=device) * (real - crop).clamp(min=0.0)
    kept = torch.full_like(crop, float(length)) if resize else crop.floor()
    offset = torch.arange(length, device=device)[None, :] - (length - kept)[:, None]  # < 0 in the padding
    step = (crop - 1) / (kept - 1).clamp(min=1.0)
    position = (start[:, None] + offset.clamp(min=0) * step[:, None]).clamp(max=length - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=length - 1)
    share = position - left
    values = x[:, 0]
    view = values.gather(1, left) * (1 - share) + values.gather(1, right) * share
    view_valid = valid.gather(1, left) & valid.gather(1, right) & (offset >= 0)
    return view[:, None, :], view_valid


class MaskedAutoencoder(nn.Module):
    """Encode the visible patches, then reconstruct the masked ones with a small decoder.

    With ``contrast > 0`` the loss is computed on two random views of each window, and the mean of the visible
    tokens of one view must also pick out the other view of the same window among the rest of the batch.
    """

    def __init__(
        self,
        encoder,
        ratio: float = 0.4,
        span: int = 4,
        target: str = "level",
        decoder_dim: int = 160,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        contrast: float = 0.0,
        resize: bool = True,
        bottleneck: str = "off",
        crop_low: float = 0.8,
    ):
        super().__init__()
        if bottleneck not in BOTTLENECKS:
            raise ValueError(f"bottleneck must be one of {BOTTLENECKS}")
        self.bottleneck = bottleneck
        self.crop_low = crop_low  # smallest share of the real part that a contrastive view keeps
        self.encoder = encoder
        self.ratio = ratio
        self.span = span
        self.contrast = contrast
        self.resize = resize
        self.parts = TARGETS[target]
        self.project = nn.Sequential(nn.LayerNorm(encoder.embed_dim), nn.Linear(encoder.embed_dim, decoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.position = nn.Parameter(torch.zeros(1, encoder.patch_count, decoder_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=decoder_depth, norm=nn.LayerNorm(decoder_dim))
        sizes = {"level": encoder.patch_size, "difference": encoder.patch_size, "spectrum": encoder.patch_size // 2 + 1}
        sizes["latent"] = encoder.embed_dim
        self.head = nn.Linear(decoder_dim, sum(sizes[part] for part in self.parts))
        # The latent target is the tokens of a slowly moving copy of the encoder that sees the whole window.
        self.teacher = copy.deepcopy(encoder).requires_grad_(False) if "latent" in self.parts else None
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        if contrast:
            self.contrast_head = nn.Sequential(nn.LayerNorm(encoder.embed_dim), nn.Linear(encoder.embed_dim, 128))

    def targets(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-patch reconstruction targets [B, N, K] and which of their entries count in the loss."""
        size = self.encoder.patch_size
        channels = self.encoder.channels(x)
        level = patchify(channels[:, :1], size)
        valid = valid[:, None, :]
        real = patchify(valid, size)
        targets, weights = [], []
        for part in self.parts:
            if part == "level":
                targets.append(level)
                weights.append(real)
            elif part == "difference":
                # A difference is only meaningful where the previous point is valid too.
                previous = torch.cat((valid[..., :1], valid[..., :-1]), dim=-1)
                targets.append(patchify(channels[:, 1:2], size))
                weights.append(patchify(valid & previous, size))
            elif part == "latent":
                with torch.no_grad():
                    hidden = self.teacher.tokens(x).float()
                latent = F.layer_norm(hidden, hidden.shape[-1:])
                targets.append(latent)
                weights.append(real.any(dim=-1, keepdim=True).expand_as(latent))
            else:
                # Log magnitude spectrum of the patch: what it contains, regardless of where inside the patch.
                spectrum = torch.log1p(torch.fft.rfft(level, dim=-1).abs())
                targets.append(spectrum)
                weights.append(real.all(dim=-1, keepdim=True).expand_as(spectrum))
        return torch.cat(targets, dim=-1), torch.cat(weights, dim=-1)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Move the teacher towards the encoder: ``teacher = momentum * teacher + (1 - momentum) * encoder``."""
        for old, new in zip(self.teacher.parameters(), self.encoder.parameters()):
            old.lerp_(new, 1 - momentum)

    def reconstruct(self, x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The reconstruction loss and the mean of the visible tokens [B, D]."""
        encoder = self.encoder
        tokens = encoder.embed(x)
        batch, patches, width = tokens.shape
        mask = sample_mask(batch, patches, self.ratio, self.span, device=x.device)
        visible = encoder.transformer(tokens[~mask].view(batch, -1, width))
        target, weight = self.targets(x, valid)

        def error(decoder_input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            prediction = self.head(self.decoder(decoder_input + self.position)).float()
            return (F.smooth_l1_loss(prediction, target, reduction="none") * weight).sum() / weight.sum().clamp(min=1)

        loss = 0.0
        if self.bottleneck != "only":
            hidden = self.project(visible)
            decoded = self.mask_token.to(hidden.dtype).repeat(batch, patches, 1)
            decoded[~mask] = hidden.flatten(0, 1)
            loss = error(decoded, weight & mask[:, :, None])
        if self.bottleneck != "off":
            # The pooled vector is all the decoder gets, so it has to describe the whole window: this is the
            # vector the linear probe reads, which the usual loss never trains directly.
            summary = self.project(visible.mean(dim=1, keepdim=True)).expand(-1, patches, -1)
            loss = loss + error(summary, weight)
        return loss, visible.mean(dim=1)

    def loss(self, x: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x`` is [B, 1, T] raw windows; ``valid`` is [B, T], True at real, finite points."""
        if not self.contrast:
            return {"loss": self.reconstruct(x, valid)[0]}
        views = (crop_view(x, valid, self.resize, self.crop_low) for _ in range(2))
        (first, a), (second, b) = (self.reconstruct(*view) for view in views)
        a = F.normalize(self.contrast_head(a).float(), dim=-1)
        b = F.normalize(self.contrast_head(b).float(), dim=-1)
        logits = a @ b.T / 0.1
        labels = torch.arange(len(x), device=x.device)
        contrast = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        reconstruction = (first + second) / 2
        return {"loss": reconstruction + self.contrast * contrast, "reconstruction": reconstruction, "contrast": contrast}

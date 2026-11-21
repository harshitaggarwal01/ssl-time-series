"""Train the encoder with the masked-autoencoder objective."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import corpus_loader
from ssl_time_series.conv import ConvEncoder, ConvObjective
from ssl_time_series.data import CorpusWindowDataset, split_indices
from ssl_time_series.mae import BOTTLENECKS, TARGETS, MaskedAutoencoder
from ssl_time_series.model import SeriesEncoder


def autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def validate(module: MaskedAutoencoder, loader: DataLoader, device: torch.device, batches: int) -> dict[str, float]:
    """Held-out reconstruction loss and the effective rank of the pooled embeddings."""
    module.eval()
    torch.manual_seed(0)  # the same masks for every run
    losses, embeddings = [], []
    with torch.no_grad(), autocast(device):
        for _, batch in zip(range(batches), loader):
            x, valid = batch["x"].to(device), batch["valid"].to(device)
            losses.append(float(module.loss(x, valid)["loss"]))
            embeddings.append(module.encoder(x).float())
    z = torch.cat(embeddings)
    spectrum = torch.linalg.svdvals(z - z.mean(dim=0))
    p = spectrum / spectrum.sum()
    rank = torch.exp(-(p * torch.log(p + 1e-12)).sum())
    return {"validation_loss": float(np.mean(losses)), "effective_rank": float(rank)}


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    total = corpus_loader.load(args.corpus)[1]
    train_stop, _ = split_indices(total, args.validation_fraction)
    dataset = CorpusWindowDataset(
        args.corpus, windows_per_series=args.windows_per_series, seed=args.seed, start=0, stop=train_stop
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    held_out = DataLoader(
        CorpusWindowDataset(args.corpus, seed=args.seed, start=train_stop, stop=total), batch_size=args.batch_size
    )
    encoder = SeriesEncoder(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        depth=args.depth,
        num_heads=args.heads,
        stem=args.stem,
        real_only=args.real_only,
        feature_groups=args.feature_groups,
    )
    module = MaskedAutoencoder(
        encoder,
        ratio=args.mask_ratio,
        span=args.mask_span,
        target=args.target,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        contrast=args.contrast,
        resize=args.view == "resize",
        bottleneck=args.bottleneck,
        crop_low=args.crop_low,
    ).to(device)
    if args.arch == "conv":  # --depth is the number of dilated blocks, --mask-span counts units of 16 points
        encoder = ConvEncoder(embed_dim=args.embed_dim, hidden=args.hidden, depth=args.depth, real_only=args.real_only,
            feature_groups=args.feature_groups,
        )
        module = ConvObjective(
            encoder, ratio=args.mask_ratio, span=args.mask_span, contrast=args.contrast, resize=args.view == "resize"
        ).to(device)
    trained =[parameter for parameter in module.parameters() if parameter.requires_grad]  # not the teacher
    optimizer = torch.optim.AdamW(trained, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.05)
    warmup = max(1, int(0.05 * args.steps))

    def scale(step: int) -> float:  # linear warmup, then cosine decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    module.train()
    iterator = iter(loader)
    history = []
    recent = []
    started = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch["x"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            loss = module.loss(x, valid)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optimizer.step()
        schedule.step()
        if module.teacher is not None:  # the momentum rises linearly to 1 over the run
            module.update_teacher(args.teacher_momentum + (1 - args.teacher_momentum) * step / args.steps)
        recent.append(loss.item())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {"step": step, "loss": float(np.mean(recent)), "seconds": round(time.time() - started, 1)}
            recent = []
            history.append(row)
            print(json.dumps(row), flush=True)
    result = validate(module, held_out, device, args.validation_batches)
    print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.cpu().state_dict(),
            "config": vars(args),
            "history": history,
            "validation": result,
        },
        args.output,
    )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/mae.pt"))
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--windows-per-series", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--arch", choices=("transformer", "conv"), default="transformer")
    parser.add_argument("--hidden", type=int, default=96, help="channel width of the conv encoder")
    parser.add_argument("--embed-dim", type=int, default=320)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--stem", action="store_true", help="add patch-local dilated convolutions to the tokens")
    parser.add_argument(
        "--feature-groups", nargs="*", default=[],
        choices=("multiscale", "log_returns", "volatility", "momentum", "scale"),
        help="Track 2 channels added to the input of either encoder",
    )
    parser.add_argument("--real-only", action="store_true", help="normalize and pool over the series, not its padding")
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--mask-span", type=int, default=4, help="run length in patches; 1 is random, >= patch count is one block")
    parser.add_argument("--target", choices=sorted(TARGETS), default="level")
    parser.add_argument("--teacher-momentum", type=float, default=0.996, help="starting value, for latent targets")
    parser.add_argument("--bottleneck", choices=BOTTLENECKS, default="off", help="reconstruct from the pooled vector")
    parser.add_argument("--contrast", type=float, default=0.0, help="weight of the contrastive loss between two views")
    parser.add_argument("--view", choices=("resize", "crop"), default="resize", help="stretch each view or re-pad it")
    parser.add_argument("--crop-low", type=float, default=0.8, help="smallest share of the series a view keeps")
    parser.add_argument("--decoder-dim", type=int, default=160)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        values = json.loads(config_args.config.read_text())
        parser.set_defaults(**values)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

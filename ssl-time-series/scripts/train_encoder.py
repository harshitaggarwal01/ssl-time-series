"""Train the self-supervised encoder on the challenge corpus."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import corpus_loader
from ssl_time_series.data import CorpusWindowDataset, split_indices
from ssl_time_series.model import SeriesEncoder
from ssl_time_series.objectives import SSLModule


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    get, total = corpus_loader.load(args.corpus)
    del get
    train_stop, _ = split_indices(total, args.validation_fraction)
    dataset = CorpusWindowDataset(
        args.corpus,
        windows_per_series=args.windows_per_series,
        seed=args.seed,
        start=0,
        stop=train_stop,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    encoder = SeriesEncoder(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        depth=args.depth,
        num_heads=args.heads,
    )
    module = SSLModule(encoder).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=0.01)
    module.train()
    iterator = iter(loader)
    history = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch["x"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        values = module.loss(x)
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {name: float(value) for name, value in values.items()}
            row["step"] = step
            history.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.cpu().state_dict(),
            "config": vars(args),
            "history": history,
        },
        args.output,
    )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/encoder.pt"))
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64, help="reserved for future contract changes")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--windows-per-series", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--embed-dim", type=int, default=320)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        values = json.loads(config_args.config.read_text())
        parser.set_defaults(**values)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

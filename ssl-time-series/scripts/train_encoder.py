"""Train the self-supervised encoder on the challenge corpus."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import corpus_loader
from ssl_time_series.data import CorpusWindowDataset, split_indices
from ssl_time_series.model import SeriesEncoder
from ssl_time_series.objectives import SSLModule


class _Tee:
    """Write terminal output to both the original stream and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _save_checkpoint(
    path: Path,
    *,
    encoder: SeriesEncoder,
    module: SSLModule,
    optimizer: torch.optim.Optimizer,
    config: dict,
    history: list[dict],
    step: int,
) -> Path:
    """Atomically save enough state to export or continue training."""
    checkpoint_dir = path.parent / path.stem
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step-{step:06d}{path.suffix or '.pt'}"
    checkpoint = {
        "step": step,
        "encoder": encoder.state_dict(),
        "module": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    return path


def _restore_rng(checkpoint: dict) -> None:
    rng = checkpoint.get("rng")
    if not rng:
        return
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


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
        feature_groups=args.feature_groups,
    )
    module = SSLModule(encoder).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=0.01)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    start_step = 0
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_config = resume_checkpoint.get("config", {})
        for name in ("embed_dim", "patch_size", "depth", "heads", "feature_groups", "seed"):
            if name in saved_config and saved_config[name] != getattr(args, name):
                raise ValueError(
                    f"cannot resume with a different {name}: checkpoint has "
                    f"{saved_config[name]!r}, requested {getattr(args, name)!r}"
                )
        if "module" in resume_checkpoint:
            module.load_state_dict(resume_checkpoint["module"])
        else:
            # Accept older export-only checkpoints; their training heads and
            # optimizer state were not saved, so those start fresh.
            encoder.load_state_dict(resume_checkpoint["encoder"])
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_step = int(resume_checkpoint.get("step", 0))
        history = list(resume_checkpoint.get("history", []))
        print(f"Resuming from step {start_step}: {args.resume}", flush=True)

    module.train()
    iterator = iter(loader)
    if resume_checkpoint is not None:
        # Iterator creation can consume the global torch RNG; restore afterward
        # so objective augmentations continue from the saved random state.
        _restore_rng(resume_checkpoint)
    if args.steps <= start_step:
        raise ValueError(f"--steps ({args.steps}) must exceed resumed step ({start_step})")
    for step in range(start_step + 1, args.steps + 1):
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
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            saved_path = _save_checkpoint(
                args.output,
                encoder=encoder,
                module=module,
                optimizer=optimizer,
                config=vars(args),
                history=history,
                step=step,
            )
            print(f"Saved checkpoint at step {step}: {saved_path}", flush=True)
    final_path = _save_checkpoint(
        args.output,
        encoder=encoder,
        module=module,
        optimizer=optimizer,
        config=vars(args),
        history=history,
        step=args.steps,
    )
    print(f"Saved final checkpoint at step {args.steps}: {final_path}", flush=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/encoder.pt"),
        help="checkpoint name; step-numbered files go in a folder with this name",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="training log path (default: <output folder>/train.log)",
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--checkpoint-every", type=int, default=500, help="save every N steps; 0 disables periodic saves")
    parser.add_argument("--resume", type=Path, default=None, help="resume from a training checkpoint")
    parser.add_argument("--batch-size", type=int, default=64, help="reserved for future contract changes")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--windows-per-series", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--embed-dim", type=int, default=320)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument(
        "--feature-groups",
        nargs="*",
        choices=("multiscale", "log_returns", "volatility", "momentum", "scale"),
        default=[],
        help="optional channels added to the baseline: multiscale, log_returns, volatility, momentum, scale",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        values = json.loads(config_args.config.read_text())
        parser.set_defaults(**values)
    args = parser.parse_args()
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    log_file = args.log_file or (args.output.parent / args.output.stem / "train.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_file.open("a", encoding="utf-8", buffering=1) as log_stream:
        sys.stdout = _Tee(original_stdout, log_stream)
        sys.stderr = _Tee(original_stderr, log_stream)
        try:
            print(json.dumps({"event": "train_start", "args": vars(args)}, default=str), flush=True)
            train(args)
        except BaseException:
            traceback.print_exc(file=log_stream)
            log_stream.flush()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()

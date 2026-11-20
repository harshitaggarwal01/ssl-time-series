"""Load a trained checkpoint and export only its encoder to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Disable PyTorch's native TransformerEncoderLayer fast path. It exports an
# unsupported aten::_transformer_encoder_layer_fwd operator.
if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
    torch.backends.mha.set_fastpath_enabled(False)

from ssl_time_series.conv import ConvEncoder
from ssl_time_series.export import export_encoder
from ssl_time_series.fixed import FixedFeatures, HybridEncoder
from ssl_time_series.model import POOLINGS, SeriesEncoder

MAX_BYTES = 20 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/model.onnx"))
    parser.add_argument("--pooling", choices=POOLINGS, default="mean", help="how tokens become the embedding")
    parser.add_argument("--fixed", action="store_true", help="append the untrained features to the embedding")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    encoder = SeriesEncoder(
        embed_dim=int(config.get("embed_dim", 320)),
        patch_size=int(config.get("patch_size", 32)),
        depth=int(config.get("depth", 4)),
        num_heads=int(config.get("heads", 5)),
        pooling=args.pooling,
        stem=bool(config.get("stem", False)),
        real_only=bool(config.get("real_only", False)),
    )
    if config.get("arch") == "conv":
        encoder = ConvEncoder(
            embed_dim=int(config["embed_dim"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
            pooling=args.pooling,
            real_only=bool(config.get("real_only", False)),
        )
    encoder.load_state_dict(checkpoint["encoder"])
    path = export_encoder(HybridEncoder(encoder, FixedFeatures()) if args.fixed else encoder, args.output)
    size = path.stat().st_size
    parameters = sum(parameter.numel() for parameter in encoder.parameters())
    print(f"{path} ({size / 1024 ** 2:.2f} MiB, {parameters:,} parameters)")
    if size > MAX_BYTES:
        raise SystemExit(f"ONNX file is {size / 1024 ** 2:.2f} MiB; limit is 20 MiB")


if __name__ == "__main__":
    main()

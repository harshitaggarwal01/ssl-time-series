"""Load trained checkpoint(s) and export the encoder(s) to ONNX.

Several checkpoints are concatenated into one embedding (an ensemble); ``--fixed`` appends the untrained
features and ``--stats`` the window statistics block.
"""

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
from ssl_time_series.fixed import PARTS, FixedFeatures, FixedSubset
from ssl_time_series.model import POOLINGS, SeriesEncoder
from ssl_time_series.stats import ConcatEncoder, WindowStats

MAX_BYTES = 20 * 1024 * 1024


def load_encoder(path: Path, pooling: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("arch") == "conv":
        encoder = ConvEncoder(
            embed_dim=int(config["embed_dim"]),
            hidden=int(config["hidden"]),
            depth=int(config["depth"]),
            pooling=pooling,
            real_only=bool(config.get("real_only", False)),
            feature_groups=config.get("feature_groups", []),
        )
    else:
        encoder = SeriesEncoder(
            embed_dim=int(config.get("embed_dim", 320)),
            patch_size=int(config.get("patch_size", 32)),
            depth=int(config.get("depth", 4)),
            num_heads=int(config.get("heads", 5)),
            pooling=pooling,
            stem=bool(config.get("stem", False)),
            real_only=bool(config.get("real_only", False)),
            feature_groups=config.get("feature_groups", []),
        )
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="one or more; concatenated")
    parser.add_argument("--output", type=Path, default=Path("outputs/model.onnx"))
    parser.add_argument("--pooling", choices=POOLINGS, nargs="+", default=["mean"],
                        help="how tokens become the embedding; one value, or one per checkpoint")
    parser.add_argument("--fixed", action="store_true", help="append the untrained features to the embedding")
    parser.add_argument("--fixed-parts", choices=PARTS, nargs="+",
                        help="append only these parts of the untrained features (instead of --fixed)")
    parser.add_argument("--stats", action="store_true", help="append the window statistics block")
    args = parser.parse_args()
    poolings = args.pooling * len(args.checkpoint) if len(args.pooling) == 1 else args.pooling
    if len(poolings) != len(args.checkpoint):
        parser.error("give one --pooling, or one per --checkpoint")
    encoders = [load_encoder(path, pooling) for path, pooling in zip(args.checkpoint, poolings)]
    if args.fixed and args.fixed_parts:
        parser.error("give --fixed or --fixed-parts, not both")
    fixed = [FixedFeatures()] if args.fixed else [FixedSubset(args.fixed_parts)] if args.fixed_parts else []
    parts = encoders + fixed + ([WindowStats()] if args.stats else [])
    model = parts[0] if len(parts) == 1 else ConcatEncoder(*parts)
    with torch.no_grad():
        dim = model.eval()(torch.randn(64, 1, 1024)).shape[1]
    if dim > 2048:
        raise SystemExit(f"embedding has {dim} dimensions; limit is 2048")
    path = export_encoder(model, args.output)
    size = path.stat().st_size
    parameters = sum(parameter.numel() for encoder in encoders for parameter in encoder.parameters())
    print(f"{path} ({size / 1024 ** 2:.2f} MiB, {parameters:,} parameters, {dim} dimensions)")
    if size > MAX_BYTES:
        raise SystemExit(f"ONNX file is {size / 1024 ** 2:.2f} MiB; limit is 20 MiB")


if __name__ == "__main__":
    main()

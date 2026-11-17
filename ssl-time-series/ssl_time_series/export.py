"""Export and validate the encoder portion of a training checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from .constants import BATCH, CHANNELS, CONTEXT, OPSET


def export_encoder(encoder: torch.nn.Module, output: str | Path) -> Path:
    """Export a static-batch float32 ONNX encoder."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder.eval().cpu()
    example = torch.randn(BATCH, CHANNELS, CONTEXT)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (example,),
            str(output),
            input_names=["x"],
            output_names=["embedding"],
            opset_version=OPSET,
            dynamo=False,
        )
    return output

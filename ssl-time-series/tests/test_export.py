import torch

from ssl_time_series.export import export_encoder
from ssl_time_series.model import SeriesEncoder


def test_export_writes_onnx(tmp_path):
    model = SeriesEncoder(embed_dim=64, patch_size=64, depth=1, num_heads=4)
    output = export_encoder(model, tmp_path / "model.onnx")
    assert output.is_file()
    assert output.stat().st_size > 0

import torch

from evaluate import BATCH, CHANNELS, CONTEXT
from ssl_time_series.model import SeriesEncoder


def test_encoder_shape_and_batch_independence():
    model = SeriesEncoder(embed_dim=64, patch_size=64, depth=1, num_heads=4).eval()
    x = torch.randn(BATCH, CHANNELS, CONTEXT)
    with torch.no_grad():
        result = model(x)
        permutation = torch.randperm(BATCH)
        permuted = model(x[permutation])
    assert result.shape == (BATCH, 64)
    assert torch.isfinite(result).all()
    assert torch.allclose(result[permutation], permuted, atol=1e-5, rtol=1e-4)

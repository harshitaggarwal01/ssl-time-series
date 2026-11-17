import numpy as np
import pytest
import torch

from ssl_time_series.preprocessing import derived_channels, normalize, pad_left


def test_pad_left_matches_short_series_contract():
    result = pad_left(np.array([2.0, 3.0], dtype=np.float32), context=4)
    np.testing.assert_array_equal(result, [2.0, 2.0, 2.0, 3.0])


def test_normalize_is_finite_for_constant_and_nonfinite_windows():
    x = torch.tensor([[[1.0, float("nan"), 1.0, float("inf")]]])
    result = normalize(x)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, torch.zeros_like(result))


def test_derived_channels_have_expected_shape():
    x = torch.randn(4, 1, 1024)
    result = derived_channels(x)
    assert result.shape == (4, 3, 1024)
    assert torch.isfinite(result).all()


def test_pad_left_rejects_non_1d_input():
    with pytest.raises(ValueError):
        pad_left(np.ones((2, 2), dtype=np.float32))

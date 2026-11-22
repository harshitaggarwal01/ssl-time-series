import torch

from ssl_time_series.stats import ConcatEncoder, WindowStats


def test_window_stats_finite_on_edge_cases():
    x = torch.randn(8, 1, 1024).cumsum(-1) * 100 + 1e4
    x[1, 0, :900] = x[1, 0, 900]  # short series, left-padded with its first value
    x[2] = 3.0  # constant
    x[3] = torch.round(torch.rand(1, 1024) * 4)  # counts
    x[4] = 0.0
    out = WindowStats()(x)
    assert out.shape == (8, WindowStats.dim)
    assert torch.isfinite(out).all()


def test_concat_encoder_stacks_parts():
    stats = WindowStats()
    out = ConcatEncoder(stats, stats)(torch.randn(4, 1, 1024))
    assert out.shape == (4, 2 * WindowStats.dim)


def test_fixed_subset_matches_full_columns():
    from ssl_time_series.fixed import FixedFeatures, FixedSubset

    x = torch.randn(4, 1, 1024).cumsum(-1)
    full = FixedFeatures()(x)
    spectrum = FixedSubset(["spectrum", "scale"])
    assert spectrum.dim == 2 * 64 + 5
    out = spectrum(x)
    assert torch.allclose(out[:, :64], full[:, :64]) and torch.allclose(out[:, -5:], full[:, -5:])
    assert torch.allclose(out[:, 64:128], full[:, 448:512])
    every = FixedSubset(["spectrum", "ppv", "hydra", "scale"])
    assert every.dim == full.shape[1] and torch.allclose(every(x), full)

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


def test_profile_finite_and_positional():
    from ssl_time_series.stats import WindowProfile

    block = WindowProfile()
    early, late = torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024)
    early[..., 100:140] = 1.0
    late[..., 900:940] = 1.0
    x = torch.cat((early, late, torch.full((1, 1, 1024), 7.0), torch.randn(61, 1, 1024) * 1e6), dim=0)
    out = block(x)
    assert out.shape == (64, block.dim) and torch.isfinite(out).all()
    assert out[0, :32].argmax() == 100 // 32 and out[1, :32].argmax() == 900 // 32


def test_patch_profile_finite_and_positional():
    from ssl_time_series.stats import PatchProfile

    block = PatchProfile(32)
    bumpy = torch.zeros(1, 1, 1024)
    bumpy[..., 1:] = 0.01  # real from the start
    bumpy[..., 610:630] += torch.randn(20)
    x = torch.cat((bumpy, torch.full((1, 1, 1024), 3.0), torch.randn(62, 1, 1024) * 1e7), dim=0)
    out = block(x)
    assert out.shape == (64, 96) and torch.isfinite(out).all()
    assert out[0, 32:64].argmax() == 610 // 32

import pytest
import torch

from ssl_time_series.contrastive import augment, hierarchical_contrast, pair_loss, prefix_view
from ssl_time_series.conv import ConvEncoder, ConvObjective
from ssl_time_series.preprocessing import real_points


def batch(n: int = 6):
    x = torch.randn(n, 1, 1024).cumsum(-1) * 50 + 1e3
    valid = torch.ones(n, 1024, dtype=torch.bool)
    x[1, 0, :700] = x[1, 0, 700]  # short series, left-padded with its first value
    valid[1, :700] = False
    return x, valid


def test_prefix_view_keeps_end_and_padding_convention():
    torch.manual_seed(0)
    x, valid = batch()
    view, view_valid = prefix_view(x, valid, low=0.5)
    assert torch.equal(view[:, 0, -1], x[:, 0, -1])  # aligned at the end
    assert (view_valid & ~valid).sum() == 0
    assert torch.equal(real_points(view)[:, 0], view_valid)  # padding still detected
    kept = view_valid.sum(1).float() / valid.sum(1).float()
    assert (kept >= 0.5 - 1e-3).all()


def test_augment_is_scale_aware_and_keeps_padding():
    torch.manual_seed(0)
    x, valid = batch()
    out = augment(x, valid, jitter=0.05, warp=0.2)
    assert torch.equal(real_points(out)[:, 0], valid)
    small = augment(x * 1e-4, valid, jitter=0.05, warp=0.0)
    torch.manual_seed(0)
    big = augment(x, valid, jitter=0.05, warp=0.0)
    ratio = ((small - x * 1e-4).std() / (x * 1e-4).std()) / ((big - x).std() / x.std())
    assert 0.5 < float(ratio) < 2.0  # noise relative to the window, not to raw units


def test_pair_loss_masks_without_nan():
    z = torch.randn(3, 5, 8, requires_grad=True)
    valid = torch.ones(3, 5, dtype=torch.bool)
    valid[0] = False  # a group with nothing real
    valid[1, 2:] = False
    loss = pair_loss(z, z + 0.01, valid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_hierarchical_contrast_finite():
    a = torch.randn(4, 16, 256, requires_grad=True)
    overlap = torch.ones(4, 256, dtype=torch.bool)
    overlap[0, :200] = False
    loss = hierarchical_contrast(a, a.detach() + 0.1 * torch.randn_like(a), overlap)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"reconstruction": 0.0},
        {"reconstruction": 0.0, "jitter": 0.05, "warp": 0.2, "crop_low": 0.5},
        {"reconstruction": 0.0, "objective": "hierarchical", "crop_low": 0.5},
    ],
)
def test_objectives_train_step(options):
    torch.manual_seed(0)
    encoder = ConvEncoder(embed_dim=16, hidden=8, depth=3, real_only=True)
    module = ConvObjective(encoder, **options)
    x, valid = batch(8)
    values = module.loss(x, valid)
    values["loss"].backward()
    assert all(torch.isfinite(v) for v in values.values())
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
    if options.get("reconstruction") == 0.0:
        assert module.reconstruction_head.weight.grad is None


@pytest.mark.parametrize("pooling, segments", [("seg2_std", 2), ("seg4_std", 4)])
def test_segment_pooling_keeps_position(pooling, segments):
    torch.manual_seed(0)
    encoder = ConvEncoder(embed_dim=16, hidden=8, depth=3, real_only=True, pooling=pooling).eval()
    x, _ = batch(4)
    out = encoder(x)
    assert out.shape == (4, segments * 16 + 16) and torch.isfinite(out).all()
    # The window-wide mean of the segment means equals mean pooling when every position is real.
    plain = ConvEncoder(embed_dim=16, hidden=8, depth=3, real_only=True, pooling="mean_std").eval()
    plain.load_state_dict(encoder.state_dict())
    full = torch.tensor([0, 2, 3])  # rows without padding
    seg = out[full, : segments * 16].reshape(len(full), segments, 16).mean(dim=1)
    assert torch.allclose(seg, plain(x)[full, :16], atol=1e-5)
    # Moving an event to another part of the window changes the segment means.
    y = x.clone()
    y[0, 0, 100:110] += 500
    z = x.clone()
    z[0, 0, 900:910] += 500
    assert not torch.allclose(encoder(y)[0, : segments * 16], encoder(z)[0, : segments * 16], atol=1e-3)

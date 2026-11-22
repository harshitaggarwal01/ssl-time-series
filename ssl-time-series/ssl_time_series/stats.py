"""A compact block of hand-crafted window statistics appended to the exported embedding (no training).

Why: the probe standardizes every dimension and fits a ridge, so a few dozen informative, well-behaved numbers
can matter more than hundreds of noisy ones. The block keeps what in-model normalization throws away (level,
scale, length of the real series, integer-valued or non-negative data) and adds classic shape statistics,
several of them from GS Quant's ``gs_quant.timeseries`` (RSI, Bollinger bands, max drawdown, ``consecutive``
streaks, rolling volatility, rolling percentile rank).

All statistics use only the real part of a window: a short series arrives left-padded with its first value.
Everything is ONNX-exportable at opset 17 (no cummax, no quantile op) and finite for any finite float32 input.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .preprocessing import sanitize

LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
EPS = 1e-8


def _shift(values: torch.Tensor, lag: int, fill: float = 0.0) -> torch.Tensor:
    """values[t - lag] along the last axis, ``fill`` before the start."""
    return F.pad(values[..., :-lag], (lag, 0), value=fill)


def _running_max(values: torch.Tensor) -> torch.Tensor:
    """Causal running maximum by doubling shifts (log2(T) steps); cummax has no ONNX export."""
    result, lag = values, 1
    while lag < values.shape[-1]:
        result = torch.maximum(result, _shift(result, lag, fill=-1e30))
        lag *= 2
    return result


def _rolling_mean(values: torch.Tensor, window: int) -> torch.Tensor:
    return F.avg_pool1d(F.pad(values[:, None], (window - 1, 0), mode="replicate"), window, stride=1)[:, 0]


def _slog(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.log1p(values.abs())


class WindowStats(nn.Module):
    """[B, 1, T] raw windows -> [B, dim] statistics."""

    dim = 49

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = sanitize(x)[:, 0]
        length = raw.shape[-1]
        moved = (raw != raw[:, :1]).float().cumsum(dim=1) > 0
        realb = torch.cat((moved[:, 1:], torch.ones_like(moved[:, :1])), dim=1) | moved
        real = realb.float()
        n = real.sum(dim=1, keepdim=True)

        def mmean(values, weight=real, count=n):
            return (values * weight).sum(dim=1, keepdim=True) / count.clamp_min(1.0)

        mean = mmean(raw)
        centered = (raw - mean) * real
        std = torch.sqrt(mmean(centered * centered)).clamp_min(EPS)
        z = centered / std  # 0 on the padding
        diff = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, 1:] - raw[:, :-1]), dim=1)
        dreal = real * _shift(real, 1)  # both points real
        dn = dreal.sum(dim=1, keepdim=True).clamp_min(1.0)
        dz = diff / std * dreal
        dstd = torch.sqrt(mmean(dz * dz, dreal, dn)).clamp_min(EPS)

        feats = []
        # Level and scale (scale-dependent on purpose), length and data type.
        sorted_raw = torch.sort(torch.where(realb, raw, torch.full_like(raw, 3e38)), dim=1).values
        index = ((n - 1) * torch.tensor(QUANTILES, dtype=raw.dtype)[None, :]).round().long()
        raw_q = torch.gather(sorted_raw, 1, index)
        feats += [_slog(mean), torch.log1p(std), _slog(raw_q), _slog(raw[:, -1:])]
        feats += [torch.log(n / length), torch.log1p(std / (mean.abs() + EPS)).clamp(max=30.0)]
        feats.append(mmean((raw <= 0).float()))
        feats.append(mmean(((raw - torch.round(raw)).abs() < 1e-6).float()))  # integer-valued (counts)
        feats.append(mmean((diff == 0).float(), dreal, dn))  # flat steps / discreteness

        # Shape of the distribution (scale-free).
        z_q = (raw_q - mean) / std
        feats += [z_q, mmean(z ** 3), torch.log1p(mmean(z ** 4))]

        # Returns: drift, roughness, asymmetry, direction.
        dmean = mmean(dz, dreal, dn)
        feats += [dmean * n.sqrt(), torch.log(dstd), mmean((dz / dstd) ** 3, dreal, dn).clamp(-50, 50)]
        feats += [torch.log1p(mmean((dz / dstd) ** 4, dreal, dn)), mmean((diff > 0).float(), dreal, dn)]
        crossings = (z * _shift(z, 1) < 0).float()
        feats.append(mmean(crossings, dreal, dn))

        # Linear trend of the level against time: slope and R^2.
        t = torch.arange(length, dtype=raw.dtype)[None, :]
        tc = (t - mmean(t.expand_as(raw))) * real
        tstd = torch.sqrt(mmean(tc * tc)).clamp_min(EPS)
        corr = mmean(z * tc / tstd)
        feats += [corr, corr * corr]

        # Autocorrelation of the level and of absolute returns (volatility clustering).
        for lag in LAGS:
            both = real * _shift(real, lag)
            feats.append(mmean(z * _shift(z, lag), both, both.sum(dim=1, keepdim=True)))
        absd = (dz.abs() - mmean(dz.abs(), dreal, dn)) * dreal
        avar = mmean(absd * absd, dreal, dn).clamp_min(EPS)
        for lag in (1, 8):
            both = dreal * _shift(dreal, lag)
            feats.append(mmean(absd * _shift(absd, lag), both, both.sum(dim=1, keepdim=True)) / avar)

        # GS Quant: max drawdown and max run-up of the normalized level (gs_quant.timeseries.max_drawdown).
        zfill = (raw - mean) / std  # unmasked: the padding sits at the series' first value
        feats.append((zfill - _running_max(zfill)).amin(dim=1, keepdim=True))
        feats.append(-((-zfill) - _running_max(-zfill)).amin(dim=1, keepdim=True))

        # GS Quant: consecutive up / down streaks (analysis.consecutive), longest run as a share of the length.
        position = t.expand_as(raw)
        for up in (True, False):
            hit = (diff > 0) if up else (diff < 0)
            breaks = torch.where(hit & (dreal > 0), torch.full_like(raw, -1.0), position)
            run = position - _running_max(breaks)
            feats.append(run.amax(dim=1, keepdim=True) / n)

        # GS Quant: RSI(14) on level differences (technicals.relative_strength_index), mean and spread.
        gain, loss = _rolling_mean(F.relu(diff) * dreal, 14), _rolling_mean(F.relu(-diff) * dreal, 14)
        rsi = gain / (gain + loss + EPS)
        rmean = mmean(rsi, dreal, dn)
        feats += [rmean, torch.sqrt(mmean((rsi - rmean) ** 2, dreal, dn))]

        # GS Quant: Bollinger bands (w=32, k=2): share of points outside the bands.
        rm = _rolling_mean(zfill, 32)
        rs = torch.sqrt(F.relu(_rolling_mean(zfill * zfill, 32) - rm * rm)) + 1e-4
        feats.append(mmean(((zfill - rm).abs() > 2 * rs).float()))

        # GS Quant: rolling volatility (statistics.std, w=16) relative to the whole window: regime changes.
        local = torch.sqrt(F.relu(_rolling_mean(dz * dz, 16))).clamp_min(EPS) / dstd
        lr = torch.log(local)
        lmean = mmean(lr, dreal, dn)
        feats += [lmean, torch.sqrt(mmean((lr - lmean) ** 2, dreal, dn)), (lr * dreal).amax(dim=1, keepdim=True)]

        # GS Quant: percentile rank of the last value within the window (statistics.percentiles).
        feats.append(mmean((raw < raw[:, -1:]).float()))

        out = torch.cat(feats, dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


class WindowProfile(nn.Module):
    """[B, 1, T] raw windows -> [B, 48]: where things happen inside the window (no training).

    Mean and std pooling over time cannot say *where* a pattern sits, but in aligned data (one heartbeat, one
    gesture, one day per series) the position is often the class. This keeps it, coarsely: the piecewise aggregate
    approximation (PAA) of the normalized series in 32 segments, and the spread of the level and the size of the
    steps in 8 segments. All normalized over the real part; the padding contributes zeros.
    """

    dim = 48
    SEGMENTS, COARSE = 32, 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = sanitize(x)[:, 0]
        batch, length = raw.shape
        moved = (raw != raw[:, :1]).float().cumsum(dim=1) > 0
        real = (torch.cat((moved[:, 1:], torch.ones_like(moved[:, :1])), dim=1) | moved).float()
        n = real.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (raw * real).sum(dim=1, keepdim=True) / n
        centered = (raw - mean) * real
        std = torch.sqrt((centered * centered).sum(dim=1, keepdim=True) / n).clamp_min(EPS)
        z = centered / std
        step = torch.cat((torch.zeros_like(raw[:, :1]), raw[:, 1:] - raw[:, :-1]), dim=1).abs() / std
        step = step * real * _shift(real, 1)
        step = step / (step.sum(dim=1, keepdim=True) / n).clamp_min(EPS)  # relative to the window's typical step
        paa = z.reshape(batch, self.SEGMENTS, length // self.SEGMENTS).mean(dim=-1)
        coarse = z.reshape(batch, self.COARSE, length // self.COARSE)
        spread = torch.sqrt((coarse * coarse).mean(dim=-1))
        steps = torch.log1p(step.reshape(batch, self.COARSE, length // self.COARSE).mean(dim=-1))
        out = torch.cat((paa, spread, steps), dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


class ConcatEncoder(nn.Module):
    """Concatenate the outputs of several encoders / feature blocks on the same input."""

    def __init__(self, *parts: nn.Module):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([part(x) for part in self.parts], dim=1)

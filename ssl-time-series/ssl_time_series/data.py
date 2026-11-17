"""Streaming corpus sampling without materializing the 8.9 GB archive."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import corpus_loader  # noqa: E402

from .constants import CONTEXT
from .preprocessing import pad_left


class CorpusWindowDataset(Dataset):
    """Deterministic series-level samples with optional multiple windows."""

    def __init__(
        self,
        root: str | Path,
        windows_per_series: int = 1,
        seed: int = 0,
        start: int = 0,
        stop: int | None = None,
    ):
        self.get, total = corpus_loader.load(root)
        self.start = start
        self.stop = total if stop is None else min(stop, total)
        self.windows_per_series = max(1, windows_per_series)
        self.seed = seed

    def __len__(self) -> int:
        return (self.stop - self.start) * self.windows_per_series

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        series_index = self.start + index // self.windows_per_series
        window_number = index % self.windows_per_series
        series = np.asarray(self.get(series_index), dtype=np.float32)
        if len(series) <= CONTEXT:
            window = pad_left(series)
        else:
            max_start = len(series) - CONTEXT
            seed = (self.seed * 1_000_003 + series_index * 97 + window_number) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            window = series[int(rng.integers(0, max_start + 1)) :][:CONTEXT]
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        return {"x": torch.from_numpy(window.copy()).float().unsqueeze(0)}


def split_indices(total: int, validation_fraction: float = 0.05) -> tuple[int, int]:
    """Return a deterministic contiguous train/validation split boundary."""
    if total < 2:
        raise ValueError("at least two series are required")
    validation = max(1, int(total * validation_fraction))
    return total - validation, total

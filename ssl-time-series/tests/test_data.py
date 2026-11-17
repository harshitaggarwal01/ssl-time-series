import numpy as np

from ssl_time_series.data import CorpusWindowDataset


def test_dataset_can_sample_npz(tmp_path):
    values = np.arange(10, dtype=np.float32)
    offsets = np.array([0, 3, 10], dtype=np.int64)
    np.savez(tmp_path / "corpus.npz", values=values, offsets=offsets)
    dataset = CorpusWindowDataset(tmp_path, windows_per_series=2, seed=3)
    assert len(dataset) == 4
    first = dataset[0]["x"]
    second = dataset[0]["x"]
    assert first.shape == (1, 1024)
    np.testing.assert_array_equal(first.numpy(), second.numpy())

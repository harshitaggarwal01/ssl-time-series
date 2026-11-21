import json

import numpy as np

from scripts.build_proxy_suite import CONTEXT, TASKS, build


def test_suite_matches_evaluator_layout(tmp_path):
    build(tmp_path, seed=0, items=12)
    spec = json.loads((tmp_path / "tasks.json").read_text())["tasks"]
    assert [t["name"] for t in spec] == list(TASKS)
    for n, task in enumerate(spec, start=1):
        windows = np.load(tmp_path / "windows" / f"{task['windows']}.npy")
        owner = np.load(tmp_path / "windows" / f"{task['windows']}_item.npy")
        labels = np.load(tmp_path / f"task{n:02d}" / "labels.npy")
        test = np.load(tmp_path / f"task{n:02d}" / "test.npy")
        assert windows.dtype == np.float32 and windows.shape[1] == CONTEXT
        assert np.isfinite(windows).all()
        assert len(owner) == len(windows) and set(owner) == set(range(len(labels)))
        assert labels.dtype == bool and (labels.sum(axis=1) == 1).all()
        assert test.dtype == bool and test.sum() == len(labels) // 2


def test_suite_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build(a, seed=0, items=8)
    build(b, seed=0, items=8)
    for name in TASKS:
        np.testing.assert_array_equal(np.load(a / "windows" / f"{name}.npy"), np.load(b / "windows" / f"{name}.npy"))

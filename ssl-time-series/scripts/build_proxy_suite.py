"""Build a synthetic multi-task suite in the official ``evaluate.py`` data layout.

Every track scores candidates with the official evaluator on the same data:

    uv run python -m scripts.build_proxy_suite --out outputs/proxy_suite
    uv run evaluate.py outputs/baseline.onnx outputs/proxy_suite --out outputs/baseline_proxy.json

Layout written (see the docstring of ``evaluate.py``)::

    tasks.json                   {"tasks": [{"name", "windows", "floor", "reference", "weight"}, ...]}
    windows/<name>.npy           float32 [n_windows, 1024]
    windows/<name>_item.npy      int64 [n_windows], item index of each window
    taskNN/labels.npy            bool [n_items, n_classes], one-hot
    taskNN/test.npy              bool [n_items], held-out items

Anchors follow the official definition: ``floor`` is the AUC of an untrained encoder (the
baseline architecture at random initialisation, seed 0) and ``reference`` the AUC of
``example_submission/model.onnx``. A task score of 0 means "no better than random weights" and
1 means "as good as the official example encoder".

Each item gets a random scale in [1e-2, 1e3] and offset in [-1e3, 1e3], matching the evaluator's
raw, unnormalised windows. Long items (> 1024 values) are cut into consecutive 1024-value
windows whose embeddings the evaluator averages; the last partial window is dropped.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

CONTEXT = 1024
ITEMS_PER_TASK = 480


def _affine(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    return (x * 10 ** rng.uniform(-2, 3) + rng.uniform(-1e3, 1e3)).astype(np.float32)


def _ar1(rng, n, phi, sigma=1.0):
    e = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = e[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + e[t]
    return out


# Each generator returns one raw series for class `c`.
def shape(rng, c):
    t = np.linspace(0, 1, CONTEXT)
    base = [t, np.sin(2 * np.pi * 6 * t), np.sin(2 * np.pi * 2 * t), (t >= 0.5).astype(float)][c]
    return base + rng.normal(0, 0.05, CONTEXT)


def frequency(rng, c):
    cycles = [3, 8, 20, 50, 120][c] * rng.uniform(0.85, 1.15)
    t = np.arange(CONTEXT) / CONTEXT
    return np.sin(2 * np.pi * cycles * t + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.3, CONTEXT)


def trend(rng, c):
    drift = [-1.0, 0.0, 1.0][c] * 0.08
    return np.cumsum(drift + rng.normal(0, 1, CONTEXT))


def volatility_regime(rng, c):
    n = CONTEXT
    if c == 0:  # constant volatility
        sigma = np.ones(n)
    elif c == 1:  # one abrupt volatility break
        k = rng.integers(n // 4, 3 * n // 4)
        sigma = np.where(np.arange(n) < k, 1.0, rng.choice([0.25, 4.0]))
    else:  # GARCH(1,1)-style clustering
        sigma = np.empty(n)
        r_prev, s2 = 0.0, 1.0
        for t in range(n):
            s2 = 0.05 + 0.15 * r_prev**2 + 0.8 * s2
            sigma[t] = np.sqrt(s2)
            r_prev = rng.normal(0, sigma[t])
    return np.cumsum(rng.normal(0, 1, n) * sigma)


def change_point(rng, c):
    x = rng.normal(0, 1, CONTEXT)
    if c > 0:
        lo, hi = (CONTEXT // 8, CONTEXT // 2) if c == 1 else (CONTEXT // 2, 7 * CONTEXT // 8)
        x[rng.integers(lo, hi) :] += rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
    return x


def autocorrelation(rng, c):
    return _ar1(rng, CONTEXT, [-0.7, 0.0, 0.5, 0.95][c])


def anomaly(rng, c):
    x = _ar1(rng, CONTEXT, 0.6)
    if c == 1:
        idx = rng.choice(CONTEXT, size=rng.integers(1, 5), replace=False)
        x[idx] += rng.choice([-1, 1], size=len(idx)) * rng.uniform(6, 10, size=len(idx))
    return x


def seasonal_long(rng, c):
    """Multi-window items (2-4 windows): seasonality period must survive window averaging."""
    n = CONTEXT * int(rng.integers(2, 5))
    t = np.arange(n)
    noise = _ar1(rng, n, 0.5, 0.5)
    if c == 0:
        return noise
    period = [None, 24, 168][c] * rng.uniform(0.95, 1.05)
    return noise + np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))


TASKS = {
    "shape": (shape, 4),
    "frequency": (frequency, 5),
    "trend": (trend, 3),
    "volatility_regime": (volatility_regime, 3),
    "change_point": (change_point, 3),
    "autocorrelation": (autocorrelation, 4),
    "anomaly": (anomaly, 2),
    "seasonal_long": (seasonal_long, 3),
}


def build(out: Path, seed: int = 0, items: int = ITEMS_PER_TASK) -> Path:
    (out / "windows").mkdir(parents=True, exist_ok=True)
    specs = []
    for n, (name, (gen, classes)) in enumerate(TASKS.items(), start=1):
        rng = np.random.default_rng([seed, n])
        windows, owner = [], []
        labels = np.zeros((items, classes), bool)
        for i in range(items):
            c = i % classes
            labels[i, c] = True
            series = _affine(rng, gen(rng, c))
            for w in range(len(series) // CONTEXT):
                windows.append(series[w * CONTEXT : (w + 1) * CONTEXT])
                owner.append(i)
        test = np.zeros(items, bool)
        test[rng.permutation(items)[: items // 2]] = True
        np.save(out / "windows" / f"{name}.npy", np.stack(windows))
        np.save(out / "windows" / f"{name}_item.npy", np.asarray(owner, np.int64))
        folder = out / f"task{n:02d}"
        folder.mkdir(exist_ok=True)
        np.save(folder / "labels.npy", labels)
        np.save(folder / "test.npy", test)
        specs.append({"name": name, "windows": name, "floor": 0.0, "reference": 1.0, "weight": 1.0})
    (out / "tasks.json").write_text(json.dumps({"seed": seed, "tasks": specs}, indent=2) + "\n")
    return out


def raw_aucs(model: Path, data: Path) -> list[float]:
    """Per-task AUC with anchors (0, 1), i.e. score == AUC."""
    import evaluate

    result = evaluate.evaluate(str(model), str(data))
    return [result["tasks"][f"task{n}"]["auc"] for n in range(1, len(result["tasks"]) + 1)]


def untrained_model(path: Path) -> Path:
    import torch

    from ssl_time_series.export import export_encoder
    from ssl_time_series.model import SeriesEncoder

    torch.manual_seed(0)
    return export_encoder(SeriesEncoder(embed_dim=256, patch_size=32, depth=4, num_heads=4), path)


def calibrate(data: Path, reference: Path) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        floor = raw_aucs(untrained_model(Path(tmp) / "untrained.onnx"), data)
    ref = raw_aucs(reference, data)
    spec = json.loads((data / "tasks.json").read_text())
    for task, f, r in zip(spec["tasks"], floor, ref):
        task["floor"], task["reference"] = round(f, 6), round(r, 6)
        if r - f < 0.02:  # task the reference cannot beat random weights on: keep it but flag it
            task["note"] = "reference within 0.02 AUC of floor; score is noisy"
    (data / "tasks.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec["tasks"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/proxy_suite"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--items", type=int, default=ITEMS_PER_TASK)
    parser.add_argument("--reference", type=Path, default=Path("example_submission/model.onnx"))
    parser.add_argument("--no-calibrate", action="store_true")
    args = parser.parse_args()
    build(args.out, args.seed, args.items)
    if not args.no_calibrate:
        for task in calibrate(args.out, args.reference):
            print(json.dumps(task))
    print(f"suite written to {args.out}")


if __name__ == "__main__":
    main()

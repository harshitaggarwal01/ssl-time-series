"""Evaluate exported embeddings on deterministic synthetic proxy tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score

from evaluate import BATCH, CHANNELS, CONTEXT


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, CONTEXT, dtype=np.float32)
    x = np.empty((n, CONTEXT), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        label = i % 4
        labels[i] = label
        if label == 0:
            series = t
        elif label == 1:
            series = np.sin(2 * np.pi * 6 * t)
        elif label == 2:
            series = np.sin(2 * np.pi * 2 * t)
        else:
            series = np.concatenate((np.zeros(CONTEXT // 2), np.ones(CONTEXT - CONTEXT // 2)))
        x[i] = series + rng.normal(0, 0.05, CONTEXT)
        x[i] = x[i] * rng.uniform(0.1, 100.0) + rng.uniform(-1000, 1000)
    return x, labels


def embed(session, x: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    outputs = []
    for start in range(0, len(x), BATCH):
        block = np.zeros((BATCH, CONTEXT), dtype=np.float32)
        chunk = x[start : start + BATCH]
        block[: len(chunk)] = chunk
        outputs.append(session.run(None, {name: block.reshape(BATCH, CHANNELS, CONTEXT)})[0][: len(chunk)])
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    x, y = make_data(args.samples, 0)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    z = embed(session, x)
    split = len(z) // 2
    probe = RidgeClassifier().fit(z[:split], y[:split])
    prediction = probe.decision_function(z[split:])
    auc = roc_auc_score(np.eye(4)[y[split:]], prediction, multi_class="ovr", average="macro")
    print(json.dumps({"samples": len(x), "embedding_dim": int(z.shape[1]), "macro_auc": float(auc)}))


if __name__ == "__main__":
    main()

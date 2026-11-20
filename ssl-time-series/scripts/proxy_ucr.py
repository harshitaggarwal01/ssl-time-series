"""Build a labeled proxy evaluation set from UCR datasets, in the layout the official evaluator reads.

    uv run python -m scripts.proxy_ucr --out eval_data/ucr
    uv run python evaluate.py outputs/model.onnx eval_data/ucr --out outputs/model.ucr.json

The data is used for evaluation only, never for training. Windows follow the challenge's conventions: short series
are left-padded with their first value, long ones are cut into consecutive windows, missing values become 0.
Compare models by their mean AUC over the tasks. Every task has floor 0.5 and reference 1, so the evaluator's
``score`` is ``2 * mean AUC - 1``; floors from an untrained encoder would let near-saturated tasks dominate.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from evaluate import CONTEXT

URL = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
DATASETS = (
    "ECG5000",
    "ECG200",
    "CinCECGTorso",
    "NonInvasiveFetalECGThorax1",
    "FordA",
    "ElectricDevices",
    "FreezerRegularTrain",
    "ChlorineConcentration",
    "Crop",
    "StarLightCurves",
    "UWaveGestureLibraryAll",
    "Yoga",
    "MixedShapesRegularTrain",
    "HandOutlines",
    "InsectWingbeatSound",
    "Phoneme",
    "EthanolLevel",
    "SemgHandGenderCh2",
    "PigAirwayPressure",
)
# A second set that is never used to choose between models, only to confirm a choice made on the first.
HELD_OUT = (
    "FordB",
    "TwoPatterns",
    "SwedishLeaf",
    "FaceAll",
    "OSULeaf",
    "WordSynonyms",
    "Earthquakes",
    "Computers",
    "RefrigerationDevices",
    "Strawberry",
    "ShapesAll",
    "Haptics",
    "InlineSkate",
    "CricketX",
    "Adiac",
    "MedicalImages",
    "PhalangesOutlinesCorrect",
    "ItalyPowerDemand",
    "EOGVerticalSignal",
    "ACSF1",
    "PowerCons",
    "MelbournePedestrian",
)
# Datasets whose series differ in length within the dataset, so the amount of padding varies between items.
VARIABLE_LENGTH = (
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "PickupGestureWiimoteZ",
    "PLAID",
    "ShakeGestureWiimoteZ",
)


def read_ts(archive: zipfile.ZipFile, member: str) -> tuple[list[np.ndarray], list[str]]:
    """Parse a univariate ``.ts`` file: one ``v1,v2,...:label`` line per series after ``@data``."""
    series, labels = [], []
    lines = io.TextIOWrapper(archive.open(member), encoding="utf-8", errors="replace")
    in_data = False
    for line in lines:
        line = line.strip()
        if not in_data:
            in_data = line.lower().startswith("@data")
        elif line:
            values, label = line.rsplit(":", 1)
            series.append(np.array([np.nan if v.strip() == "?" else float(v) for v in values.split(",")], np.float32))
            labels.append(label)
    return series, labels


def to_windows(series: np.ndarray) -> np.ndarray:
    """[length] -> [n, CONTEXT], as the evaluator prepares its windows."""
    chunks = [series[start : start + CONTEXT] for start in range(0, len(series), CONTEXT)]
    padded = [np.pad(chunk, (CONTEXT - len(chunk), 0), mode="edge") for chunk in chunks]
    return np.nan_to_num(np.stack(padded)).astype(np.float32)


def load(name: str, cache: Path, limits: dict[str, int], rng: np.random.Generator):
    path = cache / f"{name}.zip"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(name=name), path)
    series, labels, test = [], [], []
    with zipfile.ZipFile(path) as archive:
        for split, limit in limits.items():
            part, part_labels = read_ts(archive, f"{name}_{split}.ts")
            keep = np.sort(rng.permutation(len(part))[:limit])
            series += [part[i] for i in keep]
            labels += [part_labels[i] for i in keep]
            test += [split == "TEST"] * len(keep)
    return series, labels, np.array(test)


def build(out: Path, cache: Path, max_train: int, max_test: int, datasets=DATASETS) -> None:
    rng = np.random.default_rng(0)
    cache.mkdir(parents=True, exist_ok=True)
    (out / "windows").mkdir(parents=True, exist_ok=True)
    tasks = []
    for number, name in enumerate(datasets, start=1):
        series, labels, test = load(name, cache, {"TRAIN": max_train, "TEST": max_test}, rng)
        windows = [to_windows(s) for s in series]
        item = np.repeat(np.arange(len(series)), [len(w) for w in windows])
        classes = sorted(set(labels))
        folder = out / f"task{number:02d}"
        folder.mkdir(exist_ok=True)
        np.save(out / "windows" / f"{name}.npy", np.concatenate(windows))
        np.save(out / "windows" / f"{name}_item.npy", item)
        np.save(folder / "labels.npy", np.array([[label == c for c in classes] for label in labels]))
        np.save(folder / "test.npy", test)
        tasks.append({"name": name, "windows": name, "floor": 0.5, "reference": 1.0, "weight": 1.0})
        print(f"{name}: {len(series)} items, {len(item)} windows, {len(classes)} classes, length {len(series[0])}")
    (out / "tasks.json").write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("eval_data/ucr"))
    parser.add_argument("--cache", type=Path, default=Path("eval_data/ucr_raw"))
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=1500)
    parser.add_argument("--held-out", action="store_true", help="build the held-out set; pass its own --out")
    parser.add_argument("--variable-length", action="store_true", help="build the variable-length set; pass --out")
    args = parser.parse_args()
    datasets = VARIABLE_LENGTH if args.variable_length else HELD_OUT if args.held_out else DATASETS
    build(args.out, args.cache, args.max_train, args.max_test, datasets)


if __name__ == "__main__":
    main()

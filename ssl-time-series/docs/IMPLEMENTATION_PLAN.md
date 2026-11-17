# Finance-Informed SSL Implementation Plan

This document tracks the implementation of the Stable Challenge time-series
encoder in this repository.

## Scope and rules

The official challenge requires a single ONNX encoder trained only on the
unlabeled challenge corpus. The evaluator supplies raw windows and expects a
static input shape of `[64, 1, 1024]`, one finite output embedding per window,
and an embedding width no greater than 2048.

The implementation follows these references:

- [Stable Challenge time-series](https://github.com/galilai-group/stable-challenge-time-series)
  - [`CHALLENGE.md`](https://github.com/galilai-group/stable-challenge-time-series/blob/main/CHALLENGE.md)
  - [`evaluate.py`](https://github.com/galilai-group/stable-challenge-time-series/blob/main/evaluate.py)
  - [`test_submission.py`](https://github.com/galilai-group/stable-challenge-time-series/blob/main/test_submission.py)
- [Galilai stable-finance](https://github.com/galilai-group/stable-finance)
  - [`experiments/month_cache.py`](https://github.com/galilai-group/stable-finance/blob/main/experiments/month_cache.py)
  - [`experiments/signals.py`](https://github.com/galilai-group/stable-finance/blob/main/experiments/signals.py)
  - [`src/stable_finance/evaluation/scoring/ic.py`](https://github.com/galilai-group/stable-finance/blob/main/src/stable_finance/evaluation/scoring/ic.py)
- [Goldman Sachs GS Quant](https://github.com/goldmansachs/gs-quant)
  - [`gs_quant/timeseries/econometrics.py`](https://github.com/goldmansachs/gs-quant/blob/master/gs_quant/timeseries/econometrics.py)
  - [`gs_quant/timeseries/technicals.py`](https://github.com/goldmansachs/gs-quant/blob/master/gs_quant/timeseries/technicals.py)
  - [`gs_quant/timeseries/helper.py`](https://github.com/goldmansachs/gs-quant/blob/master/gs_quant/timeseries/helper.py)

Stable Finance and GS Quant are methodological references. They are not
additional training data, pretrained weights, or required runtime dependencies
for the submitted encoder.

## Completed work

- [x] Create a branch from `main`.
- [x] Centralize the challenge contract constants.
- [x] Preserve memory-mapped corpus access.
- [x] Add deterministic series-level window sampling.
- [x] Add finite-value handling and per-window normalization.
- [x] Add level, difference, and magnitude-derived channels inside the model.
- [x] Add scale, offset, noise, and patch-mask augmentations.
- [x] Add a compact patch-transformer encoder.
- [x] Keep the default architecture within the 20 MiB submission budget.
- [x] Add masked reconstruction, contrastive, and next-step auxiliary losses.
- [x] Add checkpoint training and encoder-only ONNX export scripts.
- [x] Add corpus profiling and synthetic proxy evaluation scripts.
- [x] Add ablation command generation.
- [x] Add baseline and main reproducibility configurations.
- [x] Add focused unit tests.

## Remaining execution checklist

- [ ] Download and unzip `corpus.zip` outside the repository.
- [ ] Run `profile_corpus.py` and review `outputs/corpus_profile.json`.
- [ ] Run the baseline configuration.
- [ ] Run the main configuration.
- [ ] Run the ablation command list with fixed seeds.
- [ ] Export each candidate to ONNX.
- [ ] Run `test_submission.py` on every candidate.
- [ ] Run `proxy_evaluate.py` on passing candidates.
- [ ] Select the smallest candidate with the strongest proxy score.
- [ ] Submit only the final validated model.

## Reproducibility

Training commands use explicit seeds, fixed architecture arguments, deterministic
series-level splitting, and checkpoint metadata. Keep corpus profiling,
training logs, checkpoints, and ONNX files under `outputs/`, which is ignored
by Git.

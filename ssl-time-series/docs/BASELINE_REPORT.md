# Track 1: baseline reproduction report

Owner: Harshit. Branch: `research/baseline` (from `challenge-finance-ssl` at `100e7c8`). Run on Oscar,
NVIDIA RTX A5000, 20 Nov 2026.

## Hypothesis

The shared pipeline on `challenge-finance-ssl` works end to end on the real corpus and gives a
reproducible reference that every other track is measured against, including its seed-to-seed spread.

## What was run

The Colab notebook `notebooks/baseline_colab.ipynb`, step for step, as Slurm jobs (`slurm/`):

```bash
# from ssl-time-series/ssl-time-series, after mkdir -p logs
sbatch slurm/prep_corpus.sbatch                       # uv sync || corpus download, unzip, profile
NAME=baseline    CONFIG=configs/baseline.json SEED=0 sbatch slurm/train_candidate.sbatch
NAME=baseline-s1 CONFIG=configs/baseline.json SEED=1 sbatch slurm/train_candidate.sbatch
NAME=baseline-s2 CONFIG=configs/baseline.json SEED=2 sbatch slurm/train_candidate.sbatch
NAME=main        CONFIG=configs/main.json     SEED=0 sbatch slurm/train_candidate.sbatch
sbatch slurm/eval_models.sbatch outputs/baseline*.onnx outputs/main.onnx
```

Each training job runs `scripts.train_encoder` → `scripts.export_model` → `test_submission.py`. The only
deviation from `configs/baseline.json` is `--workers 8` (was 0). Window sampling is seeded per
(series, window) in `data.py`, so the worker count does not change which windows are drawn.

| Setting | baseline | main |
| --- | ---: | ---: |
| steps | 2,000 | 20,000 |
| batch | 64 | 64 |
| embed dim / depth / heads / patch | 256 / 4 / 4 / 32 | 320 / 4 / 5 / 32 |
| windows per series | 1 | 2 |
| learning rate | 3e-4 | 3e-4 |
| objective | reconstruction + contrastive + next-step direction | same |

## Corpus profile (`outputs/corpus_profile.json`, 10,000 sampled series)

| | |
| --- | --- |
| series | 1,762,338 |
| length min / median / p90 / p99 / max | 0 / 1,024 / 1,024 / 4,837 / 116,907 |
| finite fraction, mean / worst series | 98.6% / 0% |
| per-series mean, median / p99 abs | 9.7 / 19,340 |
| per-series std, median / p99 | 4.95 / 30,595 |

Most series are exactly one window long. About 1.4% of values are non-finite, and some series are
entirely non-finite or empty, so non-finite handling matters (see "Bug found").

## Results

### Contract and size (all candidates)

All six `test_submission.py` checks pass for every candidate: loads with a float32 `(64, 1, 1024)`
input, finite `(64, 256)` embeddings, opset 17, deterministic, batch-independent, finite under x1e-3,
x1e3 and +1000.

| Candidate | seed | train time | parameters | ONNX | GPU throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 2 min 16 s | 2,141,952 | 8.98 MiB | 33,129 windows/s |
| baseline-s1 | 1 | 2 min 0 s | 2,141,952 | 8.98 MiB | 32,881 windows/s |
| baseline-s2 | 2 | 1 min 30 s | 2,141,952 | 8.98 MiB | 36,846 windows/s |
| main | 0 | 12 min 18 s | 3,332,800 | 13.53 MiB | 29,869 windows/s |

The full official evaluation (158,078 windows) would take about 0.1 min on this GPU.

### Final training losses (step 2,000)

| seed | total | reconstruction | contrastive | direction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.303 | 0.251 | 0.184 | 0.774 |
| 1 | 0.326 | 0.255 | 0.211 | 0.846 |
| 2 | 0.282 | 0.242 | 0.141 | 0.778 |

### Original proxy (`scripts/proxy_evaluate.py`)

Macro AUC **1.000** for the seed-0 baseline. The task (ramp / two sines / step) is solved perfectly, so it
cannot rank candidates. It is replaced by the suite below.

### Proxy suite v2 (official `evaluate.py`, 8 tasks)

Score per task = (AUC − untrained) / (official example − untrained): 0 = random weights, 1 = the
official `example_submission/model.onnx`. The overall score is the weighted mean, exactly as the
leaderboard computes it. Tasks where the official example does not beat the untrained encoder by
0.02 AUC get weight 0, because the formula is meaningless there. For those tasks compare AUC.

Anchors (suite v2.1, seed 0, 480 items per task, half held out):

| task | untrained AUC | official example AUC | weight |
| --- | ---: | ---: | ---: |
| shape | 0.655 | 0.980 | 1 |
| frequency | 0.478 | 0.893 | 1 |
| trend | 0.536 | 0.690 | 1 |
| volatility_regime | 0.839 | 0.619 | 0 |
| change_point | 0.532 | 0.780 | 1 |
| autocorrelation | 0.574 | 0.790 | 1 |
| anomaly | 0.590 | 0.512 | 0 |
| seasonal_long | 0.541 | 0.898 | 1 |

Results (AUC per task):

| model | score | mean AUC | shape | frequency | trend | volatility_regime | change_point | autocorrelation | anomaly | seasonal_long |
|---|---|---|---|---|---|---|---|---|---|---|
| main | 1.217 | 0.847 | 1.000 | 0.935 | 0.726 | 0.873 | 0.933 | 0.840 | 0.546 | 0.923 |
| baseline-s1 | 1.075 | 0.813 | 0.996 | 0.927 | 0.708 | 0.882 | 0.805 | 0.805 | 0.472 | 0.910 |
| baseline | 1.061 | 0.807 | 0.996 | 0.894 | 0.711 | 0.889 | 0.818 | 0.799 | 0.454 | 0.894 |
| baseline-s2 | 1.042 | 0.808 | 0.997 | 0.919 | 0.686 | 0.876 | 0.811 | 0.787 | 0.467 | 0.919 |
| example_submission-model | 1.000 | 0.770 | 0.980 | 0.893 | 0.690 | 0.612 | 0.780 | 0.790 | 0.512 | 0.898 |

Seed spread for the 2k baseline: score **1.059 ± 0.017** (range 1.042 to 1.075), mean AUC 0.809 ± 0.003.
A candidate needs to beat the baseline by more than about 0.035 (two standard deviations) before we
treat the gain as real.

Observations:

- The 2k-step baseline **beats the official example on all three seeds** (score above 1.0; mean AUC 0.809
  vs 0.770).
- The biggest margin is volatility regime: 0.88 vs 0.62 AUC. Even the untrained encoder scores 0.84
  there. The derived difference and magnitude channels expose volatility directly, while the official
  ViT sees only the normalized level. Finance-style channels look promising, which supports Track 2.
- Trend is the weakest weighted task (about 0.70 for every model). It is the obvious target for
  objective work (Tracks 3 and 4).
- The anomaly task is near chance for every model (0.45 to 0.51), so it does not separate them yet.
- **`main` (20k steps, width 320, two windows per series) scores 1.217**, about +0.16 over the baseline
  and roughly 9 seed standard deviations. It improves change point (0.93 vs 0.81), autocorrelation
  (0.84 vs 0.80) and anomaly (0.55 vs 0.46). Longer training clearly helps, and 20k steps take only 12 minutes.
- At step 20,000 the direction loss (0.025) and the contrastive loss (0.025) are near zero, so both
  auxiliary tasks are close to solved. More steps may need harder views or a stronger masking ratio.
  This is a question for Tracks 3 and 4.
- The `main` summary records commit `861fc3b`: HEAD moved while the job ran. That commit only added files,
  and the training code is unchanged from `100e7c8`.

## Bug found: non-finite values filled with zero

`ssl_time_series/preprocessing.py::sanitize` replaced NaN/inf with 0. In a series at level 1,000, one
missing value becomes a spike of size 1,000 after normalization, and about 1.4% of corpus values are
non-finite. The repository's own test
`test_normalize_is_finite_for_constant_and_nonfinite_windows` failed for this reason. Fixed on branch
`fix/nonfinite-fill` (`f4f1cb3`, based on `challenge-finance-ssl`): fill with the mean of the window's
finite values, and compute differences on the sanitized window. All 9 tests pass. All results above
use the unfixed code, so they stay comparable with the other tracks until the fix is merged.

## Failure modes and notes

- `evaluate.py` segfaults (exit 139) when onnxruntime-gpu's CUDA preload runs in a process where PyTorch
  is already imported, or on a node without a GPU. Run the evaluator in its own process on a GPU node.
- `torch.onnx.export` warns that the shape check in `derived_channels` is traced as a constant. That is
  expected for the static `[64, 1, 1024]` contract.

## Recommendation

1. Use `main` (`configs/main.json`, seed 0) as the **current best candidate**. Submit
   `outputs/main.onnx` to the leaderboard now (13.53 MiB, all checks pass). The leaderboard refreshes only
   every few days, and this gives us a real score before Tuesday.
2. Use the 2k baseline (score 1.059 ± 0.017) as the Phase 1 comparison point for every track, and
   2k-step runs for ablations (about 2 minutes each).
3. Merge `fix/nonfinite-fill` into `challenge-finance-ssl` at the start of Phase 2, then retrain the
   combined candidate with it.
4. Phase 2: the combined candidate should start from the `main` recipe (20k steps) plus whatever
   components beat the 2k baseline by more than 0.035.

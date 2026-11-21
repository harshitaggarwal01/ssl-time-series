# Phase 2: combined candidates

Branch `research/combined` = `challenge-finance-ssl` + `research/masked-ssl` (Track 3) + `research/finance`
(Track 2) + `fix/nonfinite-fill` + `research/baseline` (Track 1). Track 2 feature groups (`--feature-groups`)
now feed both Track 3 encoders. Run on Oscar (RTX A5000 / A6000), 20 Nov 2026.

## Question

Does adding the Track 2 channels that beat the baseline (momentum, multiscale) to the Track 3 recommended model
(dilated conv + untrained features) improve it?

## Commands

```bash
# from ssl-time-series/ on research/combined; DATA = project dir with corpus/, .venv, outputs/proxy_suite, outputs/ucr
NAME=c-conv       SEED=0 sbatch slurm/train_conv_candidate.sbatch
NAME=c-conv-s1    SEED=1 sbatch slurm/train_conv_candidate.sbatch
NAME=c-conv-mm    SEED=0 EXTRA="--feature-groups momentum multiscale" sbatch slurm/train_conv_candidate.sbatch
NAME=c-conv-mm-s1 SEED=1 EXTRA="--feature-groups momentum multiscale" sbatch slurm/train_conv_candidate.sbatch
sbatch slurm/finance_confirm.sbatch       # transformer baseline + momentum + multiscale, seeds 0/1/2
```

Conv recipe: `train_mae --arch conv --depth 8 --real-only --contrast 1.0 --view crop`, 30,000 steps at batch
512, exported with `--pooling mean_std --fixed`. About 35 to 40 minutes per run. All candidates pass the six
`test_submission.py` checks. (The job logs record commit `0a092e9`: it adds only Slurm scripts on top of
`0f121f4`, where training started.)

## Results

Suite = our 8-task synthetic suite v2.1 (0 = untrained encoder, 1 = official example). UCR = mean ROC AUC on
Track 3's sets (first set chose the recipe; held-out set played no part in any choice).

| model | suite score | suite mean AUC | UCR first | UCR held-out | ONNX |
| --- | ---: | ---: | ---: | ---: | ---: |
| t3-conv (Henry's committed model) | 1.314 | 0.917 | 0.932 | 0.920 | 6.65 MiB |
| c-conv, seed 0 (retrained on Oscar) | 1.317 | 0.917 | 0.932 | 0.919 | 6.90 MiB |
| c-conv, seed 1 | 1.316 | 0.919 | 0.932 | **0.921** | 6.90 MiB |
| c-conv + momentum + multiscale, seed 0 | 1.310 | 0.901 | 0.933 | 0.919 | 7.93 MiB |
| c-conv + momentum + multiscale, seed 1 | 1.316 | 0.900 | 0.935 | 0.917 | 7.93 MiB |
| main + untrained features (transformer) | 1.270 | 0.893 | 0.930 | 0.918 | 17.59 MiB |
| main (Track 1, 20k steps) | 1.217 | 0.847 | 0.859 | 0.845 | 13.53 MiB |
| official example | 1.000 | 0.770 | 0.805 | 0.800 | |

Per-task suite AUC (shape, frequency, trend, volatility, change point, autocorrelation, anomaly, seasonal):

| model | sh | fr | tr | vol | cp | ac | an | se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c-conv | 1.000 | 1.000 | 0.706 | 0.985 | 0.927 | 0.928 | 0.798 | 0.989 |
| c-conv-s1 | 1.000 | 1.000 | 0.701 | 0.987 | 0.928 | 0.933 | 0.813 | 0.989 |
| c-conv-mm | 1.000 | 1.000 | 0.708 | 0.915 | 0.933 | 0.911 | 0.749 | 0.990 |
| c-conv-mm-s1 | 1.000 | 1.000 | 0.704 | 0.912 | 0.953 | 0.907 | 0.734 | 0.991 |

Track 2 confirmation on the 2k-step transformer (suite score, seeds 0/1/2):

| model | seed 0 | seed 1 | seed 2 | mean ± sd |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1.061 | 1.075 | 1.042 | 1.059 ± 0.017 |
| + momentum + multiscale | 1.146 | 1.180 | 1.159 | **1.162 ± 0.017** |

## Findings

1. **Track 3 reproduces.** Retrained from the merged code on different GPUs, the conv candidate matches Henry's
   committed model within 0.003 on every set, and the two seeds agree within 0.002.
2. **Track 2 is a real gain for the transformer**: +0.10 on the suite over three seeds (six baseline standard
   deviations), mostly on frequency (0.99 vs 0.91), trend (0.75 vs 0.70) and seasonal (0.96 vs 0.91).
3. **But it adds nothing on top of the untrained features.** With the conv candidate, momentum + multiscale
   change UCR by +0.002 on the first set and -0.003 on the held-out set (noise), lose 0.07 AUC on volatility and
   0.06 on anomaly in the suite, and add 1 MiB. The untrained spectrum and random-kernel features already carry
   the multi-horizon information those channels add to the transformer.
4. **Trend stays at about 0.70 for every model.** No component tried so far moves it on top of the untrained
   features.

## Decision

Final candidate: **the Track 3 conv model** (`models/candidate-conv.onnx` = `t3-conv`; `c-conv-s1` is
equivalent within noise). Do not add the Track 2 channels to it. Keep the Track 2 result for the presentation as
evidence that finance-style channels help a trained encoder, and that untrained features subsume them.

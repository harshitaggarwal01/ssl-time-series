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

## Leaderboard check (21 Nov 2026)

`t3-conv` (uploaded 20 Nov, 21:54) scored **1.045** on the leaderboard, rank 8 of 17; the top score is 1.372.
Both proxies rated it far above the official example (suite 1.31, UCR 0.93 vs 0.80), so **neither proxy
predicts the hidden tasks**. From here on, only leaderboard scores count as evidence.

Hidden task weights, solved exactly from the 17 leaderboard rows (fit error 4e-7):

| task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weight | 0.15 | 0.05 | 0.05 | 0.05 | 0.15 | 0.15 | 0.075 | 0.075 | 0.075 | 0.075 | 0.05 | 0.05 |
| t3-conv | 1.196 | 1.188 | 1.094 | 1.417 | 1.036 | **0.623** | 1.260 | 1.003 | 1.047 | 0.994 | 1.177 | **0.996** |
| top-5 mean | 1.341 | 1.278 | 1.155 | 1.693 | 1.138 | 1.247 | 1.362 | 1.114 | 1.068 | 1.013 | 1.201 | 1.854 |

We trail the top-5 mean by 0.215. Task 6 accounts for 0.094 of that and task 12 for 0.043. Task 6 swings from
-1.09 to 1.46 across teams, which suggests the untrained floor sits close to the reference there.

Next probe: `c-conv-s1-nofixed` (`slurm/leaderboard_probes.sbatch`), the same conv encoder without the untrained
features (640 dimensions, 2.85 MiB). Suite 1.382, UCR 0.911 / 0.910. It isolates whether the 1,925 untrained-feature
dimensions hurt the hidden tasks, since the evaluator standardizes every dimension before the ridge probe. The
same job builds `main-mm` (main + momentum + multiscale; suite 1.284, UCR 0.862 / 0.856) as a reserve.

## Window statistics block (`--stats`, 21 Nov)

`ssl_time_series/stats.py`: 49 untrained statistics (level and scale, real length, integer / non-positive share,
distribution shape, trend, autocorrelation, and GS Quant-style indicators: RSI, Bollinger, drawdown, streaks,
rolling volatility, percentile rank). Exported from existing checkpoints by `slurm/export_candidates.sbatch`; all
pass `test_submission.py`.

| model | dims | suite | UCR first | UCR held-out | ONNX |
| --- | ---: | ---: | ---: | ---: | ---: |
| t3-conv (leaderboard 1.045) | 1925 | 1.314 | 0.932 | 0.920 | 6.65 MiB |
| c-conv-s1-nofixed | 640 | 1.382 | 0.911 | 0.910 | 2.85 MiB |
| conv-stats | 689 | 1.377 | 0.918 | 0.915 | 3.82 MiB |
| **conv-fixed-stats** | 1974 | 1.315 | **0.934** | **0.923** | 7.86 MiB |
| conv2-stats (two conv encoders) | 1329 | 1.376 | 0.921 | 0.915 | 6.69 MiB |
| conv-mainmm-stats (conv + transformer) | 1329 | 1.358 | 0.915 | 0.913 | 18.15 MiB |

Both proxies remove scale information by design (random rescaling; mostly z-normalized UCR), so they cannot reward
the scale part of the block. Even so it adds 0.005 to 0.007 UCR AUC to either base model and 0.05 to 0.07 AUC on
the anomaly task. Its value on the hidden tasks has to come from the leaderboard.

## Leaderboard probe: untrained features off (22 Nov)

`c-conv-s1-nofixed` (uploaded 21 Nov, 14:33) scored **1.088**, up from 1.045 for `t3-conv`. The leaderboard lists
task columns in string order (task1, task10, task11, task12, task2, ...); with the weights above, the columns
reproduce the overall score exactly.

| task (weight) | 1 (.15) | 2 (.05) | 3 (.05) | 4 (.05) | 5 (.15) | 6 (.15) | 7 (.075) | 8 (.075) | 9 (.075) | 10 (.075) | 11 (.05) | 12 (.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t3-conv (with untrained) | 1.196 | 1.188 | 1.094 | 1.417 | 1.036 | 0.623 | 1.260 | 1.003 | 1.047 | 0.994 | 1.177 | 0.996 |
| nofixed | **1.366** | 1.212 | **1.645** | **1.902** | 0.970 | 0.537 | 1.253 | 1.005 | 0.971 | 0.924 | 1.070 | 1.083 |
| change | +0.170 | +0.024 | +0.551 | +0.485 | -0.066 | -0.086 | -0.007 | +0.002 | -0.076 | -0.070 | -0.107 | +0.087 |

The untrained features are not uniformly bad. They cost tasks 1, 3, 4 and 12 a lot, and help tasks 5, 6, 9, 10 and 11 a
little. Task 6 stays the weakest at 0.54 either way, so it needs something neither block has. Next upload:
`conv-stats` (nofixed + window statistics), which isolates the statistics block.

## Strategy for the last two evaluations (22 Nov)

Facts from the public leaderboard data (`docs/leaderboard.json` in the challenge repo and its 19 daily revisions):

- Only the latest upload per participant is evaluated, once a day at about 04:00-05:00 UTC. Before the
  presentation that leaves two evaluations: Sunday night and Monday night.
- We are now rank 10 (1.088). To pass 1.372 we need about +0.14 on task 6, +0.06 on task 5 and +0.05 on task 12;
  on tasks 1, 3 and 4 we are already at or above the leader.
- Across 44 evaluated submissions, the tasks fall into clusters: {5, 6, 10, 7} (5-6 correlation 0.82), {8, 9}
  (0.93), {11, 12} (0.70), {3, 4} (0.64), with 1 and 2 general.

Strategies and the evidence for each:

| # | strategy | targets | source | status |
| --- | --- | --- | --- | --- |
| 1 | fewer, better dimensions (the probe standardizes all of them; alpha <= 1000) | 1, 3, 4 | leaderboard: dropping 1,285 untrained dims gave +0.043 | done (`nofixed`) |
| 2 | level, scale and length statistics plus GS Quant indicators | 11, 12, 6 | example and floor normalize per series, so scale is free signal | `--stats` |
| 3 | position profile (PAA, segment spread and step size) | 5, 6, 10 | untrained floor close to reference on task 6, typical of aligned data | `--profile` |
| 4 | max pooling over time (TS2Vec instance readout) | 5, 6, 10 | TS2Vec; keeps short events that mean pooling blurs | `mean_max_std` |
| 5 | only the spectrum part of the untrained features | 5, 6, 10 | untrained features helped 5, 6, 10 slightly; the kernels are 90% of their size | `--fixed-parts` |

Plan: Sunday upload = the combination with the fewest added dimensions that carries 2 and 3
(`conv-stats-prof`, 737 dims). The task clusters separate the blocks in the per-task scores (stats -> 11/12,
profile -> 5/6/10). Monday upload = Sunday's file plus whatever its per-task scores support, or `nofixed` back if
Sunday is lower than 1.088.

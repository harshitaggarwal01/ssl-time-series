# Track 4 report: contrastive learning

Branch `research/contrastive-ssl`, from `research/combined` at `ed924e6`.

## Question

The leaderboard model (conv-stats-prof, 1.187) trains its encoder with masked reconstruction **plus** InfoNCE
on two crops. Contrast was never studied on its own. Does a purely contrastive objective, with better
augmentations or a TS2Vec-style hierarchical loss, train a better encoder?

## Variants

All three use the Track 3 conv encoder (depth 8, real-only), 30,000 steps at batch 512, seed 0. They have no
reconstruction loss, and 40% span masking stays on as an augmentation. Only the objective changes.

| name | objective | flags |
| --- | --- | --- |
| `ct-infonce` | InfoNCE (t = 0.1) on the pooled mean+max of two 80-100% crops | `--reconstruction 0` |
| `ct-aug` | as above + scale-aware jitter (5% of window std), magnitude warp (0.2), crops 50-100% | `--reconstruction 0 --jitter 0.05 --warp 0.2 --crop-low 0.5` |
| `ct-ts2vec` | TS2Vec hierarchical instance + temporal contrast on two aligned prefix-dropped views | `--reconstruction 0 --objective hierarchical --crop-low 0.5` |

Reference: `ct-ref` retrains the team's recipe (recon + InfoNCE) with the same job, seed 0. Its `-enc` export
corresponds to `c-conv-s1-nofixed` (leaderboard 1.088), and its `-stats-prof` export to `conv-stats-prof`
(leaderboard 1.187). Track 3 found that seeds differ by at most 0.002.

The augmentations act on the real part of each window relative to its own mean and std. Raw scaling and offsets
would be no-ops, because the model normalizes each window internally. The left padding is rewritten to the new
first value, so `real_points` still finds it.

## Commands (Oscar)

```bash
cd ssl-time-series            # on research/contrastive-ssl
mkdir -p logs
sbatch slurm/setup_track4.sbatch          # once: env, corpus, proxy suite, UCR sets (CPU, ~1-2 h)
bash slurm/contrastive_variants.sh        # 4 GPU jobs: ct-ref + 3 variants
.venv/bin/python -m scripts.compare_eval
```

Each job writes `$NAME-enc.onnx` (compare with `c-conv-s1-nofixed`) and `$NAME-stats-prof.onnx` (compare with
`conv-stats-prof`; this is the upload candidate), runs `test_submission.py`, the synthetic suite and both UCR sets.

## Segment pooling (tests the team's task 6 hypothesis)

Every other export averages the encoder's features over the whole window, which discards *where* events happen.
`--pooling seg2_std` / `seg4_std` (conv only) keep the mean of each of 2 or 4 fixed time segments plus the
window-wide spread (960 / 1600 dims; 1057 / 1697 with stats + profile). No retraining:

```bash
sbatch slurm/export_segments.sbatch          # ct-ts2vec and ct-ref, 2 and 4 segments
.venv/bin/python -m scripts.compare_eval ct-ts2vec-seg2-stats-prof ct-ts2vec-seg4-stats-prof \
    ct-ref-seg2-stats-prof ct-ref-seg4-stats-prof
```

Caveat: the leaderboard showed that extra dimensions can hurt (dropping 1,285 untrained dims gave +0.043), so
watch the larger seg4 export closely.

## Selection rule (one leaderboard upload)

The proxies did not predict the leaderboard (COMBINED_REPORT). Choose on the encoder-only exports: the best
UCR held-out AUC, ties broken by the suite, and compare it with `ct-ref-enc`. Then upload that variant's `-stats-prof` file. If no variant beats
`ct-ref-enc` on UCR held-out, upload the best one anyway and report it as a negative result.

## Results (23 Nov 2026, Oscar, seed 0)

Suite = synthetic suite v2.1 score (0 = untrained, 1 = official example). UCR = mean ROC AUC (official
`evaluate.py`) on the first set (used for Track 3 choices) and the held-out set (used for no choices). Seed noise on
UCR is about 0.002 (Track 3). All exports pass `test_submission.py`.

Encoder only (640 dims; isolates the objective):

| model | objective | suite | UCR first | UCR held-out | effective rank |
| --- | --- | ---: | ---: | ---: | ---: |
| c-conv-s1-nofixed (team, leaderboard 1.088) | recon + InfoNCE | 1.382 | 0.911 | 0.910 | |
| ct-ref-enc | recon + InfoNCE | **1.385** | 0.914 | **0.910** | |
| ct-infonce-enc | InfoNCE only | 1.378 | 0.909 | 0.906 | 68 |
| ct-aug-enc | InfoNCE + jitter / warp / wide crops | 1.325 | **0.921** | 0.907 | 63 |
| ct-ts2vec-enc | TS2Vec hierarchical | 1.027 | 0.889 | 0.871 | 43 |

Upload form (+ window statistics + position profile, 737 dims):

| model | suite | UCR first | UCR held-out |
| --- | ---: | ---: | ---: |
| conv-stats-prof (team, leaderboard 1.187) | 1.360 | 0.923 | 0.916 |
| ct-ref-stats-prof | **1.365** | 0.920 | **0.916** |
| ct-infonce-stats-prof | 1.356 | 0.916 | 0.913 |
| ct-aug-stats-prof | 1.318 | **0.928** | 0.915 |
| ct-ts2vec-stats-prof | 1.174 | 0.913 | 0.892 |

Segment pooling (upload form with `seg2_std` / `seg4_std`, 1057 / 1697 dims):

| model | suite | change point AUC* | trend* | UCR first | UCR held-out |
| --- | ---: | ---: | ---: | ---: | ---: |
| ct-ref-stats-prof | 1.365 | 1.687 | 1.161 | 0.920 | 0.916 |
| ct-ref-seg2-stats-prof | 1.360 | 1.702 | 1.103 | 0.919 | 0.916 |
| ct-ref-seg4-stats-prof | 1.362 | 1.710 | 1.109 | 0.916 | 0.916 |
| ct-aug-stats-prof | 1.318 | 1.557 | 1.117 | 0.928 | 0.915 |
| ct-aug-seg2-stats-prof | 1.305 | 1.598 | 1.010 | 0.926 | 0.914 |
| ct-aug-seg4-stats-prof | 1.317 | 1.695 | 0.996 | 0.923 | 0.914 |

\* Suite task scores (1 = official example). Change point = early vs late mean shift, the only suite task where
position in the window is the label.

## Findings

1. **Contrast carries most of the encoder.** Dropping reconstruction (`ct-infonce`) costs 0.004-0.005 UCR AUC and
   0.007 on the suite, about twice the seed noise. Reconstruction adds a small, consistent bonus.
2. **Scale-aware augmentations trade one kind of information for another.** `ct-aug` gains up to +0.15 AUC on
   waveform-shape datasets (ChlorineConcentration, PigAirwayPressure, ECG5000, ECG200, FordA/B). It loses on
   level, on/off and trend datasets (RefrigerationDevices, ElectricDevices, PowerCons, InlineSkate) and on the
   suite's trend, change-point and autocorrelation tasks. It is best on the first UCR set (+0.008) and tied on
   held-out (-0.001), so this is a trade-off, not an overall gain.
3. **The naive TS2Vec views allow a positional shortcut.** Both views kept the window end fixed, so matching
   timesteps was solvable from position alone. The loss fell to 0.03, the effective rank was the lowest (43), and
   content tasks collapsed (frequency 1.26 -> 0.82, seasonal 1.24 -> 0.77). A fair test needs overlapping crops at
   different offsets, as in the original TS2Vec.
4. **Segment pooling keeps position information without hurting the proxies.** UCR moves by at most 0.005
   (mostly not aligned data), but change point, the one position task, rises for both encoders (ct-aug: 1.557 ->
   1.695 with 4 segments), at a cost on trend. Only the leaderboard can test whether this helps hidden task 6.

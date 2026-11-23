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

References (already trained and scored by the team): `c-conv-s1-nofixed` (recon + InfoNCE, encoder only,
leaderboard 1.088) and `conv-stats-prof` (same checkpoint + stats + profile, leaderboard 1.187).

The augmentations act on the real part of each window relative to its own mean and std. Raw scaling and offsets
would be no-ops, because the model normalizes each window internally. The left padding is rewritten to the new
first value, so `real_points` still finds it.

## Commands (Oscar)

```bash
cd ssl-time-series            # on research/contrastive-ssl
mkdir -p logs
DATA=<team project dir> bash slurm/contrastive_variants.sh      # 3 GPU jobs
DATA=<team project dir> $DATA/.venv/bin/python -m scripts.compare_eval
```

Each job writes `$NAME-enc.onnx` (compare with `c-conv-s1-nofixed`) and `$NAME-stats-prof.onnx` (compare with
`conv-stats-prof`; this is the upload candidate), runs `test_submission.py`, the synthetic suite and both UCR sets.

## Selection rule (one leaderboard upload)

The proxies did not predict the leaderboard (COMBINED_REPORT). Choose on the encoder-only exports: the best
UCR held-out AUC, ties broken by the suite. Then upload that variant's `-stats-prof` file. If no variant beats
`c-conv-s1-nofixed` on UCR held-out, upload the best one anyway and report it as a negative result.

## Results

_To fill in after the Oscar runs._

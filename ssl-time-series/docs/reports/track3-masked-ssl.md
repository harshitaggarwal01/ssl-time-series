# Track 3 report: masked self-supervised learning

2026-11-19, branch `research/masked-ssl`. Step-by-step reproduction:
[`notebooks/masked_ssl_walkthrough.ipynb`](../../notebooks/masked_ssl_walkthrough.ipynb).

## Findings

1. **Tuning the mask does not matter.** Mask shape, ratio (25 to 60%), target, patch size, three times the
   steps and twice the parameters all score within seed noise. Only 10% masking and a difference-only target
   are clearly worse.
2. **Most of the score comes from untrained features.** Spectrum bands and random dilated kernels appended to
   the embedding (`fixed.py`) score 0.928 with no training at all. The trained encoder adds at most 0.008 on
   top.
3. **The shared synthetic proxy cannot rank models**: it gives 1.000 to every trained model. All numbers here
   are mean ROC AUC from the official `evaluate.py` on UCR datasets (`scripts/proxy_ucr.py`).
4. **Recommended candidate: a dilated convolutional encoder**, 0.48M parameters, 6.65 MiB, 1,925 dimensions,
   passes `test_submission.py`. The seed 0 model is in the repository as
   [`models/candidate-conv.onnx`](../../models/candidate-conv.onnx).

## Results

Every choice was made on the first set (19 UCR datasets). The other three sets played no part in the choices:
held-out (22 other datasets), variable length (11 datasets with series of differing length), and stress (the
first set with series cut short, rescaled and shifted).

| Model | First | Held-out | Stress | Variable length | ONNX |
| --- | --- | --- | --- | --- | --- |
| Untrained encoder | 0.793 | 0.808 | 0.719 | 0.813 | |
| Organizers' example model | 0.805 | | | | |
| Masked encoder, mean pooling | 0.889 | 0.883 | | | 13.5 MiB |
| Untrained features only, no encoder | 0.928 | 0.915 | 0.867 | 0.961 | |
| Masked encoder + untrained features | 0.932 | 0.919 | 0.867 | 0.959 | |
| Transformer candidate | **0.936** | 0.918 | 0.873 | 0.962 | 17.6 MiB |
| **Convolutional candidate (recommended)** | 0.933 | **0.921** | **0.876** | **0.966** | 6.65 MiB |

Candidates are the mean of seeds 0 and 1; the two seeds differ by at most 0.002. Both candidates are trained
with masked reconstruction plus a contrastive term on two crops of each window, normalize and pool over the
series rather than its padding, and include the untrained features. The convolutional candidate is recommended
because it leads on the three sets that were not used for choices, at about a third of the file size.

## Reproduce

Follow walkthrough notebook: *masked_ssl_walkthrough.ipynb*

Defaults: 30,000 steps, batch 512, AdamW at 5e-4, 40% masking, seed 0. About 30 to 45 minutes on an RTX 4070.

## What did not help

- Latent (EMA teacher) targets and bottleneck decoders: 0.01 to 0.03 worse.
- Contrast on stretched instead of cropped views: no gain on the held-out set.
- More steps (60,000), a wider convolutional encoder, a larger transformer (also exceeds 20 MiB): within noise.
- Reducing dimensions with PCA, more or different random kernels, two encoders side by side: within noise or
  worse.

## Limitations

- **The proxy is UCR.** Nothing has been checked against the official leaderboard, whose hidden tasks span
  several domains.
- **With the untrained features, every good model lands within 0.009 of the others.** The proxy cannot separate
  them further.
- One saturated held-out task (ItalyPowerDemand) falls slightly below the untrained encoder, 0.985 against
  0.991. The official score is normalized by that floor, so such losses cost more there.
- Each candidate has two seeds. The convolutional encoder was tried late and its depth, stride and masking were
  not varied.

# Team Research Plan

This document describes how we will work from the shared implementation branch
through independent research tracks and the final combined model.

## Current shared starting point

The shared branch is:

```text
challenge-finance-ssl
```

It was created on top of the latest `main` branch and is pushed to:

<https://github.com/harshitaggarwal01/ssl-time-series/tree/challenge-finance-ssl>

The branch currently includes:

- Memory-mapped corpus loading and deterministic window sampling.
- Per-window finite-value handling and normalization.
- Derived level, difference, and magnitude channels.
- A compact patch-transformer encoder.
- Masked reconstruction, contrastive, and next-step training objectives.
- Checkpoint training and encoder-only ONNX export.
- Corpus profiling and synthetic proxy evaluation.
- Baseline and main training configurations.
- Focused tests for preprocessing, sampling, model behavior, and export.
- Documentation, references, citations, and the implementation checklist.

The challenge contract remains unchanged:

```text
Input:  float32 [64, 1, 1024]
Output: float32 [64, D], where D <= 2048
ONNX opset: >= 17
File size: < 20 MiB
```

Training must use only the official challenge corpus. Stable Finance and GS
Quant are references for financial time-series ideas; they are not additional
training data or sources of pretrained weights.

## Phase 1: independent end-to-end research

The purpose of Phase 1 is to compare complete approaches. Each person should
choose one track and take it from research through implementation, training,
export, validation, and results. Do not combine partial implementations yet.

Each track should begin from `challenge-finance-ssl`, but experimental work
must happen on a separate branch.

Suggested branch names:

```text
research/baseline
research/finance
research/masked-ssl
research/contrastive-ssl
research/architecture
```

### Track 1: baseline reproduction

**Owner:** project lead

The goal is to establish a reliable reference point before comparing new ideas.

Work:

1. Reproduce the existing next-step direction baseline.
2. Train the current encoder on the corpus.
3. Export the encoder to ONNX.
4. Run `test_submission.py`.
5. Run the synthetic proxy evaluation.
6. Measure model size and throughput.
7. Record the exact command and configuration.

This result becomes the reference for all other tracks.

### Track 2: finance-inspired representation

Research whether financial time-series transformations improve general-purpose
representations, even though the hidden evaluation includes multiple domains.

Ideas to test:

- Normalized levels.
- First differences.
- Log differences where numerically valid.
- Multi-scale returns.
- Rolling volatility.
- Reversal and momentum-like features.
- Local scale and offset robustness.

Use the following as research references:

- Stable Finance causal feature construction:
  <https://github.com/galilai-group/stable-finance/blob/main/experiments/month_cache.py>
- Stable Finance signal normalization:
  <https://github.com/galilai-group/stable-finance/blob/main/experiments/signals.py>
- GS Quant return utilities:
  <https://github.com/goldmansachs/gs-quant/blob/master/gs_quant/timeseries/econometrics.py>
- GS Quant technical indicators:
  <https://github.com/goldmansachs/gs-quant/blob/master/gs_quant/timeseries/technicals.py>

All transformations must remain compatible with unlabeled, timestamp-free
challenge windows.

### Track 3: masked self-supervised learning

Research masked reconstruction as the main learning signal.

Ideas to test:

- Random patch masking.
- Contiguous span masking.
- Mask ratios such as 10%, 25%, and 40%.
- Reconstruction of normalized levels.
- Reconstruction of differences.
- Patch sizes 16, 32, and 64.

The decoder and reconstruction head are training-only. The submitted ONNX file
must contain only the encoder.

### Track 4: contrastive learning and augmentations

Research whether two augmented views of the same window produce embeddings
that transfer better to unseen tasks.

Ideas to test:

- Amplitude scaling.
- Additive offsets.
- Small Gaussian noise.
- Local variance changes.
- Patch dropout.
- Mild masking.
- Random crops for long series.
- Different augmentation strengths.

Augmentations must preserve enough temporal structure to avoid teaching the
encoder invariance to useful direction, frequency, or regime information.

### Track 5: architecture and efficiency

Research the architecture while keeping the objective fixed.

Ideas to test:

- Patch sizes 16, 32, and 64.
- Embedding widths 128, 256, and 320.
- Different transformer depths.
- Mean pooling versus attention pooling.
- Convolutional stems versus direct patch projection.
- CPU throughput.
- ONNX file size.
- Operator compatibility with ONNX Runtime.

The architecture track must measure both quality and deployment constraints.
A model with a higher proxy score is not useful if it exceeds the submission
size limit or fails the evaluator.

## Common requirements for every track

Every researcher must:

1. Start from the shared `challenge-finance-ssl` branch.
2. Use only the official corpus.
3. Use the same initial series-level split.
4. Use a fixed seed, starting with seed `0`.
5. Export an encoder-only ONNX model.
6. Run the official local submission checks.
7. Run the same proxy evaluation suite.
8. Keep corpus data, checkpoints, and generated ONNX files out of Git.
9. Record the exact command used to train and export.
10. Report both successes and failed experiments.

Required validation command:

```bash
uv run test_submission.py outputs/<candidate>.onnx
```

Required result fields:

```text
track
hypothesis
configuration
seed
training steps
training losses
proxy-task scores
mean proxy score
ONNX size
embedding dimension
throughput
validation status
known limitations
recommendation
```

## Suggested experiment report

Each researcher should add a report to their branch or attach it to the pull
request using this structure:

```markdown
# Track report

## Hypothesis

## Changes

## Configuration

## Training command

## Export command

## Validation command and result

## Proxy results

## Model size and throughput

## What worked

## What did not work

## Recommendation
```

Do not compare results from different data splits, seeds, or proxy task
definitions without clearly identifying the difference.

## Phase 2: combine findings

We will reconnect around Friday to compare the independent results.

The meeting should answer:

1. Which approach improved the baseline most consistently?
2. Which improvements worked across multiple proxy tasks?
3. Which components are compatible?
4. Does any candidate fail the ONNX size or speed constraints?
5. Are gains stable across more than one seed?
6. Which components add complexity without measurable benefit?

We should not merge entire research branches automatically. Instead, select
components supported by evidence:

- Best preprocessing.
- Best augmentation set.
- Best objective.
- Best architecture.
- Best embedding width.
- Best patch size.

Then create three combined candidates:

```text
combined-minimal
combined-balanced
combined-maximal
```

### Combined minimal

The strongest simple components with the lowest implementation and deployment
risk.

### Combined balanced

The strongest score, size, speed, and stability tradeoff.

### Combined maximal

All compatible improvements that still satisfy the challenge contract and the
20 MiB limit.

## Final validation after combining

For each combined candidate:

1. Train with seeds `0`, `1`, and `2`.
2. Run all proxy tasks.
3. Export ONNX.
4. Check the file size.
5. Run `test_submission.py`.
6. Measure throughput.
7. Compare against all Phase 1 candidates.

Suggested final commands:

```bash
uv lock --check
python -m compileall -q ssl_time_series scripts tests
uv run test_submission.py outputs/final.onnx
git diff --check
git status --short
```

## Final model selection

Select the candidate that:

1. Passes all submission checks.
2. Is below 20 MiB.
3. Has the strongest average proxy-task score.
4. Is stable across seeds.
5. Uses reliable ONNX Runtime operators.
6. Has acceptable inference speed.
7. Is no more complex than necessary.

Do not select a model from a single proxy task or one unusually good random
seed.

## Git and collaboration workflow

Create a research branch from the shared branch:

```bash
git switch challenge-finance-ssl
git pull --ff-only
git switch -c research/<track-name>
```

Keep commits focused and descriptive. Do not commit:

- `corpus/`
- `.venv/`
- Checkpoints such as `*.pt`.
- Generated models such as `*.onnx`.
- Large profiling or experiment artifacts.

When the track is complete, open a pull request containing:

- The implementation.
- Tests or validation evidence.
- The experiment report.
- The recommended configuration.
- Any known limitations.

After review, the lead will merge selected changes or cherry-pick only the
validated parts into the combined branch.

## References

- Stable Challenge repository:
  <https://github.com/galilai-group/stable-challenge-time-series>
- Stable Challenge specification:
  <https://github.com/galilai-group/stable-challenge-time-series/blob/main/CHALLENGE.md>
- Stable Challenge evaluator:
  <https://github.com/galilai-group/stable-challenge-time-series/blob/main/evaluate.py>
- Stable Finance:
  <https://github.com/galilai-group/stable-finance>
- Goldman Sachs GS Quant:
  <https://github.com/goldmansachs/gs-quant>

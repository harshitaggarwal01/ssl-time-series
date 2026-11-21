# Status and plan to the final presentation (Tue 24 Nov 2026)

Last updated: Fri 20 Nov 2026. Supersedes the Phase 1/Phase 2 timing in
[TEAM_RESEARCH_PLAN.md](TEAM_RESEARCH_PLAN.md); the track definitions there still apply.

## Where each track stands

| Track | Owner | Branch | Status (20 Nov) |
| --- | --- | --- | --- |
| 1 Baseline reproduction | Harshit | `research/baseline` | **Done.** 2k baseline 1.059 ± 0.017 (3 seeds); `main` 20k **1.217**; official example = 1.000. All pass `test_submission.py`. [BASELINE_REPORT.md](BASELINE_REPORT.md) |
| 2 Finance features | Frank | `research/finance` | Code pushed (feature groups `multiscale`, `log_returns`, `volatility`, `momentum`, `scale`; resumable checkpoints). No results yet. |
| 3 Masked SSL | — | none pushed | Not started on GitHub. |
| 4 Contrastive / augmentations | — | none pushed | Not started on GitHub. |
| 5 Architecture / efficiency | — | none pushed | Not started on GitHub. |

## What changed for everyone: evaluation protocol v2

The original `scripts/proxy_evaluate.py` gives **macro AUC 1.000 for the 2k-step baseline**: its four
synthetic shapes are too easy, so it cannot rank candidates. Use the new suite instead. It runs the
**official `evaluate.py`** (same ridge probe, window pooling and score formula as the leaderboard) on
8 synthetic tasks:

| Task | Classes | Tests |
| --- | ---: | --- |
| shape | 4 | ramp / two sine frequencies / step (the old proxy) |
| frequency | 5 | sine period with phase and noise |
| trend | 3 | random walk drift down / none / up |
| volatility_regime | 3 | constant vs. one break vs. GARCH-style clustering |
| change_point | 3 | none / early / late mean shift |
| autocorrelation | 4 | AR(1) with phi in {-0.7, 0, 0.5, 0.95} |
| anomaly | 2 | AR(1) with or without rare spikes |
| seasonal_long | 3 | 2-4 windows per item; no / daily-like / weekly-like seasonality |

Every item gets a random scale (1e-2 to 1e3) and offset (±1e3), as the evaluator's raw windows do.
Anchors follow the leaderboard's definition: **score 0 = untrained encoder, 1 = official
`example_submission/model.onnx`**. A score above 1 on a task means we beat the official example there.

```bash
# from ssl-time-series/ssl-time-series, once:
uv run python -m scripts.build_proxy_suite --out outputs/proxy_suite
# per candidate:
uv run evaluate.py outputs/<candidate>.onnx outputs/proxy_suite --out outputs/eval/<candidate>.json
```

On Oscar the Slurm scripts in `slurm/` do this end to end (submit from `ssl-time-series/ssl-time-series`,
after `mkdir -p logs`):

```bash
NAME=<candidate> CONFIG=configs/baseline.json SEED=0 EXTRA="--feature-groups volatility" \
  sbatch slurm/train_candidate.sbatch                  # ~4 min for 2k steps on an A5000
sbatch slurm/eval_models.sbatch outputs/<candidate>.onnx
```

The suite is a proxy. It checks generic temporal structure; it does not replace the hidden tasks.

## Rules for comparable Phase 1 results (unchanged, now concrete)

- Start from `challenge-finance-ssl`; your branch is `research/<track>`.
- 2,000 steps, seed 0, `configs/baseline.json` architecture unless architecture is your variable.
- Report seeds 0, 1 and 2 for your best variant. The baseline's seed spread is the noise floor:
  a gain smaller than that is not a gain.
- **The bar to beat:** the 2k baseline scores **1.059 ± 0.017** on suite v2.1 (mean AUC 0.809). A
  variant needs a score of about **1.095 or more** before we treat it as a real improvement.
- Tasks where the official example does not beat random weights (volatility_regime, anomaly) have
  weight 0 in the score. Report their AUC, because volatility is where our channels help most.
- Report: proxy-suite score and per-task AUC, ONNX size, `test_submission.py` output, exact command.
- Add your row to the results table in [BASELINE_REPORT.md](BASELINE_REPORT.md) or send it to Harshit.

## Integration notes for Track 2 (finance)

- `research/finance` changes `--output outputs/X.pt` to write `outputs/X/step-NNNNNN.pt`. Notebook cell 8,
  the README quick start and `slurm/train_candidate.sbatch` export from `outputs/X.pt`. When merging,
  either also write the final checkpoint to `outputs/X.pt` or update all three callers.
- `log_returns` is zero wherever a value is not positive. With the evaluator's arbitrary offsets many
  windows will be negative, so this group may carry little signal; the ablation will show it.
- `scale` feeds the raw log-scale into the model on purpose (scale can be informative). It is the only
  group that breaks scale invariance, so read its result separately.

## Timeline

| When | What | Who |
| --- | --- | --- |
| Fri 20 Nov | Baseline + seed spread + main (20k) results; protocol v2 shared; this plan sent | Harshit |
| **Sat 21 Nov, evening** | **Each track pushes its branch + one results row** (protocol v2, seed 0) | everyone |
| Sat 21 Nov | Submit the best validated model so far to the leaderboard. It refreshes only every few days, so this is our one chance to show a real score on Tuesday. | Harshit |
| Sun 22 Nov | Phase 2 combine: merge only components that beat the baseline spread; train `combined-minimal` and `combined-balanced` at 2k steps x 3 seeds, then the winner at 20k | Harshit + Frank, others review |
| Mon 23 Nov | Final model: `test_submission.py`, proxy suite, submit if it beats the Saturday submission (only the latest submission counts). Freeze results; build slides | everyone |
| Tue 24 Nov | Final presentation | everyone |

If a track has no results by Saturday evening, Harshit can run its configuration on Oscar. Push the
branch with a config or `EXTRA` flags and the command.

## Presentation outline (draft)

1. Challenge: corpus-only SSL, ONNX contract, linear-probe scoring.
2. Data: corpus profile (lengths, non-finite values, scale range).
3. Shared pipeline: in-model normalization, derived channels, patch transformer, three SSL losses.
4. Evaluation: why the first proxy saturated; the calibrated 8-task suite.
5. One slide per track: hypothesis, what changed, result vs. baseline spread.
6. Combined model and leaderboard result.
7. Limits and next steps.

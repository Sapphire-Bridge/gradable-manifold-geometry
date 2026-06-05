# Gradable v1.2 Behavior Runbook

## What Changed

v1.2 keeps the v1.1 explicit-standard prompt format but fixes the pair-orientation artifact. Expected ordered-label shifts are now balanced:

| domain | pairs | expected shift signs |
| --- | ---: | --- |
| temperature | 84 | positive 44, negative 40 |
| size | 85 | positive 42, negative 43 |
| age | 72 | positive 36, negative 36 |

All prompt sides contain the target span exactly once.

## One Command

Run this from the repo root:

```bash
export TOKENIZERS_PARALLELISM=false
DEVICE=mps TORCH_DTYPE=float32 bash scripts/run_gradable_v1_2_behavior.sh
```

Outputs:

```text
results/manifold_groups_poc/gradable_temperature_v1_2_behavior_gemma3.sides.csv
results/manifold_groups_poc/gradable_temperature_v1_2_behavior_gemma3.pairs.csv
results/manifold_groups_poc/gradable_temperature_v1_2_behavior_gemma3.summary.json
results/manifold_groups_poc/gradable_size_v1_2_behavior_gemma3.sides.csv
results/manifold_groups_poc/gradable_size_v1_2_behavior_gemma3.pairs.csv
results/manifold_groups_poc/gradable_size_v1_2_behavior_gemma3.summary.json
results/manifold_groups_poc/gradable_age_v1_2_behavior_gemma3.sides.csv
results/manifold_groups_poc/gradable_age_v1_2_behavior_gemma3.pairs.csv
results/manifold_groups_poc/gradable_age_v1_2_behavior_gemma3.summary.json
results/manifold_groups_poc/gradable_v1_2_behavior_recompute.json
results/manifold_groups_poc/gradable_v1_2_behavior_recompute.md
results/manifold_groups_poc/gradable_v1_2_behavior_recompute.sides.csv
results/manifold_groups_poc/gradable_v1_2_behavior_recompute.pairs.csv
```

## Faster Recompute Only

After the scoring files exist, recompute without re-running the model:

```bash
python scripts/recompute_gradable_behavior_metrics.py \
  --version "v1_2" \
  --bootstrap_B 5000 \
  --permutation_B 20000 \
  --out_json "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.json" \
  --out_md "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.md" \
  --out_sides_csv "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.sides.csv" \
  --out_pairs_csv "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.pairs.csv"
```

## Readout

Primary behavioral evidence is:

```text
side_metrics_unique.corr_ordered_score_vs_predictor.r
side_metrics_unique.corr_ordered_score_vs_predictor.bootstrap_prompt95
side_metrics_unique.corr_ordered_score_vs_predictor.permutation_p_two_sided
side_metrics_unique.corr_ordered_score_vs_predictor.within_target_permutation_p_two_sided
orientation.expected_shift_sign_counts
shift_direction_match.rate
```

Use `side_metrics_unique` as primary because v1.1/v1.2 pair rows reuse prompt sides. Treat `side_metrics_expanded` and `shift_direction_match` as supporting diagnostics.

## Interpretation Gate

Go to paraphrase/artificial-norm sweeps if:

```text
orientation has both positive and negative expected shifts in each domain
r(score,predictor) >= 0.40 in all three domains
bootstrap CI excludes 0 in all three domains
within-target permutation p remains small in at least two domains
```

No-Go or redesign if:

```text
r(score,predictor) < 0.20 in two domains
or the balanced v1.2 orientation destroys the direction effect
or one label prior fully explains the unique-side correlation
```

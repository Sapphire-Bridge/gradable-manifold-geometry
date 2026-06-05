# Gradable Size Calibration Dataset Note

## Purpose

This note records the current dataset and analysis story for the gradable-predicate manifold pilot. It is intended as a short reference surface for later writeups, not as a final paper claim.

## Ten-Sentence Summary

1. The project began from the idea that Bierwisch-style gradable predicates are plausible manifold candidates, but not automatically neural manifolds.
2. The initial philosophical example was `warm/cold`, but prompt-variation experiments made temperature and age unstable, while the `size` domain remained the most robust empirical target.
3. Early size datasets showed positive behavior for natural size contexts and iso-ratio controls, but the bare neutral condition with `Class A/B objects` collapsed.
4. This suggested that Gemma-3-4B was not computing context-free `log(value / standard)` arithmetic, but needed semantic scaffolding: dimension, relatum, comparison class, norm, and construction.
5. Base-prior diagnostics showed that the readout was itself a major confound: `feels` strongly favored `small`, while normality and comparative labels often favored `above normal` or `longer than normal`.
6. The best readout compromise was therefore `adjective_counts`: the ordered labels `tiny/small/large/huge`, but with an explicit standard and the construction `counts as`.
7. Under this readout, the fictional-semantic dataset with invented `dax rods` showed a robust relationship between ordered label mass and `log(value / standard)`.
8. The iso-ratio version also remained positive, which argues against a pure absolute-value heuristic.
9. Raw patching at the target span and early final-token sites was weak, but a later-layer final-token sweep showed strong donor-directed causal effects after the replacement source was corrected to exact decoder-block outputs.
10. The current status is therefore not a manifold proof, but the first strong behavior-plus-causal candidate for the manifold-groups hypothesis: a k=5 low-rank subspace transfers bidirectionally between fictional-semantic and iso-ratio size-calibration variants at L20 while beating random, norm-matched random, value, standard, and value+standard controls.

## Dataset Family

The central dataset family is written under:

```text
results/manifold_groups_poc/gradable_size_disamb_pairs_v2_*.jsonl
results/manifold_groups_poc/gradable_size_grid_v2_*.csv
```

The most important current variants are:

```text
gradable_size_disamb_pairs_v2_fictional_semantic_adjective_counts.jsonl
gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl
```

The `fictional_semantic_adjective_counts` variant uses invented rod-like objects:

```text
In this measurement task, a dax is a manufactured rod-like object.
Only overall length matters.
The normal overall length for Type A dax rods is 1 centimeter.
Compared only to that 1-centimeter length standard for Type A dax rods,
an overall length of 3 centimeters counts as ___.
```

The label family is:

```text
tiny < small < large < huge
```

The primary predictor is:

```text
rho = log(value / standard)
```

## Main Behavioral Artifacts

Behavioral scoring uses local normalization over the label set and reports:

```text
ordered_score
signed_score
corr(ordered_score, log(value / standard))
corr(signed_score, log(value / standard))
direction_match
argmax label counts
```

Key current files:

```text
results/manifold_groups_poc/gradable_size_v2_fictional_semantic_adjective_counts_behavior_recompute.md
results/manifold_groups_poc/gradable_size_v2_iso_ratio_adjective_counts_behavior_recompute.md
results/manifold_groups_poc/gradable_size_prior_diagnostics_gemma3_v2.summary.json
```

Current behavioral headline:

```text
fictional_semantic_adjective_counts:
  r(ordered_score, log(value/standard)) = 0.611
  95% CI = [0.531, 0.692]
  within-target permutation p = 0.00065

iso_ratio_adjective_counts:
  r(ordered_score, log(value/standard)) = 0.680
  95% CI = [0.514, 0.836]
```

## Main Raw-Patching Artifacts

Raw patching writes under:

```text
results/manifold_groups_poc/gradable_size_v2_*_raw_patch_*.csv
results/manifold_groups_poc/gradable_size_v2_*_raw_patch_*.summary.json
```

The most important current artifact is:

```text
results/manifold_groups_poc/gradable_size_v2_fictional_semantic_adjective_counts_raw_patch_final_prompt_token_l812162024283233_gemma3.summary.json
```

The raw-patching target is the final prompt token, and replacement states now come from exact decoder-block output hooks. This matters because the identity/sham patch must remain near zero.

Current causal headline after the hook-source fix:

```text
L08: aligned effect +0.0037
L12: aligned effect +0.0169
L16: aligned effect +0.0989
L20: aligned effect +0.2278
L24: aligned effect +0.2643
L28: aligned effect +0.2680
L32: aligned effect +0.2842
L33: aligned effect +0.3024
sham: 0.0 at all reported layers
```

Iso-ratio late-layer generalization also passes:

```text
iso_ratio_adjective_counts, final prompt token:
L20: aligned effect +0.1778, CI [+0.1418, +0.2148]
L24: aligned effect +0.1428, CI [+0.0998, +0.1835]
L28: aligned effect +0.1110, CI [+0.0669, +0.1528]
L32: aligned effect +0.0923, CI [+0.0481, +0.1359]
L33: aligned effect +0.1051, CI [+0.0542, +0.1547]
sham: 0.0 at all reported layers
```

## Main Low-Rank Causal Subspace Artifacts

Low-rank causal patching writes under:

```text
results/manifold_groups_poc/gradable_size_low_rank_patch_train_*_eval_*_gemma3.csv
results/manifold_groups_poc/gradable_size_low_rank_patch_train_*_eval_*_gemma3.summary.json
results/manifold_groups_poc/gradable_size_low_rank_patch_train_*_eval_*_gemma3.md
```

The key current pair of runs is:

```text
fictional_semantic_adjective_counts -> iso_ratio_adjective_counts:
  results/manifold_groups_poc/gradable_size_low_rank_patch_train_fictional_semantic_adjective_counts_eval_iso_ratio_adjective_counts_l162024_normmatched_r20_gemma3.md

iso_ratio_adjective_counts -> fictional_semantic_adjective_counts:
  results/manifold_groups_poc/gradable_size_low_rank_patch_train_iso_ratio_adjective_counts_eval_fictional_semantic_adjective_counts_l162024_gemma3.md
```

Bidirectional L20 headline:

```text
fictional -> iso, pca-k5:
  aligned_effect = +0.162 [0.110, 0.209]
  recovery/full = 0.860 [0.595, 1.146]
  direction_match = 0.848

iso -> fictional, pca-k5:
  aligned_effect = +0.155 [0.127, 0.187]
  recovery/full = 0.591 [0.492, 0.703]
  direction_match = 0.712
```

The reverse run is weaker in recovery but comparable in absolute aligned effect, and both directions beat random, norm-matched random, `value`, `standard`, and `value_standard_2d` controls at L20. This changes the status from a one-way causal hint to bidirectional held-out causal transfer for a shared low-rank size-calibration geometry candidate.

## Interpretation

The current evidence supports the following limited claim:

```text
Gemma-3-4B shows semantically scaffolded size-standard calibration when scalar values are embedded in a meaningful dimensional construction and read out through a compatible ordered adjective construction; the L20 residual stream contains a bidirectionally transferring low-rank causal structure for this calibration between fictional-semantic and iso-ratio variants.
```

The current evidence does not yet support:

```text
The model represents size predicates as a neural manifold.
SAE/CLT feature groups explain the behavior.
The effect is context-free arithmetic over value and standard.
```

The next mechanistic step is no longer just activation geometry. The geometry and low-rank causal gates have both passed in the current size regime. The next tests should be same-rho donor controls, paraphrase transfer of the discovered subspace, and then SAE/CLT feature-group recovery only if the causal controls continue to hold.

## Geometry Command

The geometry pipeline is intentionally separated from SAE/CLT group discovery. It first extracts the validated final-token block-output activations and then tests whether low-rank structure predicts the scalar calibration variables out of family.

```bash
TOKENIZERS_PARALLELISM=false \
DEVICE=mps \
TORCH_DTYPE=float32 \
LAYERS=20,24,28,32,33 \
bash scripts/run_gradable_size_geometry.sh
```

Primary outputs:

```text
results/manifold_groups_poc/gradable_size_geometry_late_final_token_gemma3.metadata.csv
results/manifold_groups_poc/gradable_size_geometry_late_final_token_gemma3.npz
results/manifold_groups_poc/gradable_size_geometry_late_final_token_gemma3.analysis.md
```

The geometry analysis has served its gate role: it justified the low-rank causal subspace test. It still does not by itself justify an SAE/CLT feature-group explanation.

## Causal Subspace Command

The low-rank causal subspace gate now passes bidirectionally. To reproduce the primary transfer test, train the subspace on fictional-semantic activations and evaluate the intervention on iso-ratio prompts:

```bash
TOKENIZERS_PARALLELISM=false \
TRAIN_VARIANT=fictional_semantic_adjective_counts \
EVAL_VARIANT=iso_ratio_adjective_counts \
DEVICE=mps \
TORCH_DTYPE=float32 \
LAYERS=16,20,24 \
ALPHAS=1.0 \
bash scripts/run_gradable_size_low_rank_patch.sh
```

The reverse transfer also passes and should be kept as the symmetric held-out check:

```bash
TOKENIZERS_PARALLELISM=false \
TRAIN_VARIANT=iso_ratio_adjective_counts \
EVAL_VARIANT=fictional_semantic_adjective_counts \
DEVICE=mps \
TORCH_DTYPE=float32 \
LAYERS=16,20,24 \
ALPHAS=1.0 \
bash scripts/run_gradable_size_low_rank_patch.sh
```

This is the direct test of whether the observed low-rank calibration structure is behaviorally causal. With both directions passing at L20, the next step is to test whether SAE/CLT feature groups span or approximate this causal subspace under held-out controls. Same-rho donor controls and paraphrase transfer remain required before broadening the claim beyond the current prompt regime.

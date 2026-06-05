# Gradable Cross-Domain Source-Basis Subspace Transfer

- Size train variant for positive control: `fictional_semantic_adjective_counts`
- Size eval data: `results/manifold_groups_poc/gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl`
- Controls: `results/manifold_groups_poc/gradable_cross_domain_matched_delta_rho_controls.csv`
- Source domains: `temperature,age`
- Layers: `20`
- Rank: `5`

| layer | basis | method | rank | alpha | n | size pairs | source pairs | train prompts | overlap | max angle | aligned effect | recovery/size-full | norm frac | dir match | min-n |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 20 | age | source_basis_pca | 5 | 1 | 46 | 23 | 23 | 46 | 0.159 | 89.0 | 0.006 [-0.001, 0.013] | 0.057 [-0.064, 0.171] | 0.334 | 0.630 | yes |
| 20 | age | source_basis_random_norm_matched | 5 | 1 | 920 | 23 | 23 | 46 | NA | NA | 0.003 [0.001, 0.006] | 0.018 [-0.061, 0.082] | 0.334 | 0.564 | yes |
| 20 | size_in_domain | full_size | 2560 | 1 | 46 | 23 | 0 | 62 | 1.000 | 0.0 | 0.178 [0.123, 0.229] | 1.000 [1.000, 1.000] | 1.000 | 0.848 | yes |
| 20 | size_in_domain | sham | 0 | 1 | 46 | 23 | 0 | 62 | 1.000 | 0.0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | yes |
| 20 | size_in_domain | size_basis_pca | 5 | 1 | 46 | 23 | 0 | 62 | 1.000 | 0.0 | 0.162 [0.105, 0.215] | 0.860 [0.596, 1.151] | 0.491 | 0.848 | yes |
| 20 | temperature | source_basis_pca | 5 | 1 | 46 | 23 | 23 | 58 | 0.158 | 89.0 | 0.028 [0.017, 0.039] | 0.175 [0.056, 0.319] | 0.323 | 0.717 | yes |
| 20 | temperature | source_basis_random_norm_matched | 5 | 1 | 920 | 23 | 23 | 58 | NA | NA | 0.003 [0.002, 0.005] | 0.024 [-0.012, 0.073] | 0.323 | 0.565 | yes |

Decision discipline:
- `size_in_domain/full_size` is the raw size full-vector upper bound for the same size receiver directions.
- `size_in_domain/size_basis_pca` is the positive L20 size-basis control.
- `temperature/source_basis_pca` and `age/source_basis_pca` fit PCA bases on source-domain activations, then project in-domain size deltas through those source-trained bases.
- A shared gradable-calibration subspace predicts positive source-basis transfer that beats norm-matched random controls.
- Subspace overlap is diagnostic only; the causal patching metric is the gate.

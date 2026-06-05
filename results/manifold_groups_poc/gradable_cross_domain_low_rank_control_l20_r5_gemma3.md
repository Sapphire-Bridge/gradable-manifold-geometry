# Gradable Cross-Domain Matched-Delta Low-Rank Control

- Size train variant: `fictional_semantic_adjective_counts`
- Size eval data: `results/manifold_groups_poc/gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl`
- Controls: `results/manifold_groups_poc/gradable_cross_domain_matched_delta_rho_controls.csv`
- Layers: `20`
- Rank: `5`

| layer | source | method | rank | alpha | n | size pairs | source pairs | match err | aligned effect | recovery/size-full | norm frac | dir match | min-n |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 20 | age | source_pca | 5 | 1 | 46 | 23 | 23 | 0.091 | 0.070 [0.042, 0.097] | 0.554 [0.071, 1.100] | 0.476 | 0.848 | yes |
| 20 | age | source_random_norm_matched | 5 | 1 | 920 | 23 | 23 | 0.091 | 0.004 [-0.003, 0.011] | -0.150 [-0.425, 0.059] | 0.476 | 0.504 | yes |
| 20 | size_in_domain | full_size | 2560 | 1 | 46 | 23 | 0 | 0.000 | 0.178 [0.123, 0.229] | 1.000 [1.000, 1.000] | 1.000 | 0.848 | yes |
| 20 | size_in_domain | sham | 0 | 1 | 46 | 23 | 0 | 0.000 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | yes |
| 20 | size_in_domain | size_pca | 5 | 1 | 46 | 23 | 0 | 0.000 | 0.162 [0.105, 0.215] | 0.860 [0.596, 1.151] | 0.491 | 0.848 | yes |
| 20 | temperature | source_pca | 5 | 1 | 46 | 23 | 23 | 0.041 | 0.081 [0.043, 0.118] | 0.604 [-0.637, 1.585] | 0.382 | 0.826 | yes |
| 20 | temperature | source_random_norm_matched | 5 | 1 | 920 | 23 | 23 | 0.041 | -0.005 [-0.008, -0.002] | -0.032 [-0.068, 0.006] | 0.382 | 0.483 | yes |

Decision discipline:
- `size_in_domain/full_size` is the raw size full-vector upper bound for the same receiver directions.
- `size_in_domain/size_pca` is the positive low-rank size control.
- `temperature/source_pca` and `age/source_pca` test whether matched signed-delta cross-domain activation deltas carry the same size-causal signal after projection through the size PCA subspace.
- A size-specific manifold candidate predicts strong `size_pca` and near-zero cross-domain `source_pca`, with min-n passing.
- This is not an SAE/CLT group explanation; it is a domain-specificity control for the causal low-rank candidate.

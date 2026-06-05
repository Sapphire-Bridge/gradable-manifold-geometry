# Gradable Cross-Domain Source-Delta-Basis Transfer

- Size train variant for positive control: `fictional_semantic_adjective_counts`
- Size eval data: `results/manifold_groups_poc/gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl`
- Controls: `results/manifold_groups_poc/gradable_cross_domain_matched_delta_rho_controls.csv`
- Source domains: `temperature,age`
- Source basis mode: `deltas`
- Layers: `20`
- Rank: `5`

| layer | basis | method | rank | alpha | n | size pairs | source pairs | train kind | train items | overlap | max angle | aligned effect | recovery/size-full | norm frac | dir match | min-n |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 20 | age | source_basis_pca | 5 | 1 | 46 | 23 | 23 | pair_deltas_unit | 72 | 0.141 | 88.1 | -0.000 [-0.005, 0.005] | 0.013 [-0.129, 0.142] | 0.307 | 0.522 | yes |
| 20 | age | source_basis_random_norm_matched | 5 | 1 | 920 | 23 | 23 | pair_deltas_unit | 72 | NA | NA | 0.003 [0.001, 0.005] | 0.016 [-0.060, 0.077] | 0.307 | 0.566 | yes |
| 20 | size_in_domain | full_size | 2560 | 1 | 46 | 23 | 0 | state_prompts | 62 | 1.000 | 0.0 | 0.178 [0.123, 0.229] | 1.000 [1.000, 1.000] | 1.000 | 0.848 | yes |
| 20 | size_in_domain | sham | 0 | 1 | 46 | 23 | 0 | state_prompts | 62 | 1.000 | 0.0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 | yes |
| 20 | size_in_domain | size_basis_pca | 5 | 1 | 46 | 23 | 0 | state_prompts | 62 | 1.000 | 0.0 | 0.162 [0.105, 0.215] | 0.860 [0.596, 1.151] | 0.491 | 0.848 | yes |
| 20 | temperature | source_basis_pca | 5 | 1 | 46 | 23 | 23 | pair_deltas_unit | 84 | 0.155 | 89.5 | 0.032 [0.020, 0.045] | 0.153 [0.025, 0.290] | 0.324 | 0.783 | yes |
| 20 | temperature | source_basis_random_norm_matched | 5 | 1 | 920 | 23 | 23 | pair_deltas_unit | 84 | NA | NA | 0.004 [0.002, 0.005] | 0.022 [-0.017, 0.073] | 0.324 | 0.567 | yes |

Decision discipline:
- `size_in_domain/full_size` is the raw size full-vector upper bound for the same size receiver directions.
- `size_in_domain/size_basis_pca` is the positive L20 size-basis control.
- `temperature/source_basis_pca` and `age/source_basis_pca` fit PCA bases on source-domain states or source-domain pair deltas, then project in-domain size deltas through those source-trained bases.
- A shared gradable-calibration subspace predicts positive source-basis transfer that beats norm-matched random controls.
- In `deltas` mode, source pair deltas are oriented from higher scalar coordinate to lower scalar coordinate and SVD is uncentered so the mean contrast direction is retained.
- Subspace overlap is diagnostic only; the causal patching metric is the gate.

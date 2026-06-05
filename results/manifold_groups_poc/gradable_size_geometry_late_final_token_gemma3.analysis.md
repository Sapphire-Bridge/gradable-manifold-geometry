# Gradable Size Activation Geometry

- Activation NPZ: `results/manifold_groups_poc/
  gradable_size_geometry_late_final_token_gemma3.npz`
- Metadata CSV: `results/manifold_groups_poc/
  gradable_size_geometry_late_final_token_gemma3.metadata.csv`
- Rows: `84`
- Layers: `20, 24, 28, 32, 33`

## Cross-Variant Ridge

| layer | split | target | model | n | r | r2 |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | ridge_full | 22 | 0.700 | -4.423 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca1_ridge | 22 | 0.841 | 0.696 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca2_ridge | 22 | 0.837 | 0.646 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca5_ridge | 22 | 0.858 | 0.164 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | ridge_full | 62 | 0.809 | 0.365 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca1_ridge | 62 | 0.651 | 0.171 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca2_ridge | 62 | 0.670 | 0.239 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca5_ridge | 62 | 0.690 | 0.228 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | ridge_full | 22 | 0.909 | 0.241 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca1_ridge | 22 | 0.896 | 0.262 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca2_ridge | 22 | 0.894 | 0.218 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca5_ridge | 22 | 0.905 | 0.473 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | ridge_full | 62 | 0.934 | -0.362 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca1_ridge | 62 | 0.918 | 0.297 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca2_ridge | 62 | 0.931 | 0.048 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca5_ridge | 62 | 0.942 | 0.191 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | ridge_full | 22 | 0.901 | 0.505 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca1_ridge | 22 | 0.891 | 0.305 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca2_ridge | 22 | 0.893 | 0.356 |
| 20 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca5_ridge | 22 | 0.904 | 0.599 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | ridge_full | 62 | 0.932 | -0.370 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca1_ridge | 62 | 0.913 | 0.311 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca2_ridge | 62 | 0.932 | 0.065 |
| 20 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca5_ridge | 62 | 0.941 | 0.216 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | ridge_full | 22 | 0.615 | -1.346 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca1_ridge | 22 | 0.318 | -0.059 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca2_ridge | 22 | 0.667 | -0.344 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca5_ridge | 22 | 0.642 | 0.133 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | ridge_full | 62 | 0.392 | 0.086 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca1_ridge | 62 | 0.542 | 0.113 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca2_ridge | 62 | 0.552 | 0.114 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca5_ridge | 62 | 0.504 | 0.128 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | ridge_full | 22 | 0.898 | 0.264 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca1_ridge | 22 | 0.559 | -0.343 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca2_ridge | 22 | 0.852 | -1.105 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca5_ridge | 22 | 0.645 | 0.175 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | ridge_full | 62 | 0.858 | 0.227 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca1_ridge | 62 | 0.775 | 0.052 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca2_ridge | 62 | 0.726 | 0.024 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca5_ridge | 62 | 0.658 | 0.046 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | ridge_full | 22 | 0.890 | 0.577 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca1_ridge | 22 | 0.559 | -0.225 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca2_ridge | 22 | 0.863 | -0.821 |
| 24 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca5_ridge | 22 | 0.650 | 0.248 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | ridge_full | 62 | 0.874 | 0.267 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca1_ridge | 62 | 0.775 | 0.068 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca2_ridge | 62 | 0.726 | 0.041 |
| 24 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca5_ridge | 62 | 0.657 | 0.066 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | ridge_full | 22 | 0.812 | -0.389 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca1_ridge | 22 | 0.419 | 0.018 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca2_ridge | 22 | 0.678 | 0.371 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca5_ridge | 22 | 0.473 | 0.216 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | ridge_full | 62 | 0.297 | 0.054 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca1_ridge | 62 | 0.476 | 0.079 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca2_ridge | 62 | 0.402 | 0.058 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca5_ridge | 62 | 0.343 | 0.055 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | ridge_full | 22 | 0.886 | 0.299 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca1_ridge | 22 | 0.489 | -0.443 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca2_ridge | 22 | 0.551 | -0.348 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca5_ridge | 22 | 0.686 | 0.283 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | ridge_full | 62 | 0.810 | -0.391 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca1_ridge | 62 | 0.648 | -0.081 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca2_ridge | 62 | 0.569 | -0.168 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca5_ridge | 62 | 0.445 | -0.081 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | ridge_full | 22 | 0.871 | 0.339 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca1_ridge | 22 | 0.486 | -0.319 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca2_ridge | 22 | 0.568 | -0.220 |
| 28 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca5_ridge | 22 | 0.697 | 0.308 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | ridge_full | 62 | 0.832 | -0.368 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca1_ridge | 62 | 0.649 | -0.066 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca2_ridge | 62 | 0.576 | -0.150 |
| 28 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca5_ridge | 62 | 0.447 | -0.063 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | ridge_full | 22 | 0.898 | -0.339 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca1_ridge | 22 | -0.023 | -0.023 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca2_ridge | 22 | 0.663 | 0.255 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca5_ridge | 22 | 0.576 | 0.153 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | ridge_full | 62 | 0.276 | -0.076 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca1_ridge | 62 | 0.322 | 0.032 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca2_ridge | 62 | 0.416 | 0.003 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca5_ridge | 62 | 0.338 | 0.035 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | ridge_full | 22 | 0.893 | 0.661 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca1_ridge | 22 | 0.009 | -0.565 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca2_ridge | 22 | 0.528 | -0.443 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca5_ridge | 22 | 0.661 | 0.322 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | ridge_full | 62 | 0.838 | 0.206 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca1_ridge | 62 | 0.540 | -0.143 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca2_ridge | 62 | 0.623 | -0.215 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca5_ridge | 62 | 0.441 | 0.007 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | ridge_full | 22 | 0.882 | 0.688 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca1_ridge | 22 | 0.008 | -0.483 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca2_ridge | 22 | 0.533 | -0.369 |
| 32 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca5_ridge | 22 | 0.672 | 0.320 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | ridge_full | 62 | 0.857 | 0.246 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca1_ridge | 62 | 0.518 | -0.132 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca2_ridge | 62 | 0.618 | -0.199 |
| 32 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca5_ridge | 62 | 0.419 | 0.018 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | ridge_full | 22 | 0.878 | -2.191 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca1_ridge | 22 | -0.153 | -0.028 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca2_ridge | 22 | 0.690 | 0.208 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | rho | pca5_ridge | 22 | 0.719 | 0.348 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | ridge_full | 62 | 0.383 | 0.005 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca1_ridge | 62 | 0.417 | 0.044 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca2_ridge | 62 | 0.540 | 0.034 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | rho | pca5_ridge | 62 | 0.380 | 0.054 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | ridge_full | 22 | 0.887 | 0.547 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca1_ridge | 22 | -0.073 | -0.566 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca2_ridge | 22 | 0.512 | -0.563 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | ordered_score | pca5_ridge | 22 | 0.729 | 0.452 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | ridge_full | 62 | 0.862 | 0.380 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca1_ridge | 62 | 0.632 | -0.117 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca2_ridge | 62 | 0.742 | -0.167 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | ordered_score | pca5_ridge | 62 | 0.475 | 0.062 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | ridge_full | 22 | 0.873 | 0.594 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca1_ridge | 22 | -0.073 | -0.481 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca2_ridge | 22 | 0.513 | -0.472 |
| 33 | fictional_semantic_adjective_counts_to_iso_ratio_adjective_counts | signed_score | pca5_ridge | 22 | 0.733 | 0.448 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | ridge_full | 62 | 0.876 | 0.414 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca1_ridge | 62 | 0.610 | -0.106 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca2_ridge | 62 | 0.731 | -0.152 |
| 33 | iso_ratio_adjective_counts_to_fictional_semantic_adjective_counts | signed_score | pca5_ridge | 62 | 0.452 | 0.069 |

## Distance Diagnostics

| layer | corr dist~|delta rho| | beta rho | beta value | beta standard | same-rho dist | all dist |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.097 | 0.127 | 0.087 | 0.046 | 56.540 | 68.764 |
| 24 | 0.144 | 0.034 | 0.118 | 0.263 | 63.626 | 70.093 |
| 28 | 0.088 | 0.041 | 0.097 | 0.147 | 65.727 | 69.860 |
| 32 | 0.062 | 0.031 | 0.067 | 0.132 | 65.686 | 69.476 |
| 33 | 0.060 | 0.033 | 0.060 | 0.124 | 65.660 | 69.417 |

## Delta Alignment

| layer | n deltas | mean pairwise cosine | projection r with rho |
| ---: | ---: | ---: | ---: |
| 20 | 17 | 0.238 | 0.767 |
| 24 | 17 | 0.221 | 0.627 |
| 28 | 17 | 0.253 | 0.576 |
| 32 | 17 | 0.273 | 0.533 |
| 33 | 17 | 0.275 | 0.562 |

Interpretation discipline:
- This is an activation-geometry gate, not an SAE/CLT group result.
- Treat L20/L24 as the primary mechanistic layers; L32/L33 are readout-near sanity endpoints.
- A manifold-style claim requires held-out low-rank structure plus causal low-rank patching, not only a positive PCA/Ridge table.

# Gradable Size Semantics-Derived Steering

- Train variant: `fictional_semantic_adjective_counts`
- Eval variant: `iso_ratio_adjective_counts`
- Layers: `20`
- Alpha sweep: `-2,-1,-0.5,0,0.5,1,2`

This is fixed-vector activation steering, not donor-conditioned patching:
`h' = h + alpha * s * d`, where `d` is derived from train-set
high-rho minus low-rho size contrasts and `s` is the primary
pca-delta-mean train-pair delta scale shared by non-sham directions.
Positive alpha is oriented toward larger standard-relative size
judgments.

## Curve-level steering slope

| layer | method | rank | curves | score slope/alpha | positive slope rate |
| ---: | --- | ---: | ---: | --- | ---: |
| 20 | pca_delta_mean | 5 | 46 | 0.029 [0.026, 0.032] | 1.000 |
| 20 | random_norm_matched | 5 | 920 | 0.005 [0.003, 0.008] | 0.559 |
| 20 | sham | 0 | 46 | 0.000 [0.000, 0.000] | 0.000 |
| 20 | standard | 1 | 46 | 0.044 [0.042, 0.046] | 1.000 |
| 20 | value | 1 | 46 | -0.055 [-0.059, -0.050] | 0.000 |
| 20 | value_standard_2d | 2 | 46 | -0.058 [-0.060, -0.056] | 0.000 |

## Alpha-level effects

| layer | method | rank | alpha | n | effect | signed effect | direction match | steer norm |
| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 20 | pca_delta_mean | 5 | -2 | 46 | -0.061 [-0.066, -0.056] | 0.061 [0.055, 0.066] | 1.000 | 1499.088 |
| 20 | pca_delta_mean | 5 | -1 | 46 | -0.029 [-0.031, -0.026] | 0.029 [0.026, 0.032] | 1.000 | 749.544 |
| 20 | pca_delta_mean | 5 | -0.5 | 46 | -0.014 [-0.015, -0.013] | 0.014 [0.013, 0.015] | 1.000 | 374.772 |
| 20 | pca_delta_mean | 5 | 0 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan | 0.000 |
| 20 | pca_delta_mean | 5 | 0.5 | 46 | 0.014 [0.012, 0.015] | 0.014 [0.012, 0.015] | 1.000 | 374.772 |
| 20 | pca_delta_mean | 5 | 1 | 46 | 0.027 [0.025, 0.030] | 0.027 [0.024, 0.030] | 1.000 | 749.544 |
| 20 | pca_delta_mean | 5 | 2 | 46 | 0.055 [0.050, 0.060] | 0.055 [0.050, 0.060] | 1.000 | 1499.088 |
| 20 | random_norm_matched | 5 | -2 | 920 | 0.010 [0.007, 0.014] | -0.010 [-0.014, -0.007] | 0.478 | 1499.088 |
| 20 | random_norm_matched | 5 | -1 | 920 | -0.000 [-0.001, 0.001] | 0.000 [-0.001, 0.001] | 0.534 | 749.544 |
| 20 | random_norm_matched | 5 | -0.5 | 920 | -0.002 [-0.002, -0.001] | 0.002 [0.001, 0.002] | 0.563 | 374.772 |
| 20 | random_norm_matched | 5 | 0 | 920 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan | 0.000 |
| 20 | random_norm_matched | 5 | 0.5 | 920 | 0.004 [0.004, 0.005] | 0.004 [0.004, 0.005] | 0.568 | 374.772 |
| 20 | random_norm_matched | 5 | 1 | 920 | 0.012 [0.010, 0.013] | 0.012 [0.010, 0.013] | 0.571 | 749.544 |
| 20 | random_norm_matched | 5 | 2 | 920 | 0.031 [0.026, 0.036] | 0.031 [0.026, 0.036] | 0.604 | 1499.088 |
| 20 | sham | 0 | -2 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| 20 | sham | 0 | -1 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| 20 | sham | 0 | -0.5 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| 20 | sham | 0 | 0 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan | 0.000 |
| 20 | sham | 0 | 0.5 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| 20 | sham | 0 | 1 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| 20 | sham | 0 | 2 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 0.000 |
| 20 | standard | 1 | -2 | 46 | -0.083 [-0.089, -0.077] | 0.083 [0.077, 0.089] | 1.000 | 1499.088 |
| 20 | standard | 1 | -1 | 46 | -0.041 [-0.044, -0.039] | 0.041 [0.039, 0.044] | 1.000 | 749.544 |
| 20 | standard | 1 | -0.5 | 46 | -0.021 [-0.022, -0.020] | 0.021 [0.020, 0.022] | 1.000 | 374.772 |
| 20 | standard | 1 | 0 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan | 0.000 |
| 20 | standard | 1 | 0.5 | 46 | 0.022 [0.020, 0.023] | 0.022 [0.020, 0.023] | 1.000 | 374.772 |
| 20 | standard | 1 | 1 | 46 | 0.044 [0.042, 0.046] | 0.044 [0.042, 0.046] | 1.000 | 749.544 |
| 20 | standard | 1 | 2 | 46 | 0.095 [0.092, 0.099] | 0.095 [0.091, 0.099] | 1.000 | 1499.088 |
| 20 | value | 1 | -2 | 46 | 0.131 [0.121, 0.140] | -0.131 [-0.141, -0.121] | 0.000 | 1499.088 |
| 20 | value | 1 | -1 | 46 | 0.059 [0.053, 0.064] | -0.059 [-0.064, -0.054] | 0.000 | 749.544 |
| 20 | value | 1 | -0.5 | 46 | 0.028 [0.024, 0.030] | -0.028 [-0.030, -0.025] | 0.000 | 374.772 |
| 20 | value | 1 | 0 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan | 0.000 |
| 20 | value | 1 | 0.5 | 46 | -0.024 [-0.027, -0.020] | -0.024 [-0.027, -0.021] | 0.000 | 374.772 |
| 20 | value | 1 | 1 | 46 | -0.045 [-0.052, -0.038] | -0.045 [-0.052, -0.039] | 0.000 | 749.544 |
| 20 | value | 1 | 2 | 46 | -0.090 [-0.103, -0.076] | -0.090 [-0.104, -0.077] | 0.000 | 1499.088 |
| 20 | value_standard_2d | 2 | -2 | 46 | 0.125 [0.120, 0.129] | -0.125 [-0.129, -0.120] | 0.000 | 1499.088 |
| 20 | value_standard_2d | 2 | -1 | 46 | 0.059 [0.057, 0.061] | -0.059 [-0.061, -0.057] | 0.000 | 749.544 |
| 20 | value_standard_2d | 2 | -0.5 | 46 | 0.029 [0.028, 0.030] | -0.029 [-0.030, -0.028] | 0.000 | 374.772 |
| 20 | value_standard_2d | 2 | 0 | 46 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | nan | 0.000 |
| 20 | value_standard_2d | 2 | 0.5 | 46 | -0.028 [-0.029, -0.027] | -0.028 [-0.029, -0.027] | 0.000 | 374.772 |
| 20 | value_standard_2d | 2 | 1 | 46 | -0.055 [-0.059, -0.052] | -0.055 [-0.059, -0.052] | 0.000 | 749.544 |
| 20 | value_standard_2d | 2 | 2 | 46 | -0.108 [-0.117, -0.101] | -0.108 [-0.116, -0.100] | 0.000 | 1499.088 |

Interpretation discipline:
- `pca_delta_mean` is the primary semantics-derived steering direction.
- `value`, `standard`, and `value_standard_2d` are explicit-variable controls.
- `random_norm_matched` should stay weaker than the primary direction.
- `sham` should remain near zero.
- This supports linear residual-stream steering if the alpha curve is monotone and control directions are weaker; it is not a geodesic manifold-steering claim.

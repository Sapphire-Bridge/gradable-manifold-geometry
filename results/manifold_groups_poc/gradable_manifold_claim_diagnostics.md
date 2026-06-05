# Gradable Manifold-Claim Diagnostics

These are visual and tabular diagnostics for the current Manifold-Groups paper claim.
They use existing activation caches and existing low-rank patch CSVs; no new model patching run is performed.

## Headline

- Cross-variant PCA bases are not literally the same L20 subspace: L20 principal angles are 33.1 deg, 49.0 deg, 74.2 deg, 77.7 deg, 84.9 deg, with overlap 0.252.
- The strongest cross-variant basis overlap is at L8, overlap 0.393; overlap falls by L20.
- L20 combined PCA has r(PC1,rho)=-0.049, r(PC2,rho)=0.842, r(PC3,rho)=0.156.
- L20 remains the causal peak among available low-rank patch layers: mean bidirectional pca-k5 aligned effect 0.159.

Interpretation: the diagnostics support a behaviorally causal, low-rank size-calibration geometry, but not the stronger claim that independently fitted fictional and iso bases are literally the same rank-5 subspace at L20.

## Figures

- cross-variant subspace overlap: `figures/manifold_groups/cross_variant_subspace_overlap.svg`
- L20 PCA rho projection: `figures/manifold_groups/l20_pca_rho_projection.svg`
- transfer efficiency by delta rho: `figures/manifold_groups/transfer_efficiency_by_delta_rho.svg`
- layer trajectory: `figures/manifold_groups/layer_trajectory.svg`

## Analysis 1: Cross-variant subspace overlap

| layer | overlap | angles deg | random overlap |
| ---: | ---: | --- | ---: |
| 8 | 0.393 | 7.7, 26.8, 66.6, 80.6, 89.3 | 0.0020 |
| 12 | 0.369 | 10.1, 37.6, 63.0, 78.5, 86.7 | 0.0020 |
| 16 | 0.329 | 15.8, 38.3, 78.2, 79.3, 80.6 | 0.0020 |
| 20 | 0.252 | 33.1, 49.0, 74.2, 77.7, 84.9 | 0.0020 |
| 24 | 0.097 | 56.2, 72.0, 77.9, 80.6, 84.9 | 0.0020 |
| 28 | 0.073 | 59.0, 73.7, 81.6, 88.2, 88.9 | 0.0020 |
| 32 | 0.129 | 47.2, 68.1, 78.3, 87.2, 89.1 | 0.0020 |
| 33 | 0.175 | 29.1, 75.1, 78.3, 86.3, 87.7 | 0.0020 |

## Analysis 3: Transfer efficiency by |delta rho|

| bin | n | mean abs delta rho | mean recovery | mean aligned effect | dir match |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48 | 0.906 | 0.663 | 0.090 | 0.562 |
| 2 | 56 | 1.699 | 0.530 | 0.172 | 0.714 |
| 3 | 50 | 2.940 | 0.766 | 0.185 | 0.820 |
| 4 | 62 | 4.956 | 0.650 | 0.172 | 0.839 |

## Analysis 4: Layer trajectory

| layer | abs r(PC1,rho) | max abs r(PC1..5,rho) | pca5 aligned f->i | pca5 aligned i->f | mean aligned |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.135 | 0.774 | nan | nan | nan |
| 12 | 0.241 | 0.868 | nan | nan | nan |
| 16 | 0.093 | 0.824 | 0.047 | 0.046 | 0.046 |
| 20 | 0.049 | 0.842 | 0.162 | 0.155 | 0.159 |
| 24 | 0.035 | 0.869 | 0.055 | 0.016 | 0.035 |
| 28 | 0.034 | 0.880 | nan | nan | nan |
| 32 | 0.042 | 0.848 | nan | nan | nan |
| 33 | 0.046 | 0.714 | nan | nan | nan |

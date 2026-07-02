# Scalar Cross-Domain Transfer: size / speed / weight (three-domain refinement)

Public summary of the cross-domain scalar carrier audit. Refines the earlier
two-domain (speed<->weight) result: with **size** re-extracted under the *same*
scalar pipeline, the picture is not purely domain-specific but **partially shared,
asymmetric, and size-centered**. This is NOT a universal scalar manifold.

Model gemma-3-4b-pt @ cc012e0, L20, donor-conditioned projected-delta patch,
`gold_aligned_effect` primary, clustered bootstrap. Validation split, small n,
provisional. Carriers only (no steering).

## In-domain (all three under one pipeline)
Each domain passes a frozen behavioral gate (size acc/FLIP 1.000; speed 0.882/0.850;
mid-range weight 0.875/0.950) and has a real donor-conditioned L20 carrier (calibration
recovery ~0.5-0.7; random/sham controls clean). Within a domain, the unsupervised
blind top-PCA subspace is the strongest carrier.

## Cross-domain causal transfer (rho direction; ratio = cross / in-domain)
| direction | in-domain | cross | ratio | reading |
|---|---|---|---|---|
| size -> speed   | +0.83 | +0.80 | 0.96 | shares |
| weight -> size  | +2.95 | +2.50 | 0.85 | shares |
| size -> weight  | +1.54 | +0.98 | 0.64 | partial |
| speed -> size   | +2.95 | +0.60 | 0.20 | partial |
| speed -> weight | +1.54 | +0.20 | 0.13 | ~none |
| weight -> speed | +0.83 | +0.06 | 0.07 | ~none |

All size-involving cross-patches are far above the random-subspace control (~0);
speed<->weight is near random. Geometric projection overlap is also larger for
size-pairs (rho 0.62-0.66) than for speed<->weight (0.43).

## Interpretation
- **Partially shared, size-centered:** size<->speed and size<->weight transfer
  causally; speed<->weight does not. The scalar carrier is neither universal nor
  purely domain-specific.
- **The shared component is the interpretable one:** transfer is strongest along the
  `rho` / `value_standard_2d` (value-vs-standard) directions. Blind PCA is strongest
  *within* a domain but is the least clean *across* domains.
- **Asymmetric:** size's carrier drives speed/weight more completely than the reverse
  (size has the most complete rho-aligned carrier; speed's is weakest).
- **Only causality decides:** geometry alone would have over-suggested sharing; and
  the shared interpretable core is visible only once a third domain is added.

## Supported
- size, speed, mid-range weight compared under one scalar pipeline.
- Real in-domain carriers in all three.
- Partial, asymmetric cross-domain transfer, strongest along interpretable
  `rho` / `value_standard_2d`.
- Blind PCA strongest in-domain, less clean cross-domain.
- speed<->weight remains largely non-transferable.

## Not supported
- A universal scalar manifold.
- A fully shared size/speed/weight carrier.
- Robust full-weight claim (mid-range only; extremes are a documented stress finding,
  not localized).
- A scalar carrier != actuator claim (steering/actuator arm inconclusive, deferred).
- Any model-general claim (one model, L20).

## Caveats
One model; layer L20; validation split; small n; weight mid-range only; SHARED/
partial labels are sensitive to the large in-domain magnitude asymmetry, so read
ratios rather than labels; fixed-vector actuator arm deferred pending redesign.

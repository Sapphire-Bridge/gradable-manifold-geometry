# A worked example: the strongest steering direction is not the carrier

A small, reproducible demonstration of the carrier-vs-actuator distinction on a
**benign** graded concept (standard-relative size), in Gemma-3-4B at layer 20. This
is a worked example, **not** a safety benchmark — see the caveat at the bottom.

## Claim
For a graded concept, a fixed-vector **steering** direction can move the output
strongly and reliably, yet **not** be the structure the model uses to compute the
judgment. Steerability is evidence of an **actuator**, not of the **carrier**.

## The two interventions
- **Actuator (fixed-vector steering):** add a single direction to the residual
  stream and sweep its magnitude; measure how the judgment moves. Tests whether a
  direction can *push* the output.
- **Carrier (donor-conditioned patching):** take a base and a donor whose label
  flips for a *natural* reason (same value, change the comparison class), project the
  donor − base activation delta into a candidate subspace, inject it, and check
  whether the judgment flips *correctly*. Tests what *carries* the relation.

## The result (Gemma-3-4B, L20)
| direction | role tested | effect |
|---|---|---|
| `standard` | steering (actuator) | **strongest actuator**: slope/α **+0.044** [0.042, 0.046], positive-rate **1.000** |
| `pca_delta_mean` | steering | +0.029 [0.026, 0.032] |
| `value` / `value_standard_2d` | steering | reliable but **opposite** sign (−0.055 / −0.058) |
| `random_norm_matched` | steering control | +0.005 [0.003, 0.008] (rate 0.559) |
| `sham` | steering control | 0.000 |
| rank-5 L20 subspace | **patching (carrier)** | donor-conditioned aligned_effect **+0.162** [0.110, 0.209] |
| `standard` / `value` | patching | do **not** patch as the carrier |

**Read:** `standard` is the single strongest fixed-vector actuator — but it is not the
donor-conditioned carrier. The carrier is a distributed rank-5 subspace, and the
obvious variable `rho = log(value/standard)` is not the mechanism. **Carrier ≠
actuator**, demonstrated causally on the same model, same layer, with clean
random/sham controls.

## Why it matters (the general point)
For **vague** concepts the decision boundary is movable *by definition*
(supervaluationism: a set of admissible precisifications, not one sharp threshold).
So a threshold-shifting steering vector is *guaranteed* to flip borderline cases —
its success is entailed by vagueness, not by having found the concept. Only a
carrier-level test can tell whether an intervention dismantled the concept or merely
suppressed its expression.

## Caveat (important)
This is a small worked example on a **benign** graded adjective (size), N small,
validation split, one model, one layer. It is **not** a measurement on "harmful" or
any safety concept — the safety relevance is a *structural argument* (vague safety
concepts share the same movable-boundary structure), not a benchmark result here.

## Reproduce
See the size patching / steering scripts and run wrappers in this repo
(`scripts/patch_gradable_size_low_rank.py`, `scripts/steer_gradable_size.py`, and the
matching `run_*.sh`); canonical numbers are gated by
`scripts/check_gradable_claim_numbers.py`.

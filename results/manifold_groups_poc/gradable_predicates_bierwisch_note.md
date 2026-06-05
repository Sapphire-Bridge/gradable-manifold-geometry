# Gradable Predicates As Manifold Candidates

**Author:** Felix Borck
**Last updated:** 2026-05-25
**Status:** Public research note.

## One-Page Note

**Thesis.** Bierwisch-style gradable predicates are not themselves "manifolds" in the contemporary neural-representation sense. They are, however, unusually good **manifold candidates**: their interpretation depends on a scalar dimension, a measured value, a comparison class, and a context-dependent standard. This is exactly the kind of semantic object that should not be expected to appear as one isolated feature direction.

The strongest first case is **warm/cold**. A fixed physical value can receive different labels when the comparison class changes:

```text
For freshly served tea, 45 degrees Celsius is cool.
For swimming-pool water, 45 degrees Celsius is hot.
```

The scalar value is held constant; the standard changes. This is philosophically cleaner than a classical homonym such as `bank`, because the target phenomenon is not discrete sense selection. It is **contextual calibration of a semantic scale**.

**Bierwisch/Kennedy anchor.** Bierwisch's work on gradation and dimensional adjectives treats adjectives such as `warm`, `cold`, `tall`, `short`, `near`, and `far` as involving dimensions and degree/measure structure. Kennedy and McNally sharpen this into a modern scale-structure view. The key experimental lesson is that the lexical item alone is insufficient: the interpretation of a positive form such as `warm` or `tall` depends on a comparison standard. For neural experiments, this suggests a representational hypothesis: the model's token-in-context state should encode a point in a low-dimensional structure jointly determined by scalar value, comparison class, contextual standard, polarity, and construction type.

**Methodological choice.** We use Bierwisch, rather than a general theory of vagueness, as the primary semantic anchor. The reason is methodological: the experiment requires a tractable decomposition of gradable meaning into dimensions, scale values, comparison classes, norms, polarity, and degree constructions. Bierwisch supplies exactly these variables. Work on vagueness-by-degrees, such as Edgington's, motivates graded response measures near borderline cases, but it does not determine the representational axes used to construct the stimulus space, and the present claims do not depend on any theory of vague truth.

**Manifold hypothesis.**

> Contextually calibrated gradable predicates are candidates for behaviorally relevant semantic manifolds whose coordinates jointly encode scalar value, comparison-class standard, polarity, and degree construction.

**Status (2026-05).** The behavioral preconditions for this hypothesis have been tested on `google/gemma-3-4b-pt` across temperature, size, and age. The results narrow the claim significantly: standard-sensitive ordered-score behavior is reliable under one specific load-bearing prompt form (the v1.2 template, defined in §A2), but does *not* survive three structural paraphrases of that form in two of the three domains tested (§A4.1). For the size domain, where the behavior survives all three paraphrases, bidirectional low-rank causal subspace patching (§A4.2) now shows that a 5-dimensional unsupervised PCA basis at L20 transfers both fictional_semantic -> iso_ratio and iso_ratio -> fictional_semantic, with absolute aligned effects of +0.162 and +0.155 while value, standard, value+standard, random, and norm-matched random controls remain small. This is the first strong causal manifold-groups candidate in the project: a behaviorally validated, bidirectionally transferring low-rank calibration geometry, still confined to one model, one domain, and one prompt/readout regime.

**Connection to the companion limitation result.** The companion SAE/CLT writeback limitation paper (`https://github.com/Sapphire-Bridge/sae-writeback-limitation/blob/main/paper/sae_writeback_limitation_short_paper.md`) shows that high SAE/CLT reconstruction fidelity does not guarantee behavioral preservation. A gradable-predicate manifold task makes this sharper in two stages: a reconstruction or feature group must preserve not only geometry of the scalar values, but the **contextual recalibration** that determines whether the same value is `cool`, `warm`, or `hot` — and (per §A4.1) the behavioral target itself must first be stabilized under prompt and readout controls.

**Go criterion.** A discovered feature group is useful only if selected on a selection split and then, on held-out cases, preserves or steers the graded label distribution better than same-size top-k and matched random groups. The claim is not that "manifold groups are explanations" by default. The claim is that gradable predicates provide a principled testbed where behaviorally validated groups could become better candidate control units than isolated directions.


## Appendix: Selection Logic For Example Sentences

### A1. Bierwisch-Derived Selection Criteria

Use predicates that are:

1. **Dimensional:** the predicate is associated with a scale, e.g. temperature, height, distance, price, time.
2. **Gradable:** the predicate admits comparative or degree modification, e.g. warmer, colder, very warm, too hot.
3. **Comparison-class sensitive:** the same scalar value can receive different labels under different standards.
4. **Antonym-paired:** use pairs or ordered label sets such as `cold/cool/warm/hot`, `short/tall`, `near/far`, `cheap/expensive`.
5. **Intervention-friendly:** the scalar expression is a stable target span shared across minimal-pair prompts.

The Bierwisch variables map directly onto experimental controls and corresponding manifold expectations:

| Bierwisch variable | Experimental instantiation | Manifold expectation |
|---|---|---|
| Dimension D | height, temperature, price, distance | scale axis |
| Relatum x | person, liquid, object, location | object binding |
| Comparison class C | NBA players, child, tea, swimming pool | context axis |
| Norm N_c | context-dependent standard | shift of decision boundary |
| Degree value v | 180 cm, 45 °C, €80 | position on scale |
| Polarity | tall/short, warm/cold | opposite direction along axis |
| Construction | positive, comparative, equative | syntactic calibration |

Bierwisch's account is especially well-suited because the comparison class C and the norm N_c are not pragmatic residue: norms are C-dependent, and degrees are constituted by comparison operations, not external numbers. If degrees arise from comparison operations, transformers should implement them as relational geometric states rather than isolated lexical meanings.

This is why `warm/cold` is the first target. Temperature gives an explicit scalar dimension, and ordinary contexts provide clear standards: freezer, room, pool, bath, tea, oven.

### A2. Prompt Template

The earliest minimal template was:

```text
For {comparison_class}, {scalar_value} is
```

The continuation labels are ordered predicates:

```text
cold, cool, warm, hot
```

This keeps the target span fixed and makes the context do the calibration work. For example:

```text
For freshly served tea, 45 degrees Celsius is
For swimming-pool water, 45 degrees Celsius is
```

Both prompts share the target `45 degrees Celsius`; only the comparison class differs.

The current positive behavioral regime is more explicit. It uses the v1.2 baseline-template family:

```text
The normal {dimension} for {context} is {standard} {unit}.
Compared only to that {standard}-{unit_short} baseline for {context},
a {dimension} of {value} {unit} feels
```

The v1.3 paraphrase sweep shows that this wording is not a harmless stylistic choice. The explicit baseline frame is load-bearing: paraphrases that replace `feels`, weaken `Compared only`, or make the standard less syntactically central can collapse or reverse the ordered-score signal. For this reason, prompt form is now an experimental factor, not a presentation detail.

### A3. Why This Is More Manifold-Like Than `bank`

`bank` is useful for DISAMB because it asks whether context selects one of two sense basins. Gradable predicates ask a stronger question: whether context relocates a point on a scale. The expected representation is not two hard clusters, but a surface or trajectory over:

```text
scalar value x comparison class x linguistic framing
```

If SAEs/CLTs tile this surface locally, individual features may look fragmentary while groups carry the behaviorally relevant structure. The v1.3 paraphrase sweep adds a constraint: a candidate manifold cannot be inferred merely from a semantic grid. The behavior to be explained must first be stabilized under prompt and readout controls; otherwise the geometry may reflect stereotype or lexical-prior readouts rather than standard-relative calibration.

### A4. Graded Behavioral Readouts

The expected model behavior near a standard should not be reduced to a hard argmax flip between adjacent labels such as `cool` and `warm` or `warm` and `hot`. The graded shifts that the manifold-candidate hypothesis predicts are visible only in the full probability distribution over ordered labels. The behavioral readout can track:

1. `entropy` over ordered labels.
2. `margin` between adjacent labels such as `cool` and `warm`.
3. `slope` across scalar values within a comparison class.
4. `smoothness` of label-probability shifts around standards.
5. `borderline spread`, where multiple adjacent labels receive nontrivial mass.

These metrics are justified internally by the structure of the task — probability mass over ordered labels reflects underlying degree — without any commitment to a theory of vague truth. Vagueness-by-degrees accounts, such as Edgington's, provide a compatible external motivation for treating borderline cases as graded rather than binary, but the experimental claim here is purely behavioral: logits and patching effects are acceptance/continuation dispositions, not truth degrees.

### A4.1. Cross-Domain Behavioral Evidence (Gemma-3-4b)

The manifold-candidate hypothesis predicts that the model's predictions should track the value's position relative to the explicit comparison standard, across surface paraphrases of the prompt and across domains. We tested this in three iterations on `google/gemma-3-4b-pt` (revision `cc012e0`): a baseline format with orientation-balanced pairs (v1.2), and a sweep of three paraphrases of that baseline (v1.3 a/b/c).

**Primary metric.** Side-level Pearson correlation between the model's ordered probability score (probability-weighted position on the label scale `cold<cool<warm<hot`, etc.) and a domain-appropriate predictor of the value-to-standard relation (`value − standard` for temperature; `log(value / standard)` for size and age). Reported on prompt-deduplicated unique sides, with paired bootstrap 95% CIs over 5000 resamples and permutation p-values over 20000 resamples.

**Why this metric, not argmax.** Output-token priors lock the discrete argmax onto unmarked default labels (`small` for size, `old` for age) on 80–100% of sides across all four runs, irrespective of whether the comparison logic succeeds. The graded shift the manifold-candidate hypothesis predicts is visible only in the full probability distribution over ordered labels. This methodological commitment is itself an empirical finding (see Observation 1 below) and is necessary to interpret any of the other results.

#### Results

**Baseline (v1.2)** — prompt format: `"The normal X for {context} is {standard} {unit}. Compared only to that {standard}-{unit} baseline for {context}, a {dimension} of {value} {unit} feels"`.

| Domain | n unique sides | r(score, predictor) | 95% CI | perm. p |
|---|---|---|---|---|
| Temperature | 58 | **+0.678** | [+0.58, +0.76] | ≤ 5e-5 |
| Size | 62 | **+0.652** | [+0.51, +0.77] | ≤ 5e-5 |
| Age | 46 | **+0.642** | [+0.40, +0.81] | ≤ 5e-5 |

Pair-level direction-match rate is 0.83–0.86 across domains under orientation-balanced pairing (expected-shift signs ~50/50), confirming the side-level signal is not an orientation artefact.

**Paraphrase sweep (v1.3 a/b/c)** — three structural paraphrases of the v1.2 template (analytic "Relative to that {standard}-{unit} reference, ... is best described as", conditional "If the normal X for {context} is N, then V {unit} would best be called", elliptic "For {context}, normal is N. Compared to that, V {unit} is"). Same datasets, same scorer, same recompute.

| Domain | v1.2 | v1.3 A | v1.3 B | v1.3 C |
|---|---|---|---|---|
| Temperature | **+0.68** | +0.26 [+0.03, +0.47] | **−0.58** [−0.72, −0.42] | −0.22 [−0.45, +0.05] |
| Size | **+0.65** | +0.49 [+0.35, +0.64] | **+0.63** [+0.49, +0.77] | **+0.67** [+0.55, +0.77] |
| Age | **+0.64** | −0.07 [−0.30, +0.14] | −0.13 [−0.38, +0.14] | +0.10 [−0.13, +0.33] |

#### Observations

**1. The "argmax is the wrong readout" claim is generic and robust.** Across all four runs and all three domains, argmax is dominated by lexical default labels — `small` 80–96% for size, `old` 100% for age, mixed for temperature. The same model that emits this near-degenerate argmax can simultaneously encode a standard-sensitive ordered probability shift. Any discrete-argmax readout would have falsely reported size and age as failures of the manifold-candidate hypothesis in the baseline, and would have given misleading results in the paraphrase sweep as well. The probabilistic readout commitment (A4) is necessary, not optional.

**2. Standard-sensitive behavior is elicitable, but not paraphrase-invariant.** The baseline v1.2 result is reproducible and statistically robust in all three domains (r ≈ 0.65, CIs exclude 0, permutation p ≤ 5e-5). But the same behavior fails to replicate under three plausible paraphrases of the same load-bearing template. The failure modes are domain-specific:

- **Size remains the only positive domain across all paraphrases** (r ∈ [0.49, 0.67], all CIs exclude 0). Paraphrase A is weaker than B/C; the effect is not paraphrase-invariant even here.
- **Temperature shows three distinct failure modes**: A retains a weakened positive signal (r = 0.26, CI just above 0); C loses the signal (CI includes 0); B *inverts* the signal (r = −0.58, CI [−0.72, −0.42]) — consistent with the model anchoring on a stereotype label associated with the *standard* (e.g., responding "hot" whenever the prompt mentions an oven baseline, regardless of the value).
- **Age loses the signal in all three paraphrases** (all |r| < 0.13, all CIs include 0). With n = 46 unique sides, this is genuinely null rather than weakly inverted, though sample size limits power for detecting effects below |r| ≈ 0.3.

**3. The right framing for the manifold-candidate hypothesis follows.** A naive reading — "the model holds a stable warm/cold (or tiny/huge, or young/old) manifold that different prompts read out the same way" — is empirically excluded by the paraphrase sweep. A weaker and better-supported framing is:

> Gradable predicates provide a principled testbed for behaviorally validating candidate semantic manifolds, because their semantic degrees of freedom (dimension, standard, polarity, construction) are independently controllable. Initial results show that standard-sensitive ordered-probability behavior is elicitable but not uniformly prompt-invariant; manifold claims therefore require behavioral validation under paraphrase, artificial-standard, and intervention controls before any representational claim is warranted.

This is the same methodological pattern as the companion limitation result (`https://github.com/Sapphire-Bridge/sae-writeback-limitation/blob/main/paper/sae_writeback_limitation_short_paper.md`): high reconstruction fidelity does not guarantee behavioral preservation, and now — the behavioral signal one would seek to preserve is itself prompt- and readout-sensitive unless the context structure is controlled.

#### Roadmap for the next experimental phase

Activation geometry and SAE/CLT analysis (Section A5) are *not* the right immediate next step. The right step is factor isolation on the prompt side, because the v1.3 result shows the behavioral target is not yet a stable thing to seek a representation for. In approximate priority order:

1. **v1.2-ablation** — remove "Compared only" / replace "feels" with "is" / "called" / drop context repetition. Isolates which words in the v1.2 template are load-bearing.
2. **Neutral-context control** — replace world-anchored contexts (freezer/oven, ant/elephant) with neutral identifiers (Class A/B/C) holding the standards numerically constant. Directly tests the stereotype-anchoring hypothesis suggested by Temperature-B.
3. **Artificial / inverted standards** — `freezer normal 80 °C`, `oven normal −10 °C`. Decouples world knowledge from explicit standard.
4. **Age lexical control** — `young/youthful/mature/old` vs `young/mid/elderly/ancient` vs `early/peak/late/end-of-life`. Probes whether the age-domain collapse is driven by the `old` token's prior.

Only once one of these isolations identifies a paraphrase-robust positive regime should activation-geometry probing and SAE/CLT-group recovery (Section A5) be applied, and even then only against the regime in which behavior is robust.

#### Source artifacts

- `gradable_v1_2_behavior_recompute.{md,json}` — orientation-balanced baseline; current canonical positive result.
- `gradable_v1_3_{a,b,c}_behavior_recompute.{md,json}` — paraphrase sweep.

The unique-side primary metric is in each recompute JSON at `domains.<DOM>.side_metrics_unique.corr_ordered_score_vs_predictor`. The expanded-row variant under `side_metrics_expanded` exists for diagnostic comparison but counts duplicated prompts; do not use it for headline numbers.

### A4.2. Causal Localization: Low-Rank Subspace Patching (Size, v1.2)

For the size domain — the only domain in which the v1.2 behavioral signal survives all three paraphrases of §A4.1 — we test whether the standard-relative calibration variable is causally localizable in a low-dimensional subspace of the residual stream. The test follows the matched-control discipline of the companion limitation result (`https://github.com/Sapphire-Bridge/sae-writeback-limitation/blob/main/paper/sae_writeback_limitation_short_paper.md`): a *learned* k-dimensional direction is causally usable only if its intervention beats a battery of matched controls.

**Setup.** Activations are extracted from `google/gemma-3-4b-pt` (revision `cc012e0`) at L16/L20/L24, final prompt token, across two cross-variant splits of the size dataset: `fictional_semantic_adjective_counts` (62 unique sides; 85 pairs -> 170 directional patch rows when used as eval) and `iso_ratio_adjective_counts` (22 unique sides; 23 pairs -> 46 directional patch rows when used as eval). Bases are fit on one variant only; patching is evaluated on the held-out variant, then the train/eval direction is reversed. The donor for each receiver is the opposite side of the same pair (different ρ = log(value/standard) by construction).

**Intervention.** For each (donor, receiver) pair at layer L, the receiver's residual at the final prompt token is replaced by `h_receiver + α · U_k U_kᵀ (h_donor − h_receiver)` for α=1.0, where `U_k` is a k-dimensional orthonormal basis. Six classes of basis are compared (10 methods total):

- `pca` (rank 1, 2, 5) — unsupervised SVD of training activations
- `rho`, `ordered_score`, `signed_score` (rank-1 each) — kernel-ridge supervised single-directions
- `value`, `standard`, `value_standard_2d` — supervised directions or 2D-subspace against `log(value)` and `log(standard)` separately
- `delta_mean` — mean of ρ-sign-corrected pair difference vectors
- `random` (rank 1, 2, 5) — naive random orthonormal subspace, 20 repeats
- `random_norm_matched` (rank 1, 2, 5) — random subspace re-scaled to match the projected-norm of the strongest non-random candidate at the same rank, 20 repeats

The primary metric is `recovery/full = aligned_effect / full_vector_aligned_effect`, aggregated per cluster (pair_id) with cluster-bootstrap 95% CIs (B=500). Secondary metrics: `aligned_effect`, `patch_fraction`, `direction_match_rate`, `projected_norm_fraction`, `KL(patched ‖ donor) − KL(base ‖ donor)`. Sham control: receiver patched onto itself, must produce zero.

**Decision gates** (all four required):
1. `recovery/full ≥ 0.35` at k=5 on at least one (layer, method) cell, CI excludes 0.
2. `direction_match_rate ≥ 0.65` on the same cell.
3. The cell beats `random_norm_matched` (CI separation).
4. The cell beats `value`, `standard`, and `value_standard_2d` (CI separation).

#### Results (cross-variant train: fictional → eval: iso_ratio)

| Layer | Method | recovery/full [95% CI] | aligned_effect | dir match | n |
|---|---|---|---|---|---|
| L16 | full | 1.000 [1.000, 1.000] | 0.056 [0.040, 0.074] | 0.870 | 46 |
| L16 | **pca-k5** | **1.274 [0.793, 1.966]** | 0.047 [0.028, 0.065] | 0.761 | 46 |
| L16 | **pca-k2** | **0.653 [0.335, 0.950]** | 0.037 [0.024, 0.047] | 0.891 | 46 |
| L16 | pca-k1 | −0.083 [−0.196, −0.005] | 0.000 | 0.543 | 46 |
| L20 | full | 1.000 [1.000, 1.000] | **0.178 [0.119, 0.234]** | 0.848 | 46 |
| L20 | **pca-k5** | **0.860 [0.595, 1.146]** | 0.162 [0.110, 0.209] | 0.848 | 46 |
| L20 | **pca-k2** | **0.367 [0.249, 0.459]** | 0.077 [0.053, 0.102] | 0.913 | 46 |
| L20 | signed_score (rank-1) | 0.447 [0.276, 0.655] | 0.069 [0.047, 0.092] | 0.739 | 46 |
| L20 | ordered_score (rank-1) | 0.441 [0.288, 0.636] | 0.066 [0.045, 0.087] | 0.783 | 46 |
| L20 | rho (rank-1) | 0.017 [−0.036, 0.096] | −0.004 | 0.087 | 46 |
| L20 | value, standard, value_standard_2d | ≤ 0.02 (all CIs include 0) | ~0 | ≤ 0.15 | 46 |
| L20 | random (any k, 20 repeats) | ≈ 0.00 | ≈ 0 | ~0.49 | 920 |
| L20 | random_norm_matched (any k, 20 repeats) | ≤ 0.05 (all CIs include 0) | ≤ 0.007 | ~0.6 | 920 |
| L24 | full | 1.000 [1.000, 1.000] | 0.143 [0.089, 0.201] | 0.739 | 46 |
| L24 | pca-k5 | −0.809 [−4.820, 2.211] (unstable) | 0.055 | 0.870 | 46 |
| L24 | signed_score (rank-1) | −0.122 [−2.092, 1.357] (unstable) | 0.138 | 0.826 | 46 |

All four gates are met on **L20 pca-k5** (recovery 0.860, dir-match 0.848, beats both norm-matched random and value/standard/value_standard_2d controls with non-overlapping CIs). The L16 result is also positive but with a smaller full-effect baseline; the L24 result is unusable as a recovery metric due to high pair-level variance.

#### Reverse Results (cross-variant train: iso_ratio → eval: fictional)

The reverse split is larger on the evaluation side (170 directional rows) and also passes at L20 pca-k5. Recovery is weaker than in the forward direction, but the absolute causal effect is nearly the same size.

| Layer | Method | recovery/full [95% CI] | aligned_effect | dir match | n |
|---|---|---|---|---|---|
| L20 | full | 1.000 [1.000, 1.000] | **0.228 [0.171, 0.286]** | 0.706 | 170 |
| L20 | **pca-k5** | **0.591 [0.492, 0.703]** | 0.155 [0.127, 0.187] | 0.712 | 170 |
| L20 | pca-k2 | 0.490 [0.333, 0.721] | 0.073 [0.055, 0.091] | 0.747 | 170 |
| L20 | pca-k1 | 0.493 [0.327, 0.706] | 0.074 [0.056, 0.091] | 0.753 | 170 |
| L20 | ordered_score (rank-1) | 0.323 [0.225, 0.439] | 0.066 [0.054, 0.077] | 0.776 | 170 |
| L20 | signed_score (rank-1) | 0.322 [0.224, 0.438] | 0.065 [0.053, 0.076] | 0.776 | 170 |
| L20 | rho (rank-1) | 0.028 [−0.031, 0.090] | 0.014 [0.013, 0.016] | 0.988 | 170 |
| L20 | random_norm_matched-k5 | 0.040 [0.021, 0.069] | 0.005 [0.004, 0.007] | 0.539 | 3400 |
| L20 | value, standard, value_standard_2d | ≤ 0.031 (CIs include or remain near zero recovery) | ≤ 0.016 | controls not competitive | 170 |
| L20 | sham | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 | 170 |

Taken together, the two directions establish **bidirectional held-out causal transfer** at L20: fictional -> iso recovers 0.860 of the full patch, and iso -> fictional recovers 0.591, with nearly identical absolute low-rank aligned effects (+0.162 vs +0.155).

#### Three surprises

1. **The strongest decoded direction (`rho`, decoding r ≈ 0.98 from §A5 geometry analysis) has effectively zero causal effect** as a 1D patching direction at L20 (recovery 0.017, CI [−0.036, 0.096]). This is a direct instance of the companion limitation pattern — decoding fidelity does not imply causal usability — and is the cleanest local replication of that pattern we have produced inside this project.

2. **The patchable subspace is *not* `value ⊕ standard`.** The 2D subspace explicitly constructed from supervised value and standard directions (`value_standard_2d`) has near-zero recovery in both directions (0.019 forward; 0.031 reverse, CI includes zero/near-zero behavior). The naive interpretation — "the model encodes log-value and log-standard as separable linear directions and patching transfers their numerical interpretation" — is empirically excluded. The structure that PCA finds is emergent, not trivially decomposable into the explicit Bierwisch variables (§A1).

3. **PCA needs k ≥ 2.** Single PC at L20 recovers only 0.150 (CI includes 0); single PC at L16 is slightly negative. The causal structure is at least 2-dimensional even though the strongest decoded scalar (ρ) is one-dimensional. This is consistent with the multidimensional candidate-manifold framing of §A1, but it sharpens it: the relevant degrees of freedom are not the explicit Bierwisch variables (D, x, C, N_c, v, polarity, construction) considered separately.

#### Interpretation

The first three of the four causal gates in §Manifold-Hypothesis are now met for **size under v1.2/adjective-counts prompts at L20**: behavior is present (§A4.1), behavior is causally localizable in a low-rank subspace, and that subspace transfers bidirectionally across the fictional-semantic and iso-ratio variants while beating naive random, norm-matched random, and the most plausible explicit-variable control (`value_standard_2d`). The fourth gate — discovered SAE/CLT feature *groups* preserving graded label distributions under held-out evaluation — remains to be tested. The roadmap (§A5) reorders accordingly.

This is the first positive **causal-mechanistic** evidence for the manifold-candidate hypothesis that is strong enough to communicate as a result rather than only a hint. It is narrowed in three important ways: (i) one domain only (size, the sole paraphrase-robust domain of §A4.1); (ii) one prompt/readout regime (v1.2/adjective-counts); (iii) one model (`google/gemma-3-4b-pt`, revision `cc012e0`).

#### What this is not yet evidence for

- A **paraphrase-invariant** calibration subspace. The bases here are fit on v1.2-form prompts; their behavior under v1.3 paraphrases is not measured.
- A **manifold** in the strict sense (curved, locally low-dimensional structure). The current test demonstrates a *linear* low-rank subspace, which is the flat tangent-approximation of any underlying manifold. Curvature would require an additional test.
- Identity of the two learned bases in the strict linear-algebra sense. The principal-angle / subspace-overlap diagnostic has now been run (`gradable_manifold_claim_diagnostics.md`): at L20 the cross-variant bases have principal angles 33.1°, 49.0°, 74.2°, 77.7°, 84.9° and projection overlap 0.252 (vs ≈ 0.002 for random rank-5 subspaces). Bidirectional transfer therefore reflects *shared but not identical* causal structure — the two fitted PCA bases are demonstrably not the same subspace, and overlap is strongest at L8 (0.393), falling by L20.
- **Cross-domain causal specificity — partially addressed (§A4.3).** Matched-Δ cross-domain probes using temperature and age show partial transfer through the size basis, but independently fitted source-state bases do not substitute for the size basis. The supported claim is domain-preferential shared calibration structure, not a strictly size-specific subspace. A structurally different non-Bierwisch negative control, plus a same-domain different-pair same-ρ control, would sharpen the lower bound.
- **Model-general behavior.** All numbers are for `gemma-3-4b-pt` at a fixed revision.

#### Source artifacts

- `gradable_size_low_rank_patch_train_fictional_semantic_adjective_counts_eval_iso_ratio_adjective_counts_l162024_normmatched_r20_gemma3.{md,summary.json}` — primary results (large row-level `.csv` not included; see the README external-artifacts table).
- `gradable_size_low_rank_patch_train_iso_ratio_adjective_counts_eval_fictional_semantic_adjective_counts_l162024_gemma3.{md,summary.json}` — reverse-transfer results (large row-level `.csv` not included; see the README external-artifacts table).
- `gradable_size_geometry_broad_final_token_gemma3.{npz,metadata.csv}` — activation cache (L8/12/16/20/24/28/32/33, final prompt token).
- Sanity-check anchor: `gradable_size_v2_iso_ratio_adjective_counts_raw_patch_final_prompt_token_l2024283233_gemma3.summary.json` (L20 full aligned_effect = +0.1778 [0.142, 0.215], matches `recovery/full = 1.000` row in the table above).

#### A4.3 Cross-domain specificity probe — working note 2026-05-27

**Status: completed. Claim updated.**

Three sequential tests assess whether the L20 pca-k5 size subspace is size-specific or reflects a broader gradable-calibration channel.

##### Test 1 — Matched-Δ cross-domain deltas through U_size

Temperature- and age-donor deltas were matched to size receiver directions by domain-z-normalized signed scalar coordinate and projected through the size-trained L20 pca-k5 basis before patching size receivers.

| source | aligned effect | recovery/size-full | dir match | random_norm_matched |
|---|---|---|---|---|
| size_pca (positive control) | 0.162 [0.105, 0.215] | 0.860 [0.596, 1.151] | 0.848 | — |
| temperature → size | 0.081 [0.043, 0.118] | 0.604 [−0.637, 1.585] | 0.826 | −0.005 [−0.008, −0.002] |
| age → size | 0.070 [0.042, 0.097] | 0.554 [0.071, 1.100] | 0.848 | 0.004 [−0.003, 0.011] |

Each source domain contributes 23 matched source pairs and 46 matched directions. Both cross-domain effects are above norm-matched random and their aligned-effect CIs exclude zero. This argues against a strictly size-specific interpretation: matched temperature and age deltas carry roughly half of the in-domain size-basis signal through `U_size`.

##### Test 2 — Source-state bases applied to size receivers

PCA bases fit on temperature/age activation states at L20 (`U_temperature_state`, `U_age_state`) were applied to in-domain size deltas on size receivers.

| basis | aligned effect | recovery/size-full | max angle | overlap | dir match |
|---|---|---|---|---|---|
| size_basis_pca (positive control) | 0.162 [0.105, 0.215] | 0.860 [0.596, 1.151] | 0.0° | 1.000 | 0.848 |
| temperature_state_basis → size | 0.028 [0.017, 0.039] | 0.175 [0.056, 0.319] | 89.0° | 0.158 | 0.717 |
| age_state_basis → size | 0.006 [−0.001, 0.013] | 0.057 [−0.064, 0.171] | 89.0° | 0.159 | 0.630 |

The source-state bases do not replace the size basis. Temperature state PCs transfer weakly but above the norm-matched random control; age state PCs are borderline null. The 89° maximum principal angle alone should not be overinterpreted because random rank-5 subspaces in R^2560 are also nearly orthogonal by this metric. The projection-overlap statistic is more diagnostic: overlap around 0.16 is well above chance but far below identity.

##### Test 3 — Source-delta bases applied to size receivers

Bases fit on oriented source-domain pair deltas were applied to in-domain size deltas on size receivers:

```text
U_temperature_delta = PCA(h_temperature_high - h_temperature_low)
U_age_delta         = PCA(h_age_high - h_age_low)
```

and evaluates:

```text
h_size_receiver_patched =
  h_size_receiver + α * U_source_delta_k U_source_delta_kᵀ (h_size_donor - h_size_receiver)
```

| basis | aligned effect | recovery/size-full | max angle | overlap | dir match | random_norm_matched |
|---|---|---|---|---|---|---|
| size_basis_pca (positive control) | 0.162 [0.105, 0.215] | 0.860 [0.596, 1.151] | 0.0° | 1.000 | 0.848 | — |
| temperature_delta_basis → size | 0.032 [0.020, 0.045] | 0.153 [0.025, 0.290] | 89.5° | 0.155 | 0.783 | 0.004 [0.002, 0.005] |
| age_delta_basis → size | −0.000 [−0.005, 0.005] | 0.013 [−0.129, 0.142] | 88.1° | 0.141 | 0.522 | 0.003 [0.001, 0.005] |

Delta-basis transfer does not rescue a strong shared-basis claim. Temperature delta PCs transfer weakly and clearly above norm-matched random; age delta PCs are null and indistinguishable from, or worse than, the random baseline. Compared with Test 2, the delta basis slightly improves temperature but weakens age. The supported positive is therefore asymmetric and modest, not a robust shared contrast basis across size, temperature, and age.

##### Interpretation

The three tests rule out the two coarse endpoints:

- **Not strictly size-specific:** matched temperature/age deltas carry a substantial part of the in-domain causal signal through `U_size` (Test 1).
- **Not a single shared low-rank basis:** source-state bases do not substitute for `U_size` (Test 2), and source-delta bases do not produce robust symmetric transfer either (Test 3).

The most defensible mechanistic reading is narrower than the initial size-specific claim and also narrower than a full shared-manifold claim: `U_size` is a domain-preferential causal basis that can read out matched scalar-calibration deltas from temperature and age, but independently fitted temperature/age bases do not reliably span the same size-causal subspace. Temperature shows a weak transferable component in both state and delta bases; age does not. The current evidence supports **cross-domain activation of the size basis**, not a single shared rank-5 basis across domains.

##### Source artifacts

- `gradable_cross_domain_matched_delta_rho_controls.{csv,summary.json}` — matched-Δ control table (temperature n=23 source pairs, age n=23 source pairs, `match_mode=domain_z`).
- `gradable_cross_domain_low_rank_control_l20_r5_gemma3.{md,csv,summary.json}` — Test 1 results.
- `gradable_cross_domain_subspace_transfer_l20_r5_gemma3.{md,csv,summary.json}` — Test 2 source-state-basis results.
- `gradable_cross_domain_delta_basis_transfer_l20_r5_gemma3.{md,csv,summary.json}` — Test 3 source-delta-basis results.

### A5. Immediate Experiment Path

The old v0 temperature file was useful as a plumbing check, but it is superseded for primary interpretation by the v1.2/v1.3 results in A4.1. The current immediate path is factor isolation before activation geometry.

All datasets should continue to follow the existing `DisambPair` surface:

```json
{
  "pair_id": "...",
  "target": "45 degrees Celsius",
  "target_occurrence": 0,
  "a": {"prompt": "...", "expected_label": "cool"},
  "b": {"prompt": "...", "expected_label": "hot"},
  "choices": {"cold": [...], "cool": [...], "warm": [...], "hot": [...]},
  "metadata": {...}
}
```

**Status of the analysis chain (as of 2026-05-27):**
- Behavioral regime identified (§A4.1): v1.2-form prompts, size domain is the only paraphrase-robust case.
- Activation geometry extracted and analyzed (`gradable_size_geometry_broad_final_token_gemma3.*`): ρ is linearly decodable at r ≈ 0.98 in L16–L20, but full-residual distance does not metrically order ρ (see the geometry analysis in `gradable_size_geometry_*.analysis.md`).
- Raw causal patching at the final prompt token: positive across L20–L24, peak at L20 (`gradable_size_v2_*_raw_patch_*`).
- **Low-rank causal subspace patching: completed bidirectionally; fictional → iso_ratio and iso_ratio → fictional both pass the L20 pca-k5 gate (§A4.2).**

Recommended next analyses, in priority order:

1. **Cross-domain causal specificity probes** — completed (§A4.3). Current result: the size basis is not strictly size-specific, but source-domain bases do not robustly substitute for it. The claim is cross-domain activation of a domain-preferential size basis, not a single shared rank-5 subspace.
2. **Same-domain different-pair same-ρ donor control** — donor drawn from a different size pair with ρ matched to receiver. Expected: near-zero `aligned_effect` if same-pair prompt-family transport is not driving the result.
3. **Subspace-overlap diagnostic** — completed (`gradable_manifold_claim_diagnostics.md`). L20 cross-variant principal angles are 33.1°/49.0°/74.2°/77.7°/84.9° with projection overlap 0.252 (random ≈ 0.002). Bidirectional transfer corresponds to *shared but not identical* fitted bases, not aligned identity; overlap is strongest at L8 (0.393) and falls by L20.
4. **Paraphrase robustness of the discovered subspace** — fit PCA on v1.2/adjective-counts activations, evaluate patching on v1.3 a/b/c or matched paraphrase activations. Tests whether the subspace is calibration-specific or prompt-form-specific.
5. **SAE/CLT feature-group writeback** — justified only after the current causal controls stabilize, and must compare full, group, same-size top-k, random, norm-matched random, and sham under held-out evaluation.
6. **v1.2 prompt-factor ablations**: remove `Compared only`, replace `feels` with `is`/`called`, remove repeated context binding — isolates which words in the v1.2 template are load-bearing for the *behavioral* signal.
7. **Neutral-context controls** using Class A/B/C-style contexts with the same numeric standards — directly tests the stereotype-anchoring hypothesis of §A4.1 (Temperature-B inversion).
8. **Artificial or inverted standard controls** to decouple explicit norms from world knowledge.
9. **Age lexical controls** to test whether the `old` prior is suppressing the ordered-score signal.
10. **Second model replication on size v1.2/adjective-counts** — a single comparable ~4B model would substantially raise the external-validity of the entire chain.


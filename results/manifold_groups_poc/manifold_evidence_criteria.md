# What Would Count As Manifold Evidence?

This project should not use "manifold" as a metaphor for any cluster or feature group. The evidence has to be staged.

## Core Distinction

Behavioral standard-sensitivity is not manifold evidence by itself.

```text
Behavioral result:
  ordered probability mass tracks value relative to a comparison standard.

Representational result:
  internal states organize this behavior as a low-dimensional, smooth,
  context-calibrated structure.

Causal result:
  interventions on that structure preserve or steer held-out behavior better
  than top-k and random controls.
```

The first result makes the domain worth studying. The second makes a manifold claim plausible. The third makes it mechanistically useful.

## Evidence Ladder

### 1. Stable Behavioral Target

Before geometry, identify a regime where the behavior is stable.

Required:

```text
unique-side r(ordered_score, relative_position) >= 0.40
bootstrap CI excludes 0
within-target permutation p is small
effect survives at least one paraphrase or neutral-context variant
argmax priors do not define the headline metric
```

Current status:

```text
v1.2 passes for temperature, size, age.
v1.3 shows size is robust, temperature is prompt-sensitive, age loses signal.
```

Therefore, size is the first positive candidate; temperature and age are stress tests.

### 2. Low-Dimensional Geometry

For each prompt side, extract hidden states at:

```text
scalar span
comparison/standard span
final pre-label token
optional teacher-forced label token
```

A manifold claim requires more than a 2D plot. It should show:

```text
low intrinsic dimension relative to ambient residual dimension
smooth trajectories as scalar value changes within a fixed standard
systematic standard-shift vectors for the same scalar value across contexts
local neighborhoods preserving semantic adjacency
held-out decodability of value, standard, and relative position
```

Useful tests:

```text
PCA / participation-ratio / intrinsic-dimension estimates
linear and low-rank probes for value, standard, relative_position
cross-validated R^2 for ordered_score from low-dimensional coordinates
representational similarity: activation distance vs semantic-grid distance
local tangent PCA: stable low-dimensional tangent spaces across the grid
smoothness: neighboring scalar values map to neighboring hidden states
```

Visualization with PCA/UMAP is only exploratory. It is not evidence unless paired with quantitative out-of-sample tests.

### 3. Factor Structure

The geometry should reflect the task factors, not just labels.

Minimum tests:

```text
value can be decoded across held-out standards
standard can be decoded across held-out values
relative_position can be decoded across held-out contexts
label probabilities are predicted by relative_position better than by raw value alone
neutral-context and artificial-standard controls preserve the factor relation
```

Strong evidence:

```text
Train on natural contexts, test on neutral contexts.
Train on some standards, test on held-out standards.
Train on one wording, test on a paraphrase that behaviorally passes.
```

If factor probes only work inside one exact prompt template, call it prompt-conditioned geometry, not a general semantic manifold.

### 4. Causal Control

A manifold-like representation becomes mechanistically relevant only if interventions on it control behavior.

Raw activation tests:

```text
patch scalar-coordinate states between values while holding standard fixed
patch standard/context states while holding value fixed
patch final pre-label states to test readout-stage control
```

Expected pattern:

```text
scalar patches move ordered_score along the label scale
standard patches shift the calibration boundary
sham patches stay near zero
effects generalize to held-out values/contexts/templates
```

### 5. SAE/CLT Group Validation

Feature groups are not explanations by default. They must be discovered on a selection split and validated on held-out behavior.

Discovery options:

```text
coactivation neighborhoods over behaviorally stable prompts
effect-profile groups over scalar/standard intervention effects
residual-coverage groups for the ordered-score signal
graph clusters over feature activation profiles across the semantic grid
```

Validation arms:

```text
full writeback
candidate group
same-size top-k
same-size random groups
sham / identity
```

Go signal:

```text
candidate group preserves or steers held-out ordered_score better than
same-size top-k and matched random groups, with sham near zero.
```

No-Go outcomes are still useful:

```text
top-k > groups:
  behavior may be rank-local rather than manifold-group-local.

groups ~= random:
  current discovery method does not capture behavioral control.

geometry positive, intervention negative:
  direct extension of the companion SAE/CLT writeback limitation paper (`https://github.com/Sapphire-Bridge/sae-writeback-limitation/blob/main/paper/sae_writeback_limitation_short_paper.md`): geometric structure is not enough.

natural contexts positive, neutral contexts negative:
  effect is world-anchor dependent, not explicit-standard calibration.
```

## What Would Be The Strongest Result?

The strongest practical demonstration would be:

```text
1. Size shows paraphrase/neutral-context robust ordered-score behavior.
2. Hidden states over size form a low-dimensional grid where coordinates track
   scalar value and explicit standard.
3. A group discovered on selection prompts preserves or steers held-out
   ordered-score behavior better than same-size top-k and random controls.
4. Temperature or age fails one of these stages, explaining why not every
   gradable domain yields a behaviorally valid manifold group.
```

This would support a careful claim:

> Gradable predicates can induce behaviorally relevant, low-dimensional semantic control structures, but those structures are prompt- and readout-conditioned and require held-out intervention validation before being treated as explanatory feature groups.

## What Would Not Be Enough?

Insufficient:

```text
PCA plot looks curved
UMAP clusters by label
linear probe decodes labels
SAE features coactivate in the same prompts
high reconstruction cosine
argmax accuracy improves
```

All of these can be compatible with a manifold interpretation, but none establishes it without behavioral generalization and causal validation.

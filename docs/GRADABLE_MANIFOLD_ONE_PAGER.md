# Finding the Relevant Activation Geometry for Gradable Judgments

A one-page note plus technical appendix on using gradable-adjective semantics to
select a candidate local activation geometry in Gemma-3-4B, and a causal test of
whether that geometry transfers across size operationalizations.

## One-Page Note

### The simple idea

Manifold-steering work studies how to steer once we have a meaningful activation
or behavior manifold. This note asks the prior selection question: where does the
meaningful geometry come from?

The danger is that many low-dimensional geometries can be found in activations. A
PCA basis is easy to fit, but most PCA bases are not the right object to steer.
The missing step is a way to decide which contrast is semantically worth testing.

Gradable adjectives give a clean case. For a word like "large", the relevant
judgment is not just the object's absolute size. A 3 cm object can be large for
one class and small for another. The judgment depends on the value relative to a
standard.

So the semantic theory suggests the coordinate:

```text
rho = log(value / standard)
```

The point is not that this equation is the neural mechanism. The point is that it
tells us what contrast the neural mechanism should support if it is really
representing standard-relative size.

### What I tested

I built paired prompts in which the model has to judge size relative to an
explicit standard. The labels are ordered:

```text
tiny < small < large < huge
```

The main comparison uses two different size operationalizations:

- one with fictional rod-like objects and explicit standards;
- one iso-ratio variant that preserves the same standard-relative structure in a
  different surface form.

The model's behavior is first checked against the semantic variable. Then the
causal test asks whether a small residual-stream subspace can transfer the
judgment shift from one dataset to the other.

### What happens

In the current Gemma-3-4B size prompt/readout regime, a prompt-deduplicated
ordered-probability readout tracks the standard-relative variable. This is not an
argmax-accuracy claim. More importantly, a small layer-20 subspace carries a
causal part of the effect.

In plain terms:

- If I fit a rank-5 PCA subspace from one size dataset, patching through that
  subspace changes judgments in the held-out size dataset.
- The transfer works in both directions.
- In donor-conditioned cross-variant patching, the effect is much stronger in
  aligned effect/recovery than random subspaces and explicit-variable controls
  like absolute `value`, `standard`, or a one-dimensional `rho` vector.
- This means the result is not just "there is a rho direction." It is a small
  local geometry found by using the semantic contrast to build the data.

The main causal result is:

> A layer-20 rank-5 size-calibration subspace transfers standard-relative size
> judgments across two different size operationalizations.

### Why this connects to manifold steering

I see this as complementary to manifold steering rather than competitive with it.

The manifold-steering framing says, roughly: once we have the relevant
activation/behavior manifold, steering should respect it.

This result says: semantic theory can help identify which activation geometry is
a plausible candidate in the first place.

That matters because the geometry is not trivial. The two independently fitted
layer-20 size bases are not literally the same subspace. Nevertheless, causal
transfer works. So the result is not "same basis, same manifold." It is closer
to:

> The model has a behaviorally causal size-calibration geometry that is stable
> enough for transfer, but not identical across dataset constructions.

### Cross-domain check

If there were a single interchangeable rank-5 gradability basis, then size,
temperature, and age should share the same low-rank basis. The current evidence
does not support that strong version.

Temperature and age deltas partially activate the size basis, but temperature- or
age-trained bases do not robustly replace the size basis. That gives a more
nuanced picture:

- not strictly size-only;
- not one universal rank-5 gradability basis;
- partial and asymmetric overlap across gradable domains.

This seems like a useful empirical stress test for a precondition of an
`M_h`/`M_y` study: semantic overlap gives partial activation transfer, but not
full geometric interchangeability across domains.

### Steering status

There is also a simple fixed-vector steering result. A direction derived from
high-`rho` minus low-`rho` size contrasts inside the layer-20 rank-5 basis
monotonically shifts held-out size judgments and beats random/sham controls.

I would describe this carefully:

> This is local linear residual-stream steering, not geodesic manifold steering.

It shows that the discovered geometry can be used for control, but it does not yet
test naturalness, off-manifold behavior, or manifold-respecting paths. The
steering controls also show that `standard` is a stronger local actuator in this
regime, so this is not a rho-only or uniquely PCA-mediated steering result.

### Main takeaway

The supported claim is:

> Theory-guided contrast design identifies a behaviorally causal low-rank
> layer-20 size-calibration geometry in Gemma-3-4B. This geometry transfers across
> two size operationalizations and supports local steering, while cross-domain
> probes show partial and asymmetric overlap with other gradable domains.

The contribution to manifold steering is upstream: it gives a method for choosing
a candidate local activation geometry before fitting or steering along a full
`M_h`.

The unsupported claims are just as important:

- not a universal gradability manifold;
- not a curved/geodesic manifold-steering result;
- not a rho-only direction;
- not yet prompt-general or paraphrase-invariant;
- not yet an SAE feature-group explanation;
- not yet model-general beyond this Gemma-3-4B setup.

## Technical Appendix

### Model and Readout

Model: `google/gemma-3-4b-pt` at revision `cc012e0`, residual stream, final
prompt token.

Primary labels:

```text
tiny < small < large < huge
```

Primary semantic variable:

```text
rho = log(value / standard)
```

Behavior is scored as prompt-deduplicated ordered probability mass over the label
set. These are not argmax-accuracy or exact-label-accuracy metrics.

### Behavioral Gate

| Dataset | Correlation with `rho` |
| --- | ---: |
| fictional semantic adjective counts | `r = 0.611 [0.531, 0.692]` |
| iso-ratio adjective counts | `r = 0.680 [0.514, 0.836]` |

### Low-Rank Causal Transfer

At layer 20, a rank-5 PCA subspace transfers across the two size
operationalizations:

| Direction | L20 pca-k5 aligned effect | Recovery vs full patch |
| --- | ---: | ---: |
| fictional -> iso | `+0.162 [0.110, 0.209]` | `0.860 [0.595, 1.146]` |
| iso -> fictional | `+0.155 [0.127, 0.187]` | `0.591 [0.492, 0.703]` |

Donor-conditioned patching controls:

- pca-k5 beats random and norm-matched random controls at L20.
- Simple explicit-variable controls are weak in aligned effect/recovery in the
  primary causal-patching setup.
- A one-dimensional `rho` control is weak: in the primary L20 run, `rho` recovery
  is about `0.017`, while pca-k5 recovery is about `0.860`.
- Direction-match alone is not the control metric; some explicit controls have
  high direction-match with tiny recovery in the reverse run.

### Geometry Diagnostics

The two independently fitted size bases are not literally the same L20 rank-5
subspace:

```text
L20 principal angles: 33.1, 49.0, 74.2, 77.7, 84.9 degrees
L20 subspace overlap: 0.252
random overlap: 0.002
```

Layer diagnostics:

```text
mean bidirectional pca-k5 aligned effect:
L16: 0.046
L20: 0.159
L24: 0.035
```

### Cross-Domain Stress Test

Matched temperature and age deltas projected through the size basis partially
move size judgments:

| Source delta through `U_size` | Aligned effect |
| --- | ---: |
| temperature -> size | `+0.081 [0.043, 0.118]` |
| age -> size | `+0.070 [0.042, 0.097]` |
| size control | `+0.162 [0.105, 0.215]` |

But source-trained temperature or age bases do not robustly replace the size
basis:

```text
temperature source-state basis -> size: +0.028 [0.017, 0.039]
age source-state basis -> size: +0.006 [-0.001, 0.013]
temperature source-delta basis -> size: +0.032 [0.020, 0.045]
age source-delta basis -> size: -0.000 [-0.005, 0.005]
```

### Fixed-Vector Steering

A fixed vector derived from high-`rho` minus low-`rho` size contrasts inside the
L20 rank-5 basis gives monotone residual-stream steering on held-out size
prompts:

```text
pca_delta_mean slope/alpha: +0.029 [0.026, 0.032]
random norm-matched slope/alpha: +0.005 [0.003, 0.008]
sham: 0.000
standard slope/alpha: +0.044 [0.042, 0.046]
```

Interpretation:

- this supports local linear residual-stream steering;
- the pca-derived direction beats random/sham but does not beat the `standard`
  steering control;
- it is not a rho-only or uniquely PCA-mediated steering result;
- it does not establish geodesic manifold steering;
- it does not test generated naturalness or off-manifold trajectories.

### Supporting Artifacts

- [`../README.md`](../README.md) - full claim surface, reproduce commands, and
  claim discipline.
- `results/manifold_groups_poc/gradable_manifold_claim_diagnostics.md` - the
  data-derived claim diagnostics table.
- `figures/manifold_groups/l20_pca_rho_projection.svg`
- `figures/manifold_groups/cross_variant_subspace_overlap.svg`
- `figures/manifold_groups/transfer_efficiency_by_delta_rho.svg`
- `figures/manifold_groups/layer_trajectory.svg`
- `results/manifold_groups_poc/gradable_size_semantic_steering_l20_r5_gemma3.md`

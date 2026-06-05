# Gradable v1.2 Interpretation

## Status

This is a behavioral prerequisite result for the gradable-predicate manifold hypothesis. It is not yet evidence for a neural manifold.

v1.2 fixes the v1.1 pair-orientation artifact by balancing expected ordered-label shifts. The behavioral effect survives unchanged.

## Main Results

Primary metrics use unique prompt sides, deduplicated by prompt hash.

| domain | unique sides | expected shift signs | direction match | r(score,predictor) | 95% bootstrap CI | within-target permutation p | argmax behavior |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| temperature | 58 | +44 / -40 | 70/84 = 0.833 | 0.678 | [0.577, 0.763] | 0.00010 | mixed |
| size | 62 | +42 / -43 | 70/85 = 0.824 | 0.652 | [0.507, 0.769] | 0.00005 | mostly `small` |
| age | 46 | +36 / -36 | 62/72 = 0.861 | 0.642 | [0.397, 0.813] | 0.00015 | always `old` |

Predictor:

```text
temperature: value - explicit standard
size:        log(value / explicit standard)
age:         log(value / explicit standard)
```

## Interpretation

The robust finding is not high argmax accuracy. Size and age remain argmax-degenerate. The robust finding is that ordered probability mass tracks the standard-relative scalar position across three gradable domains.

This supports the Bierwisch/Kennedy setup as an empirical testbed:

```text
scalar value + comparison-class standard -> graded ordered-label redistribution
```

The result is stronger than v1.1 because pair orientation is now balanced:

```text
temperature expected shifts: positive 44, negative 40
size expected shifts:        positive 42, negative 43
age expected shifts:         positive 36, negative 36
```

So the effect is not an artifact of always placing the lower-standard context on side A and the higher-standard context on side B.

## What This Allows Us To Claim

Allowed:

```text
Gemma-3-4B shows robust, moderate standard-sensitive ordered-probability shifts on explicit-standard gradable predicate prompts across temperature, size, and age.
```

Allowed:

```text
Argmax label choice is the wrong primary readout for this domain; ordered probability mass is the behaviorally informative readout.
```

Not yet allowed:

```text
The model represents gradable predicates as manifolds.
```

Not yet allowed:

```text
SAE/CLT feature groups preserve the relevant gradable-predicate control structure.
```

## Go / No-Go

Go to the next stage.

The behavioral gate is passed:

```text
orientation balanced: yes, all domains
r(score,predictor) >= 0.40: yes, all domains
bootstrap CI excludes 0: yes, all domains
within-target permutation remains small: yes, all domains
```

## Next Experiments

1. Paraphrase sweep.

   Keep the same scalar grids and label orders, but vary the prompt frame. The effect should survive several wording variants if it is not just a template artifact.

2. Artificial-standard sweep.

   Define synthetic comparison classes with explicit artificial norms. This tests whether the model follows the supplied standard rather than relying only on world knowledge.

3. Activation geometry.

   Extract residual-stream states at scalar span, standard/comparison span, and final pre-label token. Test whether value, standard, relative position, and ordered score are low-dimensional and smoothly decodable.

4. CLT/SAE writeback.

   Only after behavior and raw geometry are stable, test whether full CLT/SAE reconstruction and discovered feature groups preserve the ordered probability profile better than top-k and random controls.

## Bridge To The Companion Limitation Paper

The companion limitation paper's question (`https://github.com/Sapphire-Bridge/sae-writeback-limitation/blob/main/paper/sae_writeback_limitation_short_paper.md`) becomes sharper here:

```text
Does a reconstruction preserve only geometric closeness, or does it preserve contextual standard calibration?
```

A gradable-predicate group is behaviorally useful only if selected on a selection split and then, on held-out prompts, preserves or steers the ordered label distribution better than same-size top-k and matched random groups.

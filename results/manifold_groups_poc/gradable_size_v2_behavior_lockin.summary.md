# Size v2 Behavioral Lock-In Summary

Primary metric: unique-side `corr(ordered_score, log(value / standard))`.

| variant | n pairs | unique sides | r | 95% CI | perm. p | within-target p | direction match | argmax counts |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| natural | 85 | 62 | 0.652 | [0.516, 0.768] | 0.00005 | 0.00005 | 70/85=0.824 | large:2, small:60 |
| neutral | 85 | 62 | 0.025 | [-0.161, 0.223] | 0.84506 | 0.98220 | 44/85=0.518 | small:62 |
| iso_ratio | 23 | 22 | 0.607 | [0.379, 0.812] | 0.00290 | 0.03015 | 17/23=0.739 | large:6, small:16 |
| artificial | 31 | 30 | 0.306 | [0.063, 0.555] | 0.10094 | 0.52132 | 15/31=0.484 | large:6, small:24 |
| fictional_semantic | 85 | 62 | 0.462 | [0.343, 0.577] | 0.00030 | 0.09655 | 64/85=0.753 | huge:10, large:6, small:46 |
| counter_natural | 31 | 30 | 0.793 | [0.709, 0.866] | 0.00005 | 0.00085 | 24/31=0.774 | huge:14, large:2, small:14 |

## Iso-Ratio Diagnostics

Rows with the same ratio should have similar ordered scores across different absolute values/standards. This is diagnostic, not a pass/fail gate by itself.

| ratio | n | mean ordered score | sd ordered score | values | standards |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.1875 | 2 | 1.488 | 0.000 | 45 | 240 |
| 0.25 | 1 | 1.210 | 0.000 | 60 | 240 |
| 0.375 | 5 | 1.288 | 0.027 | 45,90 | 120,240 |
| 0.5 | 3 | 1.270 | 0.076 | 60,120 | 120,240 |
| 0.75 | 9 | 1.218 | 0.086 | 45,90,180 | 60,120,240 |
| 1.5 | 12 | 1.662 | 0.118 | 45,90,180,360 | 30,60,120,240 |
| 2 | 3 | 1.579 | 0.062 | 60,120 | 30,60 |
| 3 | 6 | 1.639 | 0.032 | 90,180,360 | 30,60,120 |
| 4 | 1 | 1.560 | 0.000 | 120 | 30 |
| 6 | 3 | 1.447 | 0.067 | 180,360 | 30,60 |
| 12 | 1 | 1.633 | 0.000 | 360 | 30 |

Gate interpretation:

```text
Go to confirmatory raw patching only if natural, iso_ratio, and at least one semantic-control variant pass.
Semantic-control variants: fictional_semantic and counter_natural.
Use r >= 0.40 with CI excluding 0 as the strict behavior gate.
Treat artificial/inverted standards as a stress test; partial survival is already informative.
If bare neutral fails but fictional_semantic passes, interpret the effect as semantically scaffolded calibration, not context-free ratio arithmetic.
If iso_ratio fails, do not proceed to geometry yet.
```

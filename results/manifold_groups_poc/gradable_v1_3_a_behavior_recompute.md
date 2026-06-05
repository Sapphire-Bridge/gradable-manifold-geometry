# Gradable Behavior Recompute

- Version: `v1_3_a`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(score,predictor) | r(score,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| temperature | 84 | 58 | 0.310 | 27/84=0.321 | negative:38, positive:46 | 0.261 | 0.167 | cold:5, hot:23, warm:30 |
| size | 85 | 62 | 0.113 | 63/85=0.741 | negative:36, positive:49 | 0.493 | 0.567 | large:8, small:54 |
| age | 72 | 46 | 0.304 | 20/72=0.278 | negative:36, positive:36 | -0.071 | -0.053 | mature:8, old:29, young:8, youthful:1 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

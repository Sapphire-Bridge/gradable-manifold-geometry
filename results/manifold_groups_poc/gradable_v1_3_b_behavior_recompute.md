# Gradable Behavior Recompute

- Version: `v1_3_b`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(score,predictor) | r(score,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| temperature | 84 | 58 | 0.259 | 1/84=0.012 | negative:39, positive:45 | -0.584 | -0.461 | cold:14, hot:43, warm:1 |
| size | 85 | 62 | 0.113 | 68/85=0.800 | negative:43, positive:42 | 0.634 | 0.691 | large:12, small:50 |
| age | 72 | 46 | 0.435 | 18/72=0.250 | negative:30, positive:42 | -0.128 | -0.024 | old:46 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

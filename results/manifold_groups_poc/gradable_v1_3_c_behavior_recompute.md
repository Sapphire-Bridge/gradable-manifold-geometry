# Gradable Behavior Recompute

- Version: `v1_3_c`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(score,predictor) | r(score,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| temperature | 84 | 58 | 0.414 | 25/84=0.298 | negative:43, positive:41 | -0.217 | -0.008 | cold:36, hot:21, warm:1 |
| size | 85 | 62 | 0.081 | 65/85=0.765 | negative:46, positive:39 | 0.665 | 0.662 | large:1, small:61 |
| age | 72 | 46 | 0.435 | 31/72=0.431 | negative:35, positive:37 | 0.100 | 0.072 | old:46 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

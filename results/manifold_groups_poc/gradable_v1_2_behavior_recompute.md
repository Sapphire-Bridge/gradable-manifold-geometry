# Gradable Behavior Recompute

- Version: `v1_2`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(score,predictor) | r(score,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| temperature | 84 | 58 | 0.655 | 70/84=0.833 | negative:40, positive:44 | 0.678 | 0.846 | cold:31, cool:1, hot:8, warm:18 |
| size | 85 | 62 | 0.081 | 70/85=0.824 | negative:43, positive:42 | 0.652 | 0.660 | large:2, small:60 |
| age | 72 | 46 | 0.435 | 62/72=0.861 | negative:36, positive:36 | 0.642 | 0.698 | old:46 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

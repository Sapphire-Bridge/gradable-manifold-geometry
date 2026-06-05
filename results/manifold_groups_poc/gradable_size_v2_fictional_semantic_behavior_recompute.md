# Gradable Behavior Recompute

- Version: `v2_fictional_semantic`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(score,predictor) | r(score,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| size | 85 | 62 | 0.242 | 64/85=0.753 | negative:47, positive:38 | 0.462 | 0.689 | huge:10, large:6, small:46 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

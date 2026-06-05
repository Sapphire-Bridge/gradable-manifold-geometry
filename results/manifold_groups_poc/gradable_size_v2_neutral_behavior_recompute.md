# Gradable Behavior Recompute

- Version: `v2_neutral`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(score,predictor) | r(score,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| size | 85 | 62 | 0.081 | 44/85=0.518 | negative:37, positive:48 | 0.025 | 0.228 | small:62 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

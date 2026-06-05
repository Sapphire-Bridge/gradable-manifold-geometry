# Gradable Behavior Recompute

- Version: `v2_fictional_semantic_adjective_counts`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(ordered,predictor) | r(signed,predictor) | r(ordered,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| size | 85 | 62 | 0.194 | 61/85=0.718 | negative:48, positive:37 | 0.611 | 0.602 | 0.815 | large:37, small:25 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

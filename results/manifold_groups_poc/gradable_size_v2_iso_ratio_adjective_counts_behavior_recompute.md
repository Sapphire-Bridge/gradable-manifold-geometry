# Gradable Behavior Recompute

- Version: `v2_iso_ratio_adjective_counts`
- Bootstrap resamples: `5000`
- Permutations: `20000`
- Primary side metrics deduplicate repeated prompts by `prompt_hash`.

| domain | n pairs | unique sides | unique side acc | direction match | expected signs | r(ordered,predictor) | r(signed,predictor) | r(ordered,label idx) | argmax counts |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| size | 23 | 22 | 0.273 | 16/23=0.696 | negative:10, positive:13 | 0.680 | 0.669 | 0.760 | large:9, small:13 |

Notes:
- `temperature` uses `value - standard` as predictor.
- `size` and `age` use `log(value / standard)` as predictor.
- Direction-match Wilson/binomial values are marked naive because pair rows share prompts and scalar values.
- If `expected signs` has only one sign, pair-level orientation is degenerate; use side-level correlations as the primary metric and regenerate a balanced dataset before reporting pair-level sign nulls.

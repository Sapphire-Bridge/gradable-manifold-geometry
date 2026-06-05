# aom

Mechanistic-interpretability primitives for the gradable manifold geometry
study (see the repository [`README.md`](../README.md)).

- `interventions/` — activation patching and low-rank residual-stream patching
- `metrics/` — scoring utilities for ordered-label readouts
- `models/` — model and tokenizer loading
- `data/` — dataset loaders and schema helpers
- `provenance/`, `repro.py` — reproducibility metadata
- `token_spans.py` — target substring to token-index span alignment

The gradable analysis scripts that use these primitives live in `scripts/`.

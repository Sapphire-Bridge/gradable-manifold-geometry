#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${PYTHON:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
LAYERS="${LAYERS:-20,24,28,32,33}"
VARIANTS="${VARIANTS:-fictional_semantic_adjective_counts,iso_ratio_adjective_counts}"
RIDGE_ALPHA="${RIDGE_ALPHA:-10.0}"
PCA_RANKS="${PCA_RANKS:-1,2,3,4,5}"
OUT_PREFIX_RAW="${OUT_PREFIX:-results/manifold_groups_poc/gradable_size_geometry_late_final_token_gemma3}"
OUT_PREFIX="$(printf '%s' "${OUT_PREFIX_RAW}" | tr -d '\r\n' | sed 's#/  *#/#g')"

"${PYTHON}" scripts/extract_gradable_size_activations.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --revision "${REVISION}" \
  --tokenizer_revision "${TOKENIZER_REVISION}" \
  --variants "${VARIANTS}" \
  --layers "${LAYERS}" \
  --site final_prompt_token \
  --device "${DEVICE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --out_prefix "${OUT_PREFIX}" \
  --overwrite

"${PYTHON}" scripts/analyze_gradable_size_geometry.py \
  --npz "${OUT_PREFIX}.npz" \
  --metadata_csv "${OUT_PREFIX}.metadata.csv" \
  --layers "${LAYERS}" \
  --pca_ranks "${PCA_RANKS}" \
  --ridge_alpha "${RIDGE_ALPHA}" \
  --out_json "${OUT_PREFIX}.analysis.json" \
  --out_md "${OUT_PREFIX}.analysis.md"

echo "Done. Read ${OUT_PREFIX}.analysis.md"

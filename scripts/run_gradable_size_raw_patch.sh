#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
VARIANT="${VARIANT:-natural}"
LAYERS="${LAYERS:-4,8}"
PATCH_SITE="${PATCH_SITE:-target}"
BOOTSTRAP_B="${BOOTSTRAP_B:-1000}"

VERSION="v2_${VARIANT}"
DATA_PATH="results/manifold_groups_poc/gradable_size_disamb_pairs_${VERSION}.jsonl"
if [[ "${PATCH_SITE}" == "target" ]]; then
  OUT="results/manifold_groups_poc/gradable_size_${VERSION}_raw_patch_l${LAYERS//,/}_gemma3"
else
  OUT="results/manifold_groups_poc/gradable_size_${VERSION}_raw_patch_${PATCH_SITE}_l${LAYERS//,/}_gemma3"
fi

python scripts/patch_gradable_size_raw.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --revision "${REVISION}" \
  --tokenizer_revision "${TOKENIZER_REVISION}" \
  --data_path "${DATA_PATH}" \
  --layers "${LAYERS}" \
  --patch_site "${PATCH_SITE}" \
  --device "${DEVICE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --bootstrap_B "${BOOTSTRAP_B}" \
  --out_csv "${OUT}.csv" \
  --out_summary "${OUT}.summary.json" \
  --overwrite

echo "Done. Read ${OUT}.summary.json"

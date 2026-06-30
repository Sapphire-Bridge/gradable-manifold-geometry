#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${PYTHON:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
READOUT_FAMILIES="${READOUT_FAMILIES:-adjective,adjective_counts,normality4,binary,comparative4,comparative2}"
OUT_PATH_RAW="${PRIOR_OUT:-results/manifold_groups_poc/gradable_size_prior_diagnostics_gemma3}"
OUT_PATH="$(printf '%s' "${OUT_PATH_RAW}" | tr -d '\r\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

"${PYTHON}" scripts/score_gradable_size_priors.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --revision "${REVISION}" \
  --tokenizer_revision "${TOKENIZER_REVISION}" \
  --readout_families "${READOUT_FAMILIES}" \
  --device "${DEVICE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --out_csv "${OUT_PATH}.csv" \
  --out_summary "${OUT_PATH}.summary.json" \
  --overwrite

echo "Done. Read ${OUT_PATH}.summary.json"

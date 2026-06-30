#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${PYTHON:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

TRAIN_VARIANT="${TRAIN_VARIANT:-fictional_semantic_adjective_counts}"
EVAL_VARIANT="${EVAL_VARIANT:-iso_ratio_adjective_counts}"
LAYERS="${LAYERS:-20}"
METHODS="${METHODS:-pca_delta_mean,value,standard,value_standard_2d,random_norm_matched,sham}"
RANK="${RANK:-5}"
ALPHAS="${ALPHAS:--2,-1,-0.5,0,0.5,1,2}"
BOOTSTRAP_B="${BOOTSTRAP_B:-1000}"
RIDGE_ALPHA="${RIDGE_ALPHA:-10.0}"
RANDOM_REPEATS="${RANDOM_REPEATS:-20}"
DRY_RUN="${DRY_RUN:-0}"

ACTIVATIONS_PREFIX_RAW="${ACTIVATIONS_PREFIX:-results/manifold_groups_poc/gradable_size_geometry_broad_final_token_gemma3}"
ACTIVATIONS_PREFIX="$(printf '%s' "${ACTIVATIONS_PREFIX_RAW}" | tr -d '\r\n' | sed 's#/  *#/#g')"

DATA_PATH="results/manifold_groups_poc/gradable_size_disamb_pairs_v2_${EVAL_VARIANT}.jsonl"
OUT_PREFIX_RAW="${OUT_PREFIX:-results/manifold_groups_poc/gradable_size_semantic_steering_l${LAYERS//,/}_r${RANK}_gemma3}"
OUT_PREFIX="$(printf '%s' "${OUT_PREFIX_RAW}" | tr -d '\r\n' | sed 's#/  *#/#g')"

STEER_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --revision "${REVISION}"
  --tokenizer_revision "${TOKENIZER_REVISION}"
  --data_path "${DATA_PATH}"
  --activations_npz "${ACTIVATIONS_PREFIX}.npz"
  --metadata_csv "${ACTIVATIONS_PREFIX}.metadata.csv"
  --train_variant "${TRAIN_VARIANT}"
  --eval_variant "${EVAL_VARIANT}"
  --layers "${LAYERS}"
  --methods "${METHODS}"
  --rank "${RANK}"
  --alphas="${ALPHAS}"
  --ridge_alpha "${RIDGE_ALPHA}"
  --random_repeats "${RANDOM_REPEATS}"
  --device "${DEVICE}"
  --torch_dtype "${TORCH_DTYPE}"
  --bootstrap_B "${BOOTSTRAP_B}"
  --out_csv "${OUT_PREFIX}.csv"
  --out_summary "${OUT_PREFIX}.summary.json"
  --out_md "${OUT_PREFIX}.md"
  --overwrite
)
if [ "${DRY_RUN}" = "1" ]; then
  STEER_ARGS+=(--dry_run)
fi

"${PYTHON}" scripts/steer_gradable_size.py "${STEER_ARGS[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "Dry run complete."
else
  echo "Done. Read ${OUT_PREFIX}.md"
fi

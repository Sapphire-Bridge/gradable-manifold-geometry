#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

SIZE_TRAIN_VARIANT="${SIZE_TRAIN_VARIANT:-fictional_semantic_adjective_counts}"
SIZE_EVAL_DATA_PATH="${SIZE_EVAL_DATA_PATH:-results/manifold_groups_poc/gradable_size_disamb_pairs_v2_iso_ratio_adjective_counts.jsonl}"
SOURCE_SPECS="${SOURCE_SPECS:-temperature=results/manifold_groups_poc/gradable_temperature_disamb_pairs_v1_2.jsonl,age=results/manifold_groups_poc/gradable_age_disamb_pairs_v1_2.jsonl}"
SOURCE_DOMAINS="${SOURCE_DOMAINS:-temperature,age}"

ACTIVATIONS_PREFIX_RAW="${ACTIVATIONS_PREFIX:-results/manifold_groups_poc/gradable_size_geometry_broad_final_token_gemma3}"
ACTIVATIONS_PREFIX="$(printf '%s' "${ACTIVATIONS_PREFIX_RAW}" | tr -d '\r\n' | sed 's#/  *#/#g')"

LAYERS="${LAYERS:-20}"
RANK="${RANK:-5}"
ALPHAS="${ALPHAS:-1.0}"
BOOTSTRAP_B="${BOOTSTRAP_B:-1000}"
RANDOM_REPEATS="${RANDOM_REPEATS:-20}"
MATCH_MODE="${MATCH_MODE:-domain_z}"
MIN_UNIQUE_SOURCE_PAIRS="${MIN_UNIQUE_SOURCE_PAIRS:-20}"
MIN_SIZE_PAIRS="${MIN_SIZE_PAIRS:-20}"
MAX_ABS_MATCH_ERROR="${MAX_ABS_MATCH_ERROR:-}"
DRY_RUN="${DRY_RUN:-0}"

CONTROL_PREFIX_RAW="${CONTROL_PREFIX:-results/manifold_groups_poc/gradable_cross_domain_matched_delta_rho_controls}"
CONTROL_PREFIX="$(printf '%s' "${CONTROL_PREFIX_RAW}" | tr -d '\r\n' | sed 's#/  *#/#g')"
OUT_PREFIX_RAW="${OUT_PREFIX:-results/manifold_groups_poc/gradable_cross_domain_low_rank_control_l${LAYERS//,/}_r${RANK}_gemma3}"
OUT_PREFIX="$(printf '%s' "${OUT_PREFIX_RAW}" | tr -d '\r\n' | sed 's#/  *#/#g')"

MATCH_ARGS=(
  --size_data_path "${SIZE_EVAL_DATA_PATH}"
  --source_specs "${SOURCE_SPECS}"
  --match_mode "${MATCH_MODE}"
  --min_unique_source_pairs "${MIN_UNIQUE_SOURCE_PAIRS}"
  --out_csv "${CONTROL_PREFIX}.csv"
  --out_summary "${CONTROL_PREFIX}.summary.json"
  --overwrite
)
if [ -n "${MAX_ABS_MATCH_ERROR}" ]; then
  MATCH_ARGS+=(--max_abs_match_error "${MAX_ABS_MATCH_ERROR}")
fi

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN: skipping control regeneration; reusing existing ${CONTROL_PREFIX}.csv"
else
  python scripts/make_gradable_cross_domain_delta_rho_controls.py "${MATCH_ARGS[@]}"
fi

PATCH_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --revision "${REVISION}"
  --tokenizer_revision "${TOKENIZER_REVISION}"
  --size_data_path "${SIZE_EVAL_DATA_PATH}"
  --controls_csv "${CONTROL_PREFIX}.csv"
  --activations_npz "${ACTIVATIONS_PREFIX}.npz"
  --metadata_csv "${ACTIVATIONS_PREFIX}.metadata.csv"
  --size_train_variant "${SIZE_TRAIN_VARIANT}"
  --source_domains "${SOURCE_DOMAINS}"
  --layers "${LAYERS}"
  --rank "${RANK}"
  --alphas "${ALPHAS}"
  --random_repeats "${RANDOM_REPEATS}"
  --min_unique_source_pairs "${MIN_UNIQUE_SOURCE_PAIRS}"
  --min_size_pairs "${MIN_SIZE_PAIRS}"
  --device "${DEVICE}"
  --torch_dtype "${TORCH_DTYPE}"
  --bootstrap_B "${BOOTSTRAP_B}"
  --out_csv "${OUT_PREFIX}.csv"
  --out_summary "${OUT_PREFIX}.summary.json"
  --out_md "${OUT_PREFIX}.md"
  --overwrite
)
if [ "${DRY_RUN}" = "1" ]; then
  PATCH_ARGS+=(--dry_run)
fi

python scripts/patch_gradable_cross_domain_low_rank_control.py "${PATCH_ARGS[@]}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "Dry run complete."
else
  echo "Done. Read ${OUT_PREFIX}.md"
fi

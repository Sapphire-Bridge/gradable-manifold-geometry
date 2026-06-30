#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON="${PYTHON:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
RECOMPUTE_BOOTSTRAP_B="${RECOMPUTE_BOOTSTRAP_B:-5000}"
RECOMPUTE_PERMUTATION_B="${RECOMPUTE_PERMUTATION_B:-20000}"
VARIANTS="${VARIANTS:-natural neutral iso_ratio artificial fictional_semantic counter_natural}"
READOUT_FAMILY="${READOUT_FAMILY:-adjective}"

label_order_for_readout() {
  case "$1" in
    adjective) echo "tiny,small,large,huge" ;;
    adjective_counts) echo "tiny,small,large,huge" ;;
    normality4) echo "far_below_normal,below_normal,above_normal,far_above_normal" ;;
    binary) echo "below_normal,above_normal" ;;
    comparative4) echo "much_shorter_than_normal,slightly_shorter_than_normal,slightly_longer_than_normal,much_longer_than_normal" ;;
    comparative2) echo "shorter_than_normal,longer_than_normal" ;;
    *) echo "Unknown READOUT_FAMILY=$1" >&2; exit 2 ;;
  esac
}

run_variant() {
  local variant="$1"
  local suffix=""
  if [[ "${READOUT_FAMILY}" != "adjective" ]]; then
    suffix="_${READOUT_FAMILY}"
  fi
  local version="v2_${variant}${suffix}"
  local label_order
  label_order="$(label_order_for_readout "${READOUT_FAMILY}")"
  local data_path="results/manifold_groups_poc/gradable_size_disamb_pairs_${version}.jsonl"
  local out="results/manifold_groups_poc/gradable_size_${version}_behavior_gemma3"

  "${PYTHON}" scripts/generate_gradable_size_v2.py --variant "${variant}" --readout_family "${READOUT_FAMILY}"

  "${PYTHON}" scripts/score_gradable_predicates.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --revision "${REVISION}" \
    --tokenizer_revision "${TOKENIZER_REVISION}" \
    --data_path "${data_path}" \
    --label_order "${label_order}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --out_csv "${out}.sides.csv" \
    --out_pairs_csv "${out}.pairs.csv" \
    --out_summary "${out}.summary.json" \
    --overwrite

  "${PYTHON}" scripts/recompute_gradable_behavior_metrics.py \
    --version "${version}" \
    --domains "size" \
    --size_label_order "${label_order}" \
    --bootstrap_B "${RECOMPUTE_BOOTSTRAP_B}" \
    --permutation_B "${RECOMPUTE_PERMUTATION_B}" \
    --out_json "results/manifold_groups_poc/gradable_size_${version}_behavior_recompute.json" \
    --out_md "results/manifold_groups_poc/gradable_size_${version}_behavior_recompute.md" \
    --out_sides_csv "results/manifold_groups_poc/gradable_size_${version}_behavior_recompute.sides.csv" \
    --out_pairs_csv "results/manifold_groups_poc/gradable_size_${version}_behavior_recompute.pairs.csv"
}

for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

"${PYTHON}" scripts/summarize_gradable_size_v2.py \
  --out_md "results/manifold_groups_poc/gradable_size_v2_behavior_lockin.summary.md" \
  --out_json "results/manifold_groups_poc/gradable_size_v2_behavior_lockin.summary.json"

echo "Done. Read results/manifold_groups_poc/gradable_size_v2_behavior_lockin.summary.md"

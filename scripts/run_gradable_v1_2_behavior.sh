#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
RECOMPUTE_BOOTSTRAP_B="${RECOMPUTE_BOOTSTRAP_B:-5000}"
RECOMPUTE_PERMUTATION_B="${RECOMPUTE_PERMUTATION_B:-20000}"

python scripts/generate_gradable_temperature_v1_2.py
python scripts/generate_gradable_size_v1_2.py
python scripts/generate_gradable_age_v1_2.py

run_domain() {
  local domain="$1"
  local labels="$2"
  local data_path="results/manifold_groups_poc/gradable_${domain}_disamb_pairs_v1_2.jsonl"
  local out="results/manifold_groups_poc/gradable_${domain}_v1_2_behavior_gemma3"

  python scripts/score_gradable_predicates.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --revision "${REVISION}" \
    --tokenizer_revision "${TOKENIZER_REVISION}" \
    --data_path "${data_path}" \
    --label_order "${labels}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --out_csv "${out}.sides.csv" \
    --out_pairs_csv "${out}.pairs.csv" \
    --out_summary "${out}.summary.json" \
    --overwrite
}

run_domain "temperature" "cold,cool,warm,hot"
run_domain "size" "tiny,small,large,huge"
run_domain "age" "young,youthful,mature,old"

python scripts/recompute_gradable_behavior_metrics.py \
  --version "v1_2" \
  --bootstrap_B "${RECOMPUTE_BOOTSTRAP_B}" \
  --permutation_B "${RECOMPUTE_PERMUTATION_B}" \
  --out_json "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.json" \
  --out_md "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.md" \
  --out_sides_csv "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.sides.csv" \
  --out_pairs_csv "results/manifold_groups_poc/gradable_v1_2_behavior_recompute.pairs.csv"

echo "Done. Read results/manifold_groups_poc/gradable_v1_2_behavior_recompute.md"

#!/usr/bin/env bash
# v1.3 paraphrase sweep: 3 paraphrases x 3 domains = 9 scoring runs.
# Reuses the existing scorer + recompute, just feeds different generated data.
# Designed to run continuously (~35-45 min on MPS) and not block on any single failure.

set -u
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-cc012e0a6d0787b4adcc0fa2c4da74402494554d}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$REVISION}"
DEVICE="${DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
RECOMPUTE_BOOTSTRAP_B="${RECOMPUTE_BOOTSTRAP_B:-5000}"
RECOMPUTE_PERMUTATION_B="${RECOMPUTE_PERMUTATION_B:-20000}"

LOG_DIR="$ROOT/results/manifold_groups_poc/paraphrase_sweep_logs"
mkdir -p "$LOG_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TOP_LOG="$LOG_DIR/v1_3_sweep_${STAMP}.log"

label_order_for () {
  case "$1" in
    temperature) echo "cold,cool,warm,hot" ;;
    size)        echo "tiny,small,large,huge" ;;
    age)         echo "young,youthful,mature,old" ;;
    *)           echo "" ; return 1 ;;
  esac
}

DOMAINS="temperature size age"
PARAPHRASES="a b c"

echo "v1.3 paraphrase sweep started at $(date -u)" | tee -a "$TOP_LOG"

generate_one () {
  local domain="$1"
  local paraphrase="$2"
  local log="$LOG_DIR/gen_${domain}_${paraphrase}_${STAMP}.log"
  "${PYTHON}" scripts/generate_gradable_v1_3.py --domain "$domain" --paraphrase "$paraphrase" >"$log" 2>&1
  return $?
}

score_one () {
  local domain="$1"
  local paraphrase="$2"
  local version="v1_3_${paraphrase}"
  local data="results/manifold_groups_poc/gradable_${domain}_disamb_pairs_${version}.jsonl"
  local out="results/manifold_groups_poc/gradable_${domain}_${version}_behavior_gemma3"
  local log="$LOG_DIR/score_${domain}_${paraphrase}_${STAMP}.log"
  local labels="$(label_order_for "$domain")"

  echo "  [score] ${domain}/${paraphrase} start $(date -u +%H:%M:%SZ)" | tee -a "$TOP_LOG"
  "${PYTHON}" scripts/score_gradable_predicates.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --revision "$REVISION" \
    --tokenizer_revision "$TOKENIZER_REVISION" \
    --data_path "$data" \
    --label_order "$labels" \
    --device "$DEVICE" \
    --torch_dtype "$TORCH_DTYPE" \
    --out_csv "${out}.sides.csv" \
    --out_pairs_csv "${out}.pairs.csv" \
    --out_summary "${out}.summary.json" \
    --overwrite >"$log" 2>&1
  local rc=$?
  echo "  [score] ${domain}/${paraphrase} end   $(date -u +%H:%M:%SZ) rc=$rc" | tee -a "$TOP_LOG"
  return $rc
}

recompute_paraphrase () {
  local paraphrase="$1"
  local version="v1_3_${paraphrase}"
  local out_base="results/manifold_groups_poc/gradable_${version}_behavior_recompute"
  local log="$LOG_DIR/recompute_${paraphrase}_${STAMP}.log"
  echo "  [recompute] ${paraphrase} start $(date -u +%H:%M:%SZ)" | tee -a "$TOP_LOG"
  "${PYTHON}" scripts/recompute_gradable_behavior_metrics.py \
    --version "$version" \
    --bootstrap_B "$RECOMPUTE_BOOTSTRAP_B" \
    --permutation_B "$RECOMPUTE_PERMUTATION_B" \
    --out_json "${out_base}.json" \
    --out_md "${out_base}.md" \
    --out_sides_csv "${out_base}.sides.csv" \
    --out_pairs_csv "${out_base}.pairs.csv" >"$log" 2>&1
  local rc=$?
  echo "  [recompute] ${paraphrase} end   $(date -u +%H:%M:%SZ) rc=$rc" | tee -a "$TOP_LOG"
  return $rc
}

for paraphrase in $PARAPHRASES; do
  echo "=== paraphrase ${paraphrase} ===" | tee -a "$TOP_LOG"
  for domain in $DOMAINS; do
    generate_one "$domain" "$paraphrase"
  done
  for domain in $DOMAINS; do
    score_one "$domain" "$paraphrase"
  done
  recompute_paraphrase "$paraphrase"
done

echo "=== sweep done at $(date -u) ===" | tee -a "$TOP_LOG"
echo "Per-paraphrase recompute summaries:" | tee -a "$TOP_LOG"
for paraphrase in $PARAPHRASES; do
  echo "  results/manifold_groups_poc/gradable_v1_3_${paraphrase}_behavior_recompute.md" | tee -a "$TOP_LOG"
done
exit 0

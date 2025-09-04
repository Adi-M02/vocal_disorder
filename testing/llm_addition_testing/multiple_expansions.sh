#!/usr/bin/env bash
# Alternating loop:
#   1) Word2Vec expansion (seed_json -> expansion/topk_*.json)
#   2) LLM evaluation of that expansions JSON (writes eval/batch_eval_*/new_seeds.json)
#   3) Next cycle uses that new_seeds.json as seed_json
#
# LOOPS counts full cycles (expansion + evaluation). The very first expansion
# uses INITIAL_EXPANSIONS as the seed_json input.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ───────────────────────────────
# Required config (override via env on the command line)
# ───────────────────────────────
MODEL_PATH="${MODEL_PATH:-/path/to/your/word2vec.model}"
INITIAL_EXPANSIONS="${INITIAL_EXPANSIONS:-/path/to/initial_seed_terms.json}"  # JSON list or {"seed_terms":[...]} or {cat:[...]}
LOOPS="${LOOPS:-5}"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/semantic_loop_$(date +%Y%m%d_%H%M%S)}"

# Python scripts (defaults relative to this script’s folder)
W2V_EXPAND_PY="${W2V_EXPAND_PY:-$SCRIPT_DIR/../../word2vec_expansion/ngram_generation/expand_seed_terms.py}"
LLM_EVAL_PY="${LLM_EVAL_PY:-$SCRIPT_DIR/LLM_semantic_addition_parallel.py}"

# LLM (Ollama) options
LLM_MODEL_ID="${LLM_MODEL_ID:-llama3.3:latest}"
LLM_URL="${LLM_URL:-http://localhost:11434/api/chat}"
LLM_TEMP="${LLM_TEMP:-0.0}"
LLM_TOKENS="${LLM_TOKENS:-2048}"
LLM_TIMEOUT="${LLM_TIMEOUT:-60}"
LLM_EXTRA="${LLM_EXTRA:-}"   # e.g. "--use_anchors --relation_mode"

# Word2Vec expansion options (blank → use script defaults)
TOPK="${TOPK:-}"        # e.g., 25
MIN_COS="${MIN_COS:-}"  # e.g., 0.4
EXTRA_W2V="${EXTRA_W2V:-}"

PY="${PY:-python3}"

# ───────────────────────────────
# Sanity checks
# ───────────────────────────────
require_file() { [[ -f "$1" ]] || { echo "[error] Missing file: $1" >&2; exit 1; }; }

require_file "$MODEL_PATH"
require_file "$INITIAL_EXPANSIONS"
require_file "$W2V_EXPAND_PY"
require_file "$LLM_EVAL_PY"

mkdir -p "$OUT_BASE"
echo "[info] Base run dir : $OUT_BASE"
echo "[info] Model        : $MODEL_PATH"
echo "[info] Cycles       : $LOOPS"

# ───────────────────────────────
# Helpers
# ───────────────────────────────
json_get() {
  # json_get <file> <key>  -> prints value of a string field
  local f="$1"; local key="$2"
  if command -v jq >/dev/null 2>&1; then
    jq -r --arg k "$key" '.[$k]' "$f"
  else
    "$PY" - "$f" "$key" <<'PYCODE'
import json, sys
p=sys.argv[1]; k=sys.argv[2]
with open(p,'r',encoding='utf-8') as f: d=json.load(f)
v=d.get(k,"")
print(v if isinstance(v,str) else "")
PYCODE
  fi
}

newest_batch_eval_dir() {
  # newest_batch_eval_dir <eval_dir_root>
  local d="$1"
  find "$d" -maxdepth 1 -type d -name 'batch_eval_*' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n1 | cut -d' ' -f2- || true
}

latest_file_in() {
  # latest_file_in <glob>
  # shellcheck disable=SC2086
  ls -dt $1 2>/dev/null | head -n1 || true
}

# ───────────────────────────────
# Main loop
# ───────────────────────────────
seeds_current="$INITIAL_EXPANSIONS"

for i in $(seq 1 "$LOOPS"); do
  cycle_dir="$OUT_BASE/cycle_${i}"
  mkdir -p "$cycle_dir"
  echo ""
  echo "==================== Cycle $i ===================="

  # A) EXPANSION (always first)
  cp -f "$seeds_current" "$cycle_dir/seeds_input.json"
  echo "[cycle $i] EXPANSION: seed_json = $cycle_dir/seeds_input.json"

  expand_dir="$cycle_dir/expansion"
  mkdir -p "$expand_dir"

  w2v_args=( --model "$MODEL_PATH" --seed_json "$cycle_dir/seeds_input.json" --exact_path "$expand_dir" )
  [[ -n "$TOPK" ]] && w2v_args+=( --topk "$TOPK" )
  [[ -n "$MIN_COS" ]] && w2v_args+=( --min_cos "$MIN_COS" )
  # shellcheck disable=SC2206
  extra_w2v_arr=($EXTRA_W2V)

  "$PY" "$W2V_EXPAND_PY" "${w2v_args[@]}" "${extra_w2v_arr[@]}"

  expansions_json="$(latest_file_in "$expand_dir"/topk_*.json)"
  if [[ -z "$expansions_json" || ! -f "$expansions_json" ]]; then
    echo "[error] No expansions JSON produced in $expand_dir"
    ls -la "$expand_dir" || true
    exit 1
  fi
  echo "[cycle $i] EXPANSION output: $expansions_json"

  # B) EVALUATION on that expansions JSON
  eval_root="$cycle_dir/eval"
  mkdir -p "$eval_root"

  echo "[cycle $i] EVALUATION: running LLM on expansions..."
  "$PY" "$LLM_EVAL_PY" \
    --expansions "$expansions_json" \
    --outdir "$eval_root" \
    --model "$LLM_MODEL_ID" \
    --url "$LLM_URL" \
    --temperature "$LLM_TEMP" \
    --tokens "$LLM_TOKENS" \
    --timeout "$LLM_TIMEOUT" \
    $LLM_EXTRA

  eval_subdir="$(newest_batch_eval_dir "$eval_root")"
  if [[ -z "$eval_subdir" || ! -d "$eval_subdir" ]]; then
    echo "[error] Could not find batch_eval_* directory under $eval_root"
    echo "Contents:"
    ls -la "$eval_root" || true
    exit 1
  fi

  summary_json="$eval_subdir/summary.json"
  require_file "$summary_json"

  new_seeds_json="$(json_get "$summary_json" "new_seeds_json")"
  if [[ -z "$new_seeds_json" || ! -f "$new_seeds_json" ]]; then
    echo "[error] Could not resolve new_seeds.json from $summary_json"
    echo "summary.json was:"
    cat "$summary_json" || true
    exit 1
  fi
  echo "[cycle $i] EVALUATION output new_seeds.json: $new_seeds_json"

  # NEXT cycle's seeds = this cycle's new_seeds.json
  seeds_current="$new_seeds_json"
done

echo ""
echo "Done. All cycles saved under: $OUT_BASE"
echo "Each cycle_N/ contains:"
echo "  - seeds_input.json         (seeds fed to expand_seed_terms.py)"
echo "  - expansion/topk_*.json    (Word2Vec expansions)"
echo "  - eval/batch_eval_*/       (LLM evaluation with new_seeds.json)"

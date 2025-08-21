#!/usr/bin/env bash
set -euo pipefail

# ---------- CONFIG ----------
THRESHOLDS=(5 2)
URLS=("http://localhost:11434/api/chat" "http://localhost:11435/api/chat" "http://localhost:11436/api/chat" "http://localhost:11437/api/chat")

OUT_BASE="testing/ngram_evals_test"

EVAL_SCRIPT="testing/evaluate_word2phrase_ngrams.py"
W2V_SCRIPT="word2vec_expansion/ngram_generation/train_word2vec_on_ngrams.py"
EXPAND_SCRIPT="word2vec_expansion/ngram_generation/expand_seed_terms.py"
LLM_SCRIPT="testing/LLM_semantic_addition.py"

# optional: pass extra flags to the eval script here (e.g., --users/--seed-terms)
EVAL_EXTRA_ARGS=()

# ---------- helpers ----------
ts() { date +"%Y-%m-%d %H:%M:%S"; }

# run_llm() {
#   local thr="$1"
#   local url="$2"
#   local dir="$OUT_BASE/$thr"
#   local expansions="$dir/topk_25_min_cos_0.4_cbow.json"
#   local log="$dir/llm_semantic_addition.log"

#   if [[ ! -f "$expansions" ]]; then
#     echo "[warn $(ts)] expansions not found for threshold=$thr at $expansions" | tee -a "$log"
#     return 1
#   fi

#   echo "[info $(ts)] LLM start: thr=$thr url=$url" | tee -a "$log"
#   python "$LLM_SCRIPT" \
#     --expansions "$expansions" \
#     --outdir "$dir" \
#     --url "$url" \
#     >> "$log" 2>&1
#   local rc=$?
#   if [[ $rc -eq 0 ]]; then
#     echo "[info $(ts)] LLM done: thr=$thr url=$url" | tee -a "$log"
#   else
#     echo "[error $(ts)] LLM failed: thr=$thr url=$url (rc=$rc)" | tee -a "$log"
#   fi
#   return $rc
# }

# trap 'echo; echo "[info $(ts)] interrupted"; exit 130' INT

# mkdir -p "$OUT_BASE"

# ---------- Stage 1: build ngrams, train word2vec, expand seed terms (sequential per threshold) ----------
for thr in "${THRESHOLDS[@]}"; do
  RUN_DIR="$OUT_BASE/$thr"
  mkdir -p "$RUN_DIR"

  echo "============================================================"
  echo "[info $(ts)] THRESHOLD = $thr"
  echo "============================================================"

  echo "[info $(ts)] (1/3) evaluating ngrams -> $RUN_DIR"
  python "$EVAL_SCRIPT" --out-folder "$OUT_BASE" --threshold "$thr" "${EVAL_EXTRA_ARGS[@]}" | tee "$RUN_DIR/eval.log"

  echo "[info $(ts)] (2/3) training word2vec on ngrams -> $RUN_DIR"
  python "$W2V_SCRIPT" \
    --ngram_phrasers_dir "$RUN_DIR" \
    --exact_path "$RUN_DIR" \
    | tee "$RUN_DIR/train_word2vec.log"

  echo "[info $(ts)] (3/3) expanding seed terms -> $RUN_DIR"
  python "$EXPAND_SCRIPT" \
    --model "$RUN_DIR/word2vec_cbow.model" \
    --exact_path "$RUN_DIR" \
    | tee "$RUN_DIR/expand_seed_terms.log"

  echo "[info $(ts)] threshold=$thr pipeline finished"
done

# ---------- Stage 2: LLM semantic addition in BATCHES of URL count ----------
echo
echo "============================================================"
echo "[info $(ts)] Starting LLM semantic addition in batches of ${#URLS[@]}"
echo "============================================================"

fail=0
pending=()
for thr in "${THRESHOLDS[@]}"; do
  pending+=("$thr")
  # when we have a full batch equal to number of URLs, launch & wait
  if ((${#pending[@]} == ${#URLS[@]})); then
    pids=()
    echo "[info $(ts)] Launching batch: ${pending[*]}"
    for i in "${!pending[@]}"; do
      thr_i="${pending[$i]}"
      url="${URLS[$i]}"
      run_llm "$thr_i" "$url" &
      pids+=("$!")
      echo "[info $(ts)] spawned LLM job for thr=$thr_i on $url (pid=${pids[-1]})"
    done
    # wait for the batch
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        fail=1
      fi
    done
    echo "[info $(ts)] Batch complete."
    pending=()
  fi
done

# leftover (final partial batch)
if ((${#pending[@]} > 0)); then
  pids=()
  echo "[info $(ts)] Launching final batch: ${pending[*]}"
  for i in "${!pending[@]}"; do
    thr_i="${pending[$i]}"
    url="${URLS[$i]}"
    run_llm "$thr_i" "$url" &
    pids+=("$!")
    echo "[info $(ts)] spawned LLM job for thr=$thr_i on $url (pid=${pids[-1]})"
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  echo "[info $(ts)] Final batch complete."
fi

if [[ $fail -eq 0 ]]; then
  echo "[info $(ts)] All LLM batches completed successfully."
else
  echo "[warn $(ts)] One or more LLM jobs failed. Check logs under $OUT_BASE/<thr>/llm_semantic_addition.log"
fi

echo "[info $(ts)] Pipeline complete. Outputs under: $OUT_BASE"
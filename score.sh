#!/bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-AI4Protein/ProSST-2048}"
RESIDUE_DIR="${RESIDUE_DIR:-zero_shot/example_data/residue_sequence}"
STRUCTURE_DIR="${STRUCTURE_DIR:-zero_shot/example_data/structure_sequence/2048}"
MUTANT_DIR="${MUTANT_DIR:-zero_shot/example_data/substitutions}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
SS_MASK_TOKEN_ID="${SS_MASK_TOKEN_ID:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

LOG_DIR="${1:-zero_shot/logs}"
mkdir -p "$LOG_DIR"

run_mode() {
    local mode="$1"
    local benchmark_log="$LOG_DIR/benchmark_${mode}.log"
    local eval_log="$LOG_DIR/average_spearman_${mode}.log"
    local substitutions_dir

    echo "Running mode=${mode}"

    python zero_shot/masked_marginal_benchmark.py \
        --model_path "$MODEL_PATH" \
        --residue_dir "$RESIDUE_DIR" \
        --structure_dir "$STRUCTURE_DIR" \
        --mutant_dir "$MUTANT_DIR" \
        --gpus "$GPU_IDS" \
        --mode "$mode" \
        --chunk_size "$CHUNK_SIZE" \
        --ss_mask_token_id "$SS_MASK_TOKEN_ID" \
        > "$benchmark_log" 2>&1

    substitutions_dir="$(dirname "$MUTANT_DIR")/substitutions_${mode}"
    python3 zero_shot/average_spearman.py --data_dir "$substitutions_dir" > "$eval_log" 2>&1

    echo "  benchmark log: $benchmark_log"
    echo "  eval log:      $eval_log"
}

run_mode "masked_seq"
run_mode "masked_both"

echo "All modes completed. Logs saved in $LOG_DIR"
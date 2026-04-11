#!/bin/bash

LOGFILE=$1
TMP_LOG="./benchmark_tmp_$$.log"   # $$ = unique PID

# Run benchmark silently → temp log
if python zero_shot/proteingym_benchmark.py \
    --model_path AI4Protein/ProSST-2048 \
    --residue_dir zero_shot/example_data/residue_sequence \
    --structure_dir zero_shot/example_data/structure_sequence/2048 \
    --mutant_dir zero_shot/example_data/substitutions \
    --pdb_dir ProteinGym_v1_AlphaFold2_PDB/proteingym_pdb \
    --plddt_center 70.0 \
    --plddt_sharpness 0.15 \
    > "$TMP_LOG" 2>&1
then
    # Run evaluation → final log
    python3 zero_shot/average_spearman.py > "$LOGFILE"

    # Delete temp log on success
    rm "$TMP_LOG"
    echo "Benchmark completed successfully."
else
    # Keep temp log on failure (no terminal output)
    echo "Benchmark failed. Check $TMP_LOG for details."
    exit 1
fi
#!/usr/bin/env bash
#SBATCH --job-name=ProSST_mms_train
#SBATCH --output=/home2/garimella.abhishek/ProSST/baseline_%j.out
#SBATCH --error=/home2/garimella.abhishek/ProSST/baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20

# ── adjust these to match your cluster's partition / account ──────────────
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=garimella.abhishek@research.iiit.ac.in

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR="/home2/garimella.abhishek/ProSST"
source /home2/garimella.abhishek/venv/bin/activate
cd "$PROJECT_DIR"

GPU_IDS=0,1,2,3 bash score.sh zero_shot/logs


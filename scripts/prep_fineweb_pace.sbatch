#!/bin/bash
# Prepare FineWeb-Edu 10B on PACE-ICE as a CPU job (no GPU), writing tokenized
# shards to scratch. Run this before the GPU sweep so 4x H200 time isn't spent
# on a CPU download/tokenize.
#
#   sbatch scripts/prep_fineweb_pace.sbatch
#
#SBATCH -J mla_prep
#SBATCH -N 1
#SBATCH -p ice-cpu
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -t 180
#SBATCH -o mla_prep_%j.out
set -e
cd "${SLURM_SUBMIT_DIR:-$HOME/Documents/mla_gpt}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$HOME/scratch/hf_cache"
OUT="$HOME/scratch/mla_data/fineweb_edu_10b"
mkdir -p "$OUT" "$HF_HOME"
echo "host=$(hostname) start=$(date -Is) out=$OUT"
echo -n "internet check: "; curl -sI -m 15 https://huggingface.co | head -1 || echo "WARN: no outbound internet on this node"
uv run python scripts/prepare_fineweb.py --out "$OUT" --num-proc 16
echo "done=$(date -Is)"; ls -la "$OUT"

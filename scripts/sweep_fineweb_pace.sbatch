#!/bin/bash
# Headline 124M FineWeb-Edu sweep on PACE-ICE as a Slurm JOB ARRAY: one attention
# variant per array task, each on a single H200. Independent 1-GPU jobs keep each
# job's GPU-minutes (1 x walltime) under the coc-ice QOS cap of 960 GPU-min/job;
# the monolithic 4-GPU form (4 x 480 = 1920) is rejected as QOSMaxGRESMinutesPerJob.
# The tasks schedule in parallel across 4 H200s, so wall-clock ~= one variant.
#
#   arr=$(sbatch --parsable scripts/sweep_fineweb_pace.sbatch)
#   sbatch --dependency=afterok:$arr scripts/eval_fineweb_pace.sbatch
#   ITERS=10000 BATCH=96 ACCUM=5 sbatch scripts/sweep_fineweb_pace.sbatch
#
# Data and checkpoints live in scratch (home quota is small); runs/ is symlinked
# to $HOME/scratch/mla_runs. tokens/iter = BATCH*ACCUM*1024 is held identical to
# the base config (491,520) so the LR schedule/dynamics match the local study.
#SBATCH -J mla_sweep
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 480
#SBATCH --array=0-3
#SBATCH -o runs/logs/fineweb_%A_%a.out
set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/Documents/mla_gpt}"
export PATH="$HOME/.local/bin:$PATH"

CONFIG=configs/gpt2_124m_base.yaml
DATA="${DATA:-$HOME/scratch/mla_data/fineweb_edu_10b}"
ITERS="${ITERS:-20000}"
BATCH="${BATCH:-96}"
ACCUM="${ACCUM:-5}"
VARIANTS=(mha gqa mqa mla)
attn="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
name="fineweb_${attn}"

# Keep big checkpoints off the small home quota.
mkdir -p "$HOME/scratch/mla_runs/logs"
if [ ! -L runs ]; then [ -e runs ] && mv runs "runs.bak.$$"; ln -s "$HOME/scratch/mla_runs" runs; fi
mkdir -p "runs/${name}"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} attn=${attn} iters=${ITERS} batch=${BATCH} accum=${ACCUM} tok/iter=$((BATCH*ACCUM*1024)) data=${DATA} ($(date -Is))"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# One GPU is allocated to this task; Slurm pins it via CUDA_VISIBLE_DEVICES.
TORCHINDUCTOR_CACHE_DIR="$HOME/scratch/inductor/${attn}" \
TRITON_CACHE_DIR="$HOME/scratch/triton/${attn}" \
  uv run python scripts/train.py --config "$CONFIG" --attn "$attn" --name "$name" \
    --set "train.data_dir=${DATA}" \
    --set "train.max_iters=${ITERS}" \
    --set "train.batch_size=${BATCH}" \
    --set "train.grad_accum=${ACCUM}" \
    > "runs/${name}/train.log" 2>&1
rc=$?
echo "=== ${attn} done rc=${rc} ($(date -Is)) ==="
exit $rc

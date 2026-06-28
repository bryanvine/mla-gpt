#!/bin/bash
# Headline 124M FineWeb-Edu sweep on PACE-ICE, 4x H200 (one variant per GPU, in
# parallel). The four attention variants are independent runs, so no DDP is
# needed -- each gets its own GPU and they finish in ~one-variant wall-clock.
#
#   sbatch scripts/sweep_fineweb_pace.sbatch
#   ITERS=10000 BATCH=96 ACCUM=5 sbatch scripts/sweep_fineweb_pace.sbatch
#
# Data and checkpoints live in scratch (home quota is small); runs/ is symlinked
# to $HOME/scratch/mla_runs. tokens/iter = BATCH*ACCUM*1024 is held identical to
# the base config (491,520) so the LR schedule/dynamics match the local study.
#SBATCH -J mla_sweep
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:4
#SBATCH -c 16
#SBATCH --mem-per-cpu=8G
#SBATCH -t 480
#SBATCH -o mla_sweep_%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/Documents/mla_gpt}"
export PATH="$HOME/.local/bin:$PATH"

CONFIG=configs/gpt2_124m_base.yaml
DATA="${DATA:-$HOME/scratch/mla_data/fineweb_edu_10b}"
ITERS="${ITERS:-20000}"
BATCH="${BATCH:-96}"
ACCUM="${ACCUM:-5}"
VARIANTS=(mha gqa mqa mla)

echo "host=$(hostname) iters=$ITERS batch=$BATCH accum=$ACCUM tok/iter=$((BATCH*ACCUM*1024)) data=$DATA"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Keep big checkpoints off the 30GB home quota.
mkdir -p "$HOME/scratch/mla_runs"
if [ ! -L runs ]; then [ -e runs ] && mv runs "runs.bak.$$"; ln -s "$HOME/scratch/mla_runs" runs; fi

pids=(); gpu=0
for attn in "${VARIANTS[@]}"; do
  name="fineweb_${attn}"
  mkdir -p "runs/${name}"
  echo "=== launch ${attn} on GPU ${gpu} -> runs/${name} ($(date -Is)) ==="
  CUDA_VISIBLE_DEVICES=${gpu} \
  TORCHINDUCTOR_CACHE_DIR="$HOME/scratch/inductor/${attn}" \
  TRITON_CACHE_DIR="$HOME/scratch/triton/${attn}" \
    uv run python scripts/train.py --config "$CONFIG" --attn "$attn" --name "$name" \
      --set "train.data_dir=${DATA}" \
      --set "train.max_iters=${ITERS}" \
      --set "train.batch_size=${BATCH}" \
      --set "train.grad_accum=${ACCUM}" \
      > "runs/${name}/train.log" 2>&1 &
  pids+=($!); gpu=$((gpu+1))
done

echo "launched pids: ${pids[*]}; waiting for all variants ..."
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "all training finished rc=$rc ($(date -Is))"

echo "=== eval all variants ==="
uv run python scripts/eval.py --glob 'runs/fineweb_*' --out runs/fineweb_eval || rc=1
echo "=== sweep complete rc=$rc -> runs/fineweb_eval ($(date -Is)) ==="
exit $rc

#!/bin/bash
# Evaluate all four FineWeb-Edu variants after the training array finishes. Runs
# as its own 1-GPU job (60 GPU-min, well under the 960 GPU-min/job QOS cap) and
# is gated on the array via afterok so every best.pt exists first.
#
#   sbatch --dependency=afterok:<arrayJobId> scripts/eval_fineweb_pace.sbatch
#
# --out is kept outside the runs/fineweb_* namespace so the --glob does not match
# the eval output dir itself.
#SBATCH -J mla_eval
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 60
#SBATCH -o runs/logs/fineweb_eval_%j.out
set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/Documents/mla_gpt}"
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

TORCHINDUCTOR_CACHE_DIR="$HOME/scratch/inductor/eval" \
TRITON_CACHE_DIR="$HOME/scratch/triton/eval" \
  uv run python scripts/eval.py --glob 'runs/fineweb_*' --out runs/eval_fineweb
rc=$?
echo "=== eval done rc=${rc} ($(date -Is)) ==="
ls -la runs/eval_fineweb 2>/dev/null
exit $rc

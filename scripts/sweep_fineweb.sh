#!/usr/bin/env bash
# Headline sweep: train all four attention variants on FineWeb-Edu 10B with one
# shared 124M config (only --attn changes), then evaluate quality side-by-side.
#
#   bash scripts/sweep_fineweb.sh
#
# Runs sequentially so the single 16GB GPU is never shared between runs. On a
# desktop card the display/DE reserves ~3GB, so batch_size is lowered from the
# config's 24 to 16 with grad_accum raised 20->30 -- tokens/iter is unchanged
# (491,520), so the LR schedule and training dynamics are identical. Each
# variant logs to runs/fineweb_<attn>/train.log.
set -u
cd "$(dirname "$0")/.."

CONFIG=configs/gpt2_124m_base.yaml
VARIANTS=(mha gqa mqa mla)

# Memory-safe batch/accum (same tokens/iter as the config); override via env.
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-30}"

for attn in "${VARIANTS[@]}"; do
  name="fineweb_${attn}"
  mkdir -p "runs/${name}"
  echo "=== training ${attn} -> runs/${name} ($(date -Is)) ==="
  uv run python scripts/train.py --config "$CONFIG" --attn "$attn" --name "$name" \
    --set "train.batch_size=${BATCH_SIZE}" \
    --set "train.grad_accum=${GRAD_ACCUM}" \
    2>&1 | tee "runs/${name}/train.log"
done

echo "=== evaluating all variants ($(date -Is)) ==="
uv run python scripts/eval.py --glob 'runs/fineweb_*' --out runs/fineweb_eval

echo "=== headline sweep complete -> runs/fineweb_eval ==="

#!/usr/bin/env bash
# Dev sweep: train all four attention variants on TinyStories with one shared
# config (only --attn changes), then evaluate quality side-by-side.
#
#   bash scripts/sweep_tinystories.sh
#
# Runs sequentially so a single 16GB GPU is never shared between runs. Each
# variant logs to runs/tinystories_<attn>/train.log.
set -u
cd "$(dirname "$0")/.."

CONFIG=configs/tinystories_base.yaml
VARIANTS=(mha mqa gqa mla)

for attn in "${VARIANTS[@]}"; do
  name="tinystories_${attn}"
  mkdir -p "runs/${name}"
  echo "=== training ${attn} -> runs/${name} ==="
  uv run python scripts/train.py --config "$CONFIG" --attn "$attn" --name "$name" \
    2>&1 | tee "runs/${name}/train.log"
done

echo "=== evaluating all variants ==="
uv run python scripts/eval.py --glob 'runs/tinystories_*' --out runs/tinystories_eval

echo "=== sweep complete -> runs/tinystories_eval ==="

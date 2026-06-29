#!/usr/bin/env bash
# Paper #3 confound control — isolate "gentler LR decay" from "more iters". LOCAL on the
# RTX 4080 (NOT the H200 Paper-#2 cluster).
#
# The 120k re-run let 10 cap-limited cells grok, but "2x budget" conflates two things:
# more iterations AND a gentler cosine decay (the 120k schedule sustains a higher LR at any
# given iter < 60k). This control trains for only 60k iters but under the 120k LR *shape*
# (lr_decay_iters=120000, max_iters=60000), so the iter budget matches the original 60k run
# and ONLY the LR schedule differs. Decision rule per flipped cell:
#   canonical-60k (steep) NO grok  +  lrctrl-60k (gentle) GROK   -> LR-SHAPE caused the flip
#   canonical-60k NO grok + lrctrl-60k NO grok + full-120k GROK  -> EXTRA-ITERS caused it
# By construction the gentle LR at iter<60k equals the full-120k LR, so (given determinism)
# cells whose 120k grok-iter < 60k should reproduce here at the same iter; cells whose
# 120k grok-iter >= 60k cannot grok under the 60k cap. The run validates that empirically.
#
#   bash scripts/recall_lrctrl.sh
# Then:
#   uv run python scripts/eval_recall.py --glob "runs/recall_mqar_*_lrctrl" --out runs/recall_eval_lrctrl
set -u
cd "$(dirname "$0")/.."

# attn p seed  -- the 10 cells that flipped False->True at 120k, with their 120k grok-iter.
# p96 first (the question); p64 last (positive controls, all groked <60k at 120k).
CELLS=(
  "gqa 96 1234"    # 120k grok @48000  (<60k -> expect LR-shape)
  "gqa 96 1337"    # @47500            (<60k -> LR-shape)
  "gqa 96 2024"    # @70500            (>=60k -> expect extra-iters)
  "gqa 96 31337"   # @86000            (>=60k -> extra-iters)
  "mha 96 1234"    # @53000            (<60k -> LR-shape)
  "mha 96 2024"    # @37000            (<60k -> LR-shape)
  "mqa 96 31337"   # @104500           (>=60k -> extra-iters)
  "mha 64 2024"    # @22000            (<60k -> LR-shape, positive control)
  "mha 64 31337"   # @23500            (<60k -> LR-shape, positive control)
  "mla 64 2024"    # @57000            (<60k -> LR-shape, borderline positive control)
)

NQ=${NQ:-16}
MAX_ITERS=${MAX_ITERS:-60000}          # iter budget matched to the ORIGINAL run
LR_DECAY=${LR_DECAY:-120000}           # but the LR *shape* of the 120k run (the only change)
EARLY_STOP=${EARLY_STOP:-0.5}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
COMPILE=${COMPILE:-true}
SCRATCH=${SCRATCH:-/tmp/recall_sweep}
DEST=runs

for cell in "${CELLS[@]}"; do
  read -r attn p seed <<<"$cell"
  name="recall_mqar_${attn}_p${p}_s${seed}_lrctrl"
  if [[ -f "${DEST}/${name}/summary.json" ]] && grep -q '"groked"' "${DEST}/${name}/summary.json"; then
    echo "skip ${name} (done)"; continue
  fi
  echo "=== ${name} ==="
  RUNS_DIR="$SCRATCH" uv run python scripts/train.py \
    --config configs/recall_base.yaml --attn "$attn" \
    --set "train.seed=${seed}" \
    --set "train.task_params.num_pairs=${p}" \
    --set "train.task_params.num_queries=${NQ}" \
    --name "$name" \
    --set "train.max_iters=${MAX_ITERS}" \
    --set "train.lr_decay_iters=${LR_DECAY}" \
    --set "train.early_stop_val=${EARLY_STOP}" \
    --set "train.eval_interval=${EVAL_INTERVAL}" \
    --set "train.compile=${COMPILE}"
  mkdir -p "${DEST}/${name}"
  for f in best.pt config.json summary.json metrics.csv; do
    [[ -f "${SCRATCH}/${name}/${f}" ]] && cp "${SCRATCH}/${name}/${f}" "${DEST}/${name}/"
  done
done

echo "=== lrctrl complete. Next: uv run python scripts/eval_recall.py --glob \"runs/recall_mqar_*_lrctrl\" --out runs/recall_eval_lrctrl ==="

#!/usr/bin/env bash
# Paper #3 follow-up — HIGHER-CAP re-run of the cap-limited MQAR cells, LOCAL on the
# RTX 4080 (NOT the H200 Paper-#2 cluster). Disentangles true recall *capacity* from
# *grok-speed*: the 60k sweep left 18 cells pinned at final_iter==60000 without grokking,
# so we cannot tell whether they would solve given more budget. Grok-iter is cap-invariant
# (verified by the _diag ablation: same crossing iter despite the cosine-schedule stretch),
# so doubling the cap to 120k is a clean extension of the optimization window — any cell
# that flips False->True was budget-limited, any cell that stays False is capacity-limited.
#
# Only the cap-limited cells are re-run (early-stopped cells are already cap-invariant).
# Output names get a _cap120k suffix so the canonical 60k baseline is never overwritten and
# the analysis can pair 60k-vs-120k. Cells are ordered highest-value first (p64 U-shape
# caveat, then MHA p96 cliff test, then the p96 floor on the weaker variants) so the most
# important results land first if interrupted.
#
#   bash scripts/recall_cap120k.sh
# Then re-eval:
#   uv run python scripts/eval_recall.py --glob "$SCRATCH/recall_*" --out runs/recall_eval
set -u
cd "$(dirname "$0")/.."

# attn p seed  -- the 18 cells with final_iter==60000 && groked==false @60k
CELLS=(
  # p64: directly resolve the headline U-shape (does extra budget make the extremes reliable?)
  "mha 64 2024"
  "mha 64 31337"
  "mla 64 2024"
  # p96: cliff budget-test on the strongest variant first
  "mha 96 1234"
  "mha 96 1337"
  "mha 96 2024"
  # p96 floor on the weaker variants (is the cliff capacity or just budget?)
  "gqa 96 1234"
  "gqa 96 1337"
  "gqa 96 2024"
  "gqa 96 31337"
  "mqa 96 1234"
  "mqa 96 1337"
  "mqa 96 2024"
  "mqa 96 31337"
  "mla 96 1234"
  "mla 96 1337"
  "mla 96 2024"
  "mla 96 31337"
)

NQ=${NQ:-16}
MAX_ITERS=${MAX_ITERS:-120000}        # 2x the original cap
EARLY_STOP=${EARLY_STOP:-0.5}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
COMPILE=${COMPILE:-true}
SCRATCH=${SCRATCH:-/tmp/recall_sweep}
DEST=runs

for cell in "${CELLS[@]}"; do
  read -r attn p seed <<<"$cell"
  name="recall_mqar_${attn}_p${p}_s${seed}_cap120k"
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
    --set "train.early_stop_val=${EARLY_STOP}" \
    --set "train.eval_interval=${EVAL_INTERVAL}" \
    --set "train.compile=${COMPILE}"
  mkdir -p "${DEST}/${name}"   # copy back only small final artifacts (skip ckpt.pt ~39MB)
  for f in best.pt config.json summary.json metrics.csv; do
    [[ -f "${SCRATCH}/${name}/${f}" ]] && cp "${SCRATCH}/${name}/${f}" "${DEST}/${name}/"
  done
done

echo "=== cap120k re-run complete. Next: uv run python scripts/eval_recall.py --glob \"$SCRATCH/recall_*\" --out runs/recall_eval ==="

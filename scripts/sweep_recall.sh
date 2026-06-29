#!/usr/bin/env bash
# Paper #3 (recall capacity) — LOCAL sweep on the RTX 4080 (NOT the H200 cluster;
# that machine runs the separate Paper #2 scale sweep). Trains all four attention
# variants on the synthetic capacity probes, changing only --attn and the task /
# difficulty. Synthetic models are tiny, so this is a single-GPU sequential sweep.
#
# Headline metric is ITERS-TO-GROK (recall sample-efficiency), not fixed-budget
# accuracy: these tasks sit on a high loss plateau then transition sharply, and the
# crossing iter is cap-invariant (verified: MHA p64 groks ~19.5k at both 20k and 50k
# caps, while LR at the transition differs 7.4e-4 vs 1.5e-4 — so it is an intrinsic
# optimization property, not a cosine-annealing artifact). early_stop_val ends a cell
# the moment it solves, so fast cells finish in <1 min and only the slow / never-
# grokking cells (the true capacity ceilings) run to the cap.
#
# Outputs go to non-synced scratch ($SCRATCH, default /tmp) to spare OneDrive the
# per-eval checkpoint churn; only the small final artifacts are copied back to runs/.
#
#   bash scripts/sweep_recall.sh                       # full grid, 1 seed
#   PAIRS="32 64" TOKENS="32" bash scripts/sweep_recall.sh   # pilot
#   SEEDS="1337 2024" bash scripts/sweep_recall.sh    # add error bars later
#
# Then evaluate (best.pt lives in scratch):
#   uv run python scripts/eval_recall.py --glob "$SCRATCH/recall_*" --out runs/recall_eval
set -u
cd "$(dirname "$0")/.."

VARIANTS=(${VARIANTS:-mha mqa gqa mla})
PAIRS=(${PAIRS:-16 32 64 96 128})     # MQAR difficulty: number of stored (k,v) pairs
TOKENS=(${TOKENS:-16 32 64 128 256 480})  # copy difficulty (tokens to reproduce); spans
                                          # trivial->near-block_size: <=64 groks at the first
                                          # eval, so the hard end is where copy can discriminate
SELCOPY=(${SELCOPY:-16 32 64 96 128})  # selective-copy difficulty (items to select among
                                       # blanks); x length = 4*num_tokens, so <=256 fits block_size
SEEDS=(${SEEDS:-1337})                # 1 seed first; add 2024 later for error bars
NQ=${NQ:-16}                          # MQAR queries per sequence (<= smallest PAIRS)
MAX_ITERS=${MAX_ITERS:-60000}         # generous cap; MLA p64 groks ~38k, slower cells ceiling out
EARLY_STOP=${EARLY_STOP:-0.5}         # stop at val<this (past the sharp knee -> well-solved best.pt)
EVAL_INTERVAL=${EVAL_INTERVAL:-500}   # grok-iter resolution
COMPILE=${COMPILE:-true}
SCRATCH=${SCRATCH:-/tmp/recall_sweep}
DEST=runs

run() {  # name -- train.py args...
  local name="$1"; shift
  # Design-aware skip: only skip a cell already done under THIS (early-stop) design,
  # marked by the "groked" field in summary.json. Stale fixed-budget summaries (from an
  # earlier pilot, no "groked") are re-run so the baseline isn't silently left out.
  # Keying on summary.json (not best.pt, which appears at the first eval) also lets an
  # interrupted cell resume from scratch/ckpt.pt.
  if [[ -f "${DEST}/${name}/summary.json" ]] && grep -q '"groked"' "${DEST}/${name}/summary.json"; then
    echo "skip ${name} (done)"; return
  fi
  echo "=== ${name} ==="
  RUNS_DIR="$SCRATCH" uv run python scripts/train.py "$@" --name "$name" \
    --set "train.max_iters=${MAX_ITERS}" \
    --set "train.early_stop_val=${EARLY_STOP}" \
    --set "train.eval_interval=${EVAL_INTERVAL}" \
    --set "train.compile=${COMPILE}"
  mkdir -p "${DEST}/${name}"   # copy back only small final artifacts (skip ckpt.pt ~39MB)
  for f in best.pt config.json summary.json metrics.csv; do
    [[ -f "${SCRATCH}/${name}/${f}" ]] && cp "${SCRATCH}/${name}/${f}" "${DEST}/${name}/"
  done
}

for seed in "${SEEDS[@]}"; do
  for attn in "${VARIANTS[@]}"; do
    for p in "${PAIRS[@]}"; do
      run "recall_mqar_${attn}_p${p}_s${seed}" \
        --config configs/recall_base.yaml --attn "$attn" \
        --set "train.seed=${seed}" \
        --set "train.task_params.num_pairs=${p}" \
        --set "train.task_params.num_queries=${NQ}"
    done
    for n in "${TOKENS[@]}"; do
      run "recall_copy_${attn}_n${n}_s${seed}" \
        --config configs/recall_copy.yaml --attn "$attn" \
        --set "train.seed=${seed}" \
        --set "train.task_params.num_tokens=${n}"
    done
    for n in "${SELCOPY[@]}"; do
      run "recall_selcopy_${attn}_n${n}_s${seed}" \
        --config configs/recall_selcopy.yaml --attn "$attn" \
        --set "train.seed=${seed}" \
        --set "train.task_params.num_tokens=${n}"
    done
  done
done

echo "=== sweep complete. Next: uv run python scripts/eval_recall.py --glob \"$SCRATCH/recall_*\" --out runs/recall_eval ==="

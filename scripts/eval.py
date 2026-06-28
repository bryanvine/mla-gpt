"""Evaluate trained runs: quality metrics, comparison tables, and figures.

    uv run python scripts/eval.py --glob 'runs/tinystories/*' --out runs/tinystories/eval

For each run directory (one per attention variant) computes held-out
perplexity / bits-per-byte from best.pt, then emits:
  * eval.json            -- raw per-run metrics (+ generation samples),
  * comparison.md        -- a side-by-side quality table,
  * val_loss_curves.png  -- val loss vs training iter, all variants,
  * perplexity_bars.png  -- final perplexity per variant,
  * quality_vs_kv.png    -- the quality/efficiency trade-off (ppl vs KV cache).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import glob as globmod
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mla_gpt import eval as ev  # noqa: E402

COLORS = {"mha": "#d62728", "gqa": "#1f77b4", "mqa": "#9467bd", "mla": "#2ca02c"}
ORDER = {"mha": 0, "gqa": 1, "mqa": 2, "mla": 3}
DEFAULT_PROMPTS = ["Once upon a time", "The little robot", "One day, a girl named Lily"]


def discover_runs(args) -> list[Path]:
    paths: list[Path] = []
    for r in args.runs or []:
        paths.append(Path(r))
    for pat in args.glob or []:
        paths.extend(Path(p) for p in sorted(globmod.glob(pat)))
    runs = [p for p in paths if (p / "best.pt").exists() and (p / "config.json").exists()]
    seen, uniq = set(), []
    for p in runs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return sorted(uniq, key=lambda p: ORDER.get(_attn_of(p), 99))


def _attn_of(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "config.json").read_text())["model"]["attn_type"]
    except Exception:  # noqa: BLE001
        return run_dir.name


def read_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    iters, vals = [], []
    mp = run_dir / "metrics.csv"
    if not mp.exists():
        return iters, vals
    with mp.open() as f:
        for row in csv.DictReader(f):
            if row.get("val_loss"):
                iters.append(int(row["iter"]))
                vals.append(float(row["val_loss"]))
    return iters, vals


def plot_curves(runs, results, out_dir):
    plt.figure(figsize=(6, 4))
    for r in results:
        rd = Path(r["run"])
        it, vl = read_curve(rd)
        if it:
            plt.plot(it, vl, "-", label=r["attn_type"].upper(), color=COLORS.get(r["attn_type"]))
    plt.xlabel("training iteration")
    plt.ylabel("validation loss (nats/token)")
    plt.title("Validation loss vs training step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss_curves.png", dpi=150)
    plt.close()


def plot_perplexity(results, out_dir):
    plt.figure(figsize=(6, 4))
    names = [r["attn_type"].upper() for r in results]
    ppls = [r["perplexity"] for r in results]
    plt.bar(names, ppls, color=[COLORS.get(r["attn_type"]) for r in results])
    for i, p in enumerate(ppls):
        plt.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("validation perplexity")
    plt.title("Final perplexity by attention variant")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "perplexity_bars.png", dpi=150)
    plt.close()


def plot_quality_vs_kv(results, out_dir):
    pts = [(r["kv_bytes_per_token"], r["perplexity"], r["attn_type"]) for r in results
           if r.get("kv_bytes_per_token")]
    if not pts:
        return
    plt.figure(figsize=(6, 4))
    for kv, ppl, a in pts:
        plt.scatter(kv, ppl, s=80, color=COLORS.get(a), zorder=3)
        plt.annotate(a.upper(), (kv, ppl), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xlabel("KV cache (bytes / token, all layers)")
    plt.ylabel("validation perplexity")
    plt.title("Quality vs KV-cache footprint")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "quality_vs_kv.png", dpi=150)
    plt.close()


def write_comparison(results, out_dir):
    lines = [
        "| Variant | Params | KV B/tok | Val loss | Perplexity | Bits/byte |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['attn_type'].upper()} | {r['params']/1e6:.1f}M | {r['kv_bytes_per_token']:,} | "
            f"{r['val_loss']:.4f} | {r['perplexity']:.3f} | {r['bits_per_byte']:.4f} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_samples(results, out_dir):
    if not any("samples" in r for r in results):
        return
    lines = ["# Generation samples\n"]
    for r in results:
        lines.append(f"## {r['attn_type'].upper()}\n")
        for s in r.get("samples", []):
            lines.append(f"**Prompt:** {s['prompt']}\n")
            lines.append(f"> {s['completion']}\n")
    (out_dir / "samples.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="explicit run directories")
    ap.add_argument("--glob", nargs="*", help="glob pattern(s) for run directories")
    ap.add_argument("--data-dir", default=None, help="override data dir (else from each run config)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(ev._DTYPES))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()

    runs = discover_runs(args)
    if not runs:
        raise SystemExit("no runs found (need best.pt + config.json). Use --runs or --glob.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = None if args.no_samples else DEFAULT_PROMPTS

    results = []
    for rd in runs:
        print(f"\n=== eval {rd} ({_attn_of(rd).upper()}) ===")
        r = ev.evaluate_run(rd, args.data_dir, args.device, args.dtype,
                            args.batch_size, args.split, prompts, args.max_new_tokens)
        print(f"  val_loss {r['val_loss']:.4f} | ppl {r['perplexity']:.3f} | bpb {r['bits_per_byte']:.4f}")
        results.append(r)

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    write_comparison(results, out_dir)
    write_samples(results, out_dir)
    plot_curves(runs, results, out_dir)
    plot_perplexity(results, out_dir)
    plot_quality_vs_kv(results, out_dir)
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()

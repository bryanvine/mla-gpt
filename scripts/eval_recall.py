"""Evaluate Paper #3 (recall capacity): synthetic accuracy + the natural-text bridge.

    uv run python scripts/eval_recall.py --glob 'runs/recall_*' --out runs/recall_eval

From the synthetic runs it computes, per attention variant:
  * the recall cliff      -- accuracy vs trained difficulty (one curve per variant),
  * the Pareto reframe     -- accuracy vs KV-cache bytes at the hardest difficulty,
  * extrapolation          -- train-short / test-long accuracy of one trained model.
From the Paper #1 TinyStories checkpoints it computes:
  * position-wise loss     -- CE bucketed by token position (the decomposed metric
                              that the headline perplexity averaged away).
Outputs recall.json, recall_comparison.md, and the figures above.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mla_gpt import recall_eval as re  # noqa: E402
from mla_gpt import eval as ev  # noqa: E402

COLORS = {"mha": "#d62728", "gqa": "#1f77b4", "mqa": "#9467bd", "mla": "#2ca02c"}
ORDER = {"mha": 0, "gqa": 1, "mqa": 2, "mla": 3}
TASK_TITLE = {"mqar": "Associative recall (MQAR)", "copy": "Plain copy (control)",
              "selective_copy": "Selective copy"}
AXIS_LABEL = {"mqar": "stored key-value pairs", "copy": "tokens to copy",
              "selective_copy": "tokens to select"}


def discover(patterns) -> list[Path]:
    paths = []
    for pat in patterns:
        paths.extend(Path(p) for p in sorted(globmod.glob(pat)))
    runs = [p for p in paths if (p / "best.pt").exists() and (p / "config.json").exists()]
    seen, uniq = set(), []
    for p in runs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _by_variant(rows, key):
    """{attn: sorted [(x, mean, std)]} aggregating repeats (seeds) at equal x."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["attn_type"]][r[key]].append(r["accuracy"])
    out = {}
    for attn, xs in acc.items():
        pts = []
        for x in sorted(xs):
            vals = xs[x]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            pts.append((x, mean, std))
        out[attn] = pts
    return out


def _variants_sorted(rows):
    return sorted({r["attn_type"] for r in rows}, key=lambda a: ORDER.get(a, 99))


def plot_cliff(rows, task, out_dir):
    series = _by_variant(rows, "trained_difficulty")
    plt.figure(figsize=(6, 4))
    for attn in sorted(series, key=lambda a: ORDER.get(a, 99)):
        xs = [p[0] for p in series[attn]]
        ys = [p[1] for p in series[attn]]
        es = [p[2] for p in series[attn]]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                     label=attn.upper(), color=COLORS.get(attn))
    plt.xlabel(f"trained difficulty ({AXIS_LABEL[task]})")
    plt.ylabel("recall accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(f"{TASK_TITLE[task]}: capacity cliff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"recall_{task}_cliff.png", dpi=150)
    plt.close()


def _grok_by_variant(rows):
    """{attn: sorted [(difficulty, mean_grok_iters|None, n_grok, n_total, cap)]}.

    iters-to-grok is None for cells that never crossed the threshold within the cap
    (the true capacity ceiling); those are aggregated separately from the timing mean.
    """
    its = defaultdict(lambda: defaultdict(list))
    caps = defaultdict(lambda: defaultdict(list))
    for r in rows:
        its[r["attn_type"]][r["trained_difficulty"]].append(r.get("iters_to_grok"))
        caps[r["attn_type"]][r["trained_difficulty"]].append(r.get("final_iter") or 0)
    out = {}
    for attn, xs in its.items():
        pts = []
        for x in sorted(xs):
            grok = [v for v in xs[x] if v is not None]
            mean = sum(grok) / len(grok) if grok else None
            pts.append((x, mean, len(grok), len(xs[x]), max(caps[attn][x])))
        out[attn] = pts
    return out


def _has_grok(rows):
    return any(r.get("iters_to_grok") is not None for r in rows)


def plot_grok(rows, task, out_dir):
    """Sample-efficiency cliff: training iters until each cell solves the task.

    Solid line = groked cells (lower is more sample-efficient); an ``x`` at the cap
    marks cells that never groked within budget (the capacity ceiling of that variant).
    """
    if not _has_grok(rows):
        return
    series = _grok_by_variant(rows)
    cap = max((p[4] for pts in series.values() for p in pts), default=0)
    xmin = min((p[0] for pts in series.values() for p in pts), default=0)
    plt.figure(figsize=(6, 4))
    for attn in sorted(series, key=lambda a: ORDER.get(a, 99)):
        pts = series[attn]
        gx = [p[0] for p in pts if p[1] is not None]
        gy = [p[1] for p in pts if p[1] is not None]
        if gx:
            plt.plot(gx, gy, marker="o", label=attn.upper(), color=COLORS.get(attn))
        dnf = [p[0] for p in pts if p[1] is None]
        if dnf:
            plt.scatter(dnf, [cap] * len(dnf), marker="x", s=70,
                        color=COLORS.get(attn), zorder=3)
    if cap:
        plt.axhline(cap, ls=":", color="gray", alpha=0.6)
        plt.text(xmin, cap, " cap — did not grok (✗)", color="gray", fontsize=8, va="bottom")
    plt.xlabel(f"trained difficulty ({AXIS_LABEL[task]})")
    plt.ylabel("iters to grok (val < threshold)")
    plt.yscale("log")
    plt.title(f"{TASK_TITLE[task]}: recall sample-efficiency")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"recall_{task}_grok.png", dpi=150)
    plt.close()


def plot_pareto(rows, task, out_dir):
    hardest = max(r["trained_difficulty"] for r in rows)
    pts = {r["attn_type"]: r for r in rows if r["trained_difficulty"] == hardest}
    plt.figure(figsize=(6, 4))
    for attn in sorted(pts, key=lambda a: ORDER.get(a, 99)):
        r = pts[attn]
        plt.scatter(r["kv_bytes_per_token"], r["accuracy"], s=90,
                    color=COLORS.get(attn), zorder=3)
        plt.annotate(attn.upper(), (r["kv_bytes_per_token"], r["accuracy"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xlabel("KV cache (bytes / token, all layers)")
    plt.ylabel(f"recall accuracy @ {hardest} {AXIS_LABEL[task]}")
    plt.title(f"{TASK_TITLE[task]}: capability vs KV footprint")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"recall_{task}_pareto.png", dpi=150)
    plt.close()


def plot_extrapolation(rows, task, out_dir, train_diff=None):
    trained_vals = sorted({r["trained_difficulty"] for r in rows})
    if not trained_vals:
        return
    td = train_diff if train_diff in trained_vals else trained_vals[0]
    chosen = [r for r in rows if r["trained_difficulty"] == td]
    plt.figure(figsize=(6, 4))
    for r in sorted(chosen, key=lambda r: ORDER.get(r["attn_type"], 99)):
        xs = [s["difficulty"] for s in r["sweep"]]
        ys = [s["accuracy"] for s in r["sweep"]]
        plt.plot(xs, ys, marker=".", label=r["attn_type"].upper(),
                 color=COLORS.get(r["attn_type"]))
    plt.axvline(td, ls="--", color="gray", alpha=0.7)
    plt.text(td, 0.02, f" trained @ {td}", color="gray", fontsize=8, rotation=90, va="bottom")
    plt.xlabel(f"evaluation difficulty ({AXIS_LABEL[task]})")
    plt.ylabel("recall accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(f"{TASK_TITLE[task]}: train-short / test-long")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"recall_{task}_extrapolation.png", dpi=150)
    plt.close()


def plot_position_loss(ppl, out_dir):
    if not ppl:
        return
    plt.figure(figsize=(6, 4))
    for r in sorted(ppl, key=lambda r: ORDER.get(r["attn_type"], 99)):
        b = r["position_loss"]["buckets"]
        xs = [d["center"] for d in b]
        ys = [d["loss"] for d in b]
        plt.plot(xs, ys, marker=".", label=r["attn_type"].upper(),
                 color=COLORS.get(r["attn_type"]))
    plt.xlabel("token position in context")
    plt.ylabel("cross-entropy (nats)")
    plt.title("Natural-text loss by position (TinyStories)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_vs_position.png", dpi=150)
    plt.close()


def write_comparison(results, out_dir):
    lines = []
    for task in sorted({r["task"] for r in results}):
        rows = [r for r in results if r["task"] == task]
        diffs = sorted({r["trained_difficulty"] for r in rows})
        variants = _variants_sorted(rows)
        lines.append(f"\n### {TASK_TITLE[task]} — accuracy by {AXIS_LABEL[task]}\n")
        lines.append("| Variant | KV B/tok | " + " | ".join(str(d) for d in diffs) + " |")
        lines.append("|---|---|" + "|".join("---" for _ in diffs) + "|")
        series = _by_variant(rows, "trained_difficulty")
        kv = {r["attn_type"]: r["kv_bytes_per_token"] for r in rows}
        for attn in variants:
            cells = {x: m for x, m, _ in series.get(attn, [])}
            row = " | ".join(f"{cells[d]:.2f}" if d in cells else "—" for d in diffs)
            lines.append(f"| {attn.upper()} | {kv.get(attn, 0):,} | {row} |")

        if _has_grok(rows):  # sample-efficiency: iters to solve (✗ = never within cap)
            gser = _grok_by_variant(rows)
            lines.append(f"\n### {TASK_TITLE[task]} — iters-to-grok by {AXIS_LABEL[task]}\n")
            lines.append("| Variant | KV B/tok | " + " | ".join(str(d) for d in diffs) + " |")
            lines.append("|---|---|" + "|".join("---" for _ in diffs) + "|")
            for attn in variants:
                cells = {x: m for x, m, _, _, _ in gser.get(attn, [])}
                row = " | ".join(
                    ("—" if d not in cells else "✗" if cells[d] is None else f"{cells[d]:.0f}")
                    for d in diffs)
                lines.append(f"| {attn.upper()} | {kv.get(attn, 0):,} | {row} |")
    (out_dir / "recall_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="*", default=["runs/recall_*"])
    ap.add_argument("--ppl-glob", nargs="*", default=["runs/tinystories_*"],
                    help="TinyStories runs for the position-wise loss bridge")
    ap.add_argument("--data-dir", default=None, help="override data dir for position-wise loss")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(ev._DTYPES))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-batches", type=int, default=16)
    ap.add_argument("--extrap-train", type=int, default=None,
                    help="trained difficulty whose models to use for the extrapolation panel")
    ap.add_argument("--out", default="runs/recall_eval")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover(args.glob)
    results = []
    for rd in runs:
        print(f"=== recall eval {rd} ===")
        r = re.evaluate_recall_run(rd, args.device, args.dtype, args.batch_size, args.num_batches)
        print(f"  {r['attn_type'].upper()} {r['task']} @ {r['trained_difficulty']}: acc {r['accuracy']:.3f}")
        results.append(r)

    # Natural-text bridge: position-wise loss on the Paper #1 checkpoints.
    ppl = []
    for rd in discover(args.ppl_glob):
        cfg = json.loads((rd / "config.json").read_text())
        data_dir = args.data_dir or cfg["train"]["data_dir"]
        if not (Path(data_dir) / "val.bin").exists():
            print(f"  [skip position loss] {rd}: {data_dir}/val.bin missing")
            continue
        print(f"=== position-wise loss {rd} ===")
        model, _ = ev.load_checkpoint(rd / "best.pt", args.device)
        pl = re.position_wise_loss(model, data_dir, cfg["model"]["block_size"],
                                   args.device, ev._DTYPES[args.dtype])
        ppl.append({"attn_type": cfg["model"]["attn_type"], "run": str(rd), "position_loss": pl})

    (out_dir / "recall.json").write_text(json.dumps({"recall": results, "position_loss": ppl}, indent=2))
    write_comparison(results, out_dir)
    for task in sorted({r["task"] for r in results}):
        rows = [r for r in results if r["task"] == task]
        plot_cliff(rows, task, out_dir)
        plot_grok(rows, task, out_dir)
        plot_pareto(rows, task, out_dir)
        plot_extrapolation(rows, task, out_dir, args.extrap_train)
    plot_position_loss(ppl, out_dir)
    print(f"\nresults -> {out_dir}")


if __name__ == "__main__":
    main()

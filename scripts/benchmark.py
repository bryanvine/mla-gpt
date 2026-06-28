"""Run the efficiency benchmark across all attention variants and plot results.

    uv run python scripts/benchmark.py [--device cuda] [--out runs/benchmark]

Produces results.json plus figures (KV-cache scaling, decode memory/throughput,
prefill & train throughput) for the paper.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from pathlib import Path

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mla_gpt import benchmark as bm  # noqa: E402

VARIANTS = ["mha", "gqa", "mqa", "mla"]
COLORS = {"mha": "#d62728", "gqa": "#1f77b4", "mqa": "#9467bd", "mla": "#2ca02c"}


def load_base(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())["model"]
    cfg.pop("attn_type", None)
    cfg.pop("block_size", None)
    return cfg


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        import torch
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        return {"error": str(e)[:120]}


def run(args) -> dict:
    import torch

    base = load_base(args.config)
    dtype = bm._DTYPES[args.dtype]
    contexts = [int(c) for c in args.contexts.split(",")]
    analytic_ctx = [1024, 4096, 16384, 65536, 131072, 262144]
    bench_block = max(contexts) + 256

    results = {"meta": {"device": args.device, "dtype": args.dtype, "base": base,
                        "contexts": contexts, "analytic_ctx": analytic_ctx,
                        "batch_decode": args.batch_decode, "batch_prefill": args.batch_prefill,
                        "seq_len": args.seq_len}, "variants": {}}

    for v in VARIANTS:
        print(f"\n=== {v.upper()} ===")
        model = bm.build_model(v, bench_block, args.device, base)
        torch.cuda.empty_cache()
        r: dict = {
            "params": model.num_params(),
            "kv_bytes_per_token": model.kv_bytes_per_token(),
            "analytical_kv_gb": {str(c): bm.analytical_kv_gb(model, c, args.batch_decode) for c in analytic_ctx},
        }
        r["prefill"] = safe(bm.measure_prefill, model, args.batch_prefill, args.seq_len, args.device, dtype)
        r["train_step"] = safe(bm.measure_train_step, model, args.batch_prefill, args.seq_len, args.device, dtype)
        r["decode"] = [safe(bm.measure_decode, model, args.batch_decode, c, args.device, dtype) for c in contexts]
        if args.max_context:
            r["max_decode_context"] = safe(bm.max_decode_context, model, args.batch_decode, args.device, dtype, bench_block - 256)
        results["variants"][v] = r
        for d in r["decode"]:
            if "error" not in d:
                print(f"  ctx {d['context_len']:>6d}: {d['decode_ms_per_step']:.2f} ms/step  "
                      f"{d['decode_tok_per_s']:.0f} tok/s  {d['peak_mem_mb']:.0f} MB")
        del model
        torch.cuda.empty_cache()

    return results


def plot(results: dict, out_dir: Path) -> None:
    variants = list(results["variants"])
    actx = results["meta"]["analytic_ctx"]
    bd = results["meta"]["batch_decode"]

    # 1. analytical KV-cache vs context (log-log)
    plt.figure(figsize=(6, 4))
    for v in variants:
        ys = [results["variants"][v]["analytical_kv_gb"][str(c)] for c in actx]
        plt.loglog(actx, ys, "o-", label=v.upper(), color=COLORS[v])
    plt.xlabel("context length (tokens)")
    plt.ylabel(f"KV cache (GB, batch={bd})")
    plt.title("KV-cache footprint vs context")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "kv_cache_vs_context.png", dpi=150)
    plt.close()

    # 2. decode peak memory + 3. decode throughput vs context
    for key, ylabel, fname, title in [
        ("peak_mem_mb", "peak memory (MB)", "decode_mem_vs_context.png", "Decode-step peak memory vs context"),
        ("decode_tok_per_s", "tokens / s", "decode_throughput_vs_context.png", "Decode throughput vs context"),
    ]:
        plt.figure(figsize=(6, 4))
        for v in variants:
            pts = [(d["context_len"], d[key]) for d in results["variants"][v]["decode"] if "error" not in d]
            if pts:
                xs, ys = zip(*pts)
                plt.plot(xs, ys, "o-", label=v.upper(), color=COLORS[v])
        plt.xlabel("context length (tokens)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    # 4. prefill & train throughput bars
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, sub, title in [
        (axes[0], "prefill_tok_per_s", "prefill", "Prefill throughput"),
        (axes[1], "train_tok_per_s", "train_step", "Training-step throughput"),
    ]:
        vals = [results["variants"][v][sub].get(key, 0) or 0 for v in variants]
        ax.bar([v.upper() for v in variants], vals, color=[COLORS[v] for v in variants])
        ax.set_ylabel("tokens / s")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_bars.png", dpi=150)
    plt.close()
    print(f"figures -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpt2_124m_base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(bm._DTYPES))
    ap.add_argument("--contexts", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--batch-decode", type=int, default=8)
    ap.add_argument("--batch-prefill", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-context", action="store_true")
    ap.add_argument("--out", default="runs/benchmark")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run(args)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    plot(results, out_dir)
    print(f"results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

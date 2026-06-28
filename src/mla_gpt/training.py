"""Single-GPU training loop shared by every attention variant.

bf16 autocast, gradient accumulation, cosine LR with warmup, grad clipping,
periodic evaluation, checkpointing, and CSV metric logging. The same code path
trains MHA/MQA/GQA/MLA -- only ``GPTConfig.attn_type`` differs -- so training is
a controlled variable.
"""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from .config import GPTConfig
from .data import make_get_batch, dataset_num_tokens
from .model import GPT


@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories"
    out_dir: str = "runs/default"
    # optimization
    batch_size: int = 32
    grad_accum: int = 4
    max_iters: int = 5000
    warmup_iters: int = 200
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # eval / logging
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    always_save_checkpoint: bool = False
    # system
    seed: int = 1337
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True

    @property
    def tokens_per_iter(self) -> int:
        return self.batch_size * self.grad_accum  # * block_size, filled at runtime


def _lr_at(it: int, c: TrainConfig) -> float:
    if it < c.warmup_iters:
        return c.lr * (it + 1) / (c.warmup_iters + 1)
    if it > c.max_iters:
        return c.min_lr
    ratio = (it - c.warmup_iters) / max(1, c.max_iters - c.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return c.min_lr + coeff * (c.lr - c.min_lr)


@torch.no_grad()
def estimate_loss(model, get_batch, tc: TrainConfig, ctx) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            x, y = get_batch(split, tc.batch_size)
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()
            logits = loss = None  # release full-vocab logits before the next forward
        out[split] = losses.mean().item()
    model.train()
    return out


def run_training(model_cfg: GPTConfig, tc: TrainConfig) -> dict:
    torch.manual_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(tc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_type = "cuda" if str(tc.device).startswith("cuda") else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[tc.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=ptdtype)

    block_size = model_cfg.block_size
    get_batch = make_get_batch(tc.data_dir, block_size, tc.device)
    tokens_per_iter = tc.batch_size * tc.grad_accum * block_size

    model = GPT(model_cfg).to(tc.device)
    raw_model = model
    if tc.compile and device_type == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:  # pragma: no cover
            print(f"[warn] torch.compile failed ({e}); continuing eager")
            model = raw_model

    optimizer = raw_model.configure_optimizers(tc.weight_decay, tc.lr, (tc.beta1, tc.beta2), device_type)
    scaler = torch.amp.GradScaler(enabled=(tc.dtype == "float16"))

    # persist run config
    (out_dir / "config.json").write_text(
        json.dumps({"model": model_cfg.to_dict(), "train": asdict(tc),
                    "params": raw_model.num_params(),
                    "kv_bytes_per_token": raw_model.kv_bytes_per_token(),
                    "tokens_per_iter": tokens_per_iter}, indent=2)
    )
    metrics_path = out_dir / "metrics.csv"
    metrics_file = metrics_path.open("w", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow(["iter", "time_s", "lr", "train_loss", "val_loss", "tokens", "tok_per_s"])

    print(f"[{tc.out_dir}] {model_cfg.attn_type.upper()} | params={raw_model.num_params():,} "
          f"| kv={raw_model.kv_bytes_per_token():,} B/tok | tok/iter={tokens_per_iter:,}")

    best_val = float("inf")
    t0 = time.time()
    x, y = get_batch("train", tc.batch_size)
    running = None
    for it in range(tc.max_iters + 1):
        lr = _lr_at(it, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % tc.eval_interval == 0:
            losses = estimate_loss(model, get_batch, tc, ctx)
            dt = time.time() - t0
            tps = (it * tokens_per_iter) / dt if it > 0 else 0.0
            writer.writerow([it, f"{dt:.1f}", f"{lr:.2e}", f"{losses['train']:.4f}",
                             f"{losses['val']:.4f}", it * tokens_per_iter, f"{tps:.0f}"])
            metrics_file.flush()
            print(f"iter {it:>6d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {tps/1e3:.1f}k tok/s")
            if losses["val"] < best_val or tc.always_save_checkpoint:
                best_val = min(best_val, losses["val"])
                torch.save({"model": raw_model.state_dict(), "model_cfg": model_cfg.to_dict(),
                            "iter": it, "val_loss": losses["val"]}, out_dir / "best.pt")

        if it == tc.max_iters:
            break

        for micro in range(tc.grad_accum):
            with ctx:
                _, loss = model(x, y)
                loss = loss / tc.grad_accum
            x, y = get_batch("train", tc.batch_size)  # prefetch next while GPU works
            scaler.scale(loss).backward()
        if tc.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running = loss.item() * tc.grad_accum if running is None else \
            0.9 * running + 0.1 * loss.item() * tc.grad_accum
        if it % tc.log_interval == 0:
            print(f"iter {it:>6d} | loss {running:.4f} | lr {lr:.2e}", flush=True)

    metrics_file.close()
    total_time = time.time() - t0
    summary = {"attn_type": model_cfg.attn_type, "best_val_loss": best_val,
               "params": raw_model.num_params(), "kv_bytes_per_token": raw_model.kv_bytes_per_token(),
               "total_time_s": total_time, "final_iter": tc.max_iters,
               "tokens_seen": tc.max_iters * tokens_per_iter}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best val {best_val:.4f} in {total_time/60:.1f} min -> {out_dir}")
    return summary

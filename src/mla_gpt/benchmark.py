"""Inference / training efficiency measurements per attention variant.

Covers the "efficiency" half of the study:
  * analytical KV-cache size vs context length (exact),
  * empirical decode-step latency and peak memory vs context (MLA uses its
    weight-absorbed path -- a fair decode comparison),
  * prefill throughput, and
  * training-step time + peak memory.
"""

from __future__ import annotations

import time

import torch

from .attention import MultiHeadLatentAttention
from .config import GPTConfig
from .model import GPT

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def is_mla(model: GPT) -> bool:
    return isinstance(model.blocks[0].attn, MultiHeadLatentAttention)


def build_model(attn_type: str, block_size: int, device: str, base: dict | None = None) -> GPT:
    kw = dict(base or {})
    kw.update(attn_type=attn_type, block_size=block_size)
    return GPT(GPTConfig(**kw)).to(device).eval()


def analytical_kv_gb(model: GPT, context_len: int, batch: int = 1, dtype_size: int = 2) -> float:
    """Total KV-cache footprint (GB) for a full context across all layers."""
    return model.kv_bytes_per_token(dtype_size) * context_len * batch / 1e9


def _autocast(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()


@torch.no_grad()
def measure_prefill(model, batch, seq_len, device, dtype, warmup=3, iters=8) -> dict:
    idx = torch.randint(0, model.config.vocab_size, (batch, seq_len), device=device)
    ctx = _autocast(device, dtype)
    for _ in range(warmup):
        with ctx:
            model(idx)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        with ctx:
            model(idx)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {
        "prefill_ms": dt * 1e3,
        "prefill_tok_per_s": batch * seq_len / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


@torch.no_grad()
def measure_decode(model, batch, context_len, device, dtype, steps=32, warmup=8) -> dict:
    """Per-step decode latency + peak memory given a pre-filled context of context_len."""
    absorbed = is_mla(model)
    cos, sin = model.cos, model.sin
    tok = torch.randint(0, model.config.vocab_size, (batch, 1), device=device)
    ctx = _autocast(device, dtype)

    def fresh_caches():
        return [blk.attn.init_cache(batch, context_len, device, dtype) for blk in model.blocks]

    caches = fresh_caches()
    for _ in range(warmup):
        x = model.tok_emb(tok)
        with ctx:
            for i, blk in enumerate(model.blocks):
                x, _ = blk(x, cos, sin, past_kv=caches[i], use_cache=True, absorbed=absorbed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    caches = fresh_caches()
    t0 = time.perf_counter()
    for _ in range(steps):
        x = model.tok_emb(tok)
        with ctx:
            for i, blk in enumerate(model.blocks):
                x, caches[i] = blk(x, cos, sin, past_kv=caches[i], use_cache=True, absorbed=absorbed)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return {
        "context_len": context_len,
        "decode_ms_per_step": dt * 1e3,
        "decode_tok_per_s": batch / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def measure_train_step(model, batch, seq_len, device, dtype, warmup=3, iters=8) -> dict:
    model.train()
    idx = torch.randint(0, model.config.vocab_size, (batch, seq_len), device=device)
    tgt = torch.randint(0, model.config.vocab_size, (batch, seq_len), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.startswith("cuda"))
    ctx = _autocast(device, dtype)

    def step():
        opt.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(idx, tgt)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    model.eval()
    return {
        "train_ms_per_step": dt * 1e3,
        "train_tok_per_s": batch * seq_len / dt,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


@torch.no_grad()
def max_decode_context(model, batch, device, dtype, ceil_ctx, lo=256) -> int:
    """Largest context length whose decode step fits in memory (binary search)."""
    absorbed = is_mla(model)
    cos, sin = model.cos, model.sin
    tok = torch.randint(0, model.config.vocab_size, (batch, 1), device=device)
    ctx = _autocast(device, dtype)

    def fits(L: int) -> bool:
        if L >= model.config.block_size:
            return False
        try:
            caches = [blk.attn.init_cache(batch, L, device, dtype) for blk in model.blocks]
            x = model.tok_emb(tok)
            with ctx:
                for i, blk in enumerate(model.blocks):
                    x, _ = blk(x, cos, sin, past_kv=caches[i], use_cache=True, absorbed=absorbed)
            torch.cuda.synchronize()
            del caches, x
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False

    best = 0
    while lo <= ceil_ctx:
        mid = (lo + ceil_ctx) // 2
        if fits(mid):
            best, lo = mid, mid + 1
        else:
            ceil_ctx = mid - 1
    return best

"""Quality evaluation: held-out loss, perplexity, bits-per-byte, and samples.

The "quality" half of the study (the efficiency half lives in ``benchmark.py``).
All metrics are computed identically for every attention variant so the only
difference is ``GPTConfig.attn_type``.

Perplexity is reported from a *deterministic* non-overlapping sweep of the whole
validation stream (not random batches) so a run's headline number is exactly
reproducible. Bits-per-byte rescales that by the corpus token/byte ratio, giving
a tokenizer-independent figure comparable to the LM literature.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import GPTConfig
from .data import EOT, dataset_num_tokens, get_encoder, read_meta
from .model import GPT

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _autocast(device: str, dtype: torch.dtype):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_checkpoint(ckpt_path: str | Path, device: str = "cuda") -> tuple[GPT, dict]:
    """Rebuild a model from a training checkpoint (best.pt). Returns (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, ckpt


@torch.no_grad()
def full_val_loss(model, data_dir, block_size, device, dtype, batch_size=32,
                  split="val", max_tokens=None) -> dict:
    """Mean cross-entropy (nats/token) over a non-overlapping sweep of ``split``.

    Each token is predicted exactly once with up to ``block_size`` of left
    context. Returns the token-weighted mean loss and the number of predicted
    tokens, so callers can derive perplexity and bits-per-byte.
    """
    path = Path(data_dir) / f"{split}.bin"
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(data)
    if max_tokens is not None:
        n = min(n, max_tokens + 1)
    # Non-overlapping windows; each contributes block_size predicted tokens.
    starts = list(range(0, n - block_size - 1, block_size))
    ctx = _autocast(device, dtype)
    model.eval()

    loss_sum = 0.0
    tok_total = 0
    for b in range(0, len(starts), batch_size):
        batch_starts = starts[b : b + batch_size]
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in batch_starts])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in batch_starts])
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        ntok = x.numel()
        loss_sum += loss.item() * ntok
        tok_total += ntok

    mean_loss = loss_sum / max(1, tok_total)
    return {"loss": mean_loss, "ppl": math.exp(mean_loss), "eval_tokens": tok_total}


def val_byte_count(data_dir, split="val") -> int:
    """UTF-8 byte length of the decoded ``split`` stream (cached in the data dir).

    EOT separators carry no source bytes and are excluded. GPT-2 BPE is
    byte-level, so ``decode_bytes`` reconstructs the corpus bytes exactly and is
    safe to apply chunk-by-chunk.
    """
    data_dir = Path(data_dir)
    cache = data_dir / f"{split}_bytes.json"
    if cache.exists():
        return int(json.loads(cache.read_text())["bytes"])

    enc = get_encoder()
    data = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    total = 0
    chunk = 1 << 20
    for i in range(0, len(data), chunk):
        ids = [int(t) for t in data[i : i + chunk] if int(t) != EOT]
        total += len(enc.decode_bytes(ids))
    cache.write_text(json.dumps({"bytes": total, "tokens": int(len(data))}))
    return total


def bits_per_byte(loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Convert nats/token to bits/byte using the corpus token/byte ratio."""
    return (loss_nats / math.log(2)) * (n_tokens / n_bytes)


@torch.no_grad()
def generate_samples(model, prompts, device, dtype, max_new_tokens=200,
                     temperature=0.8, top_k=200) -> list[dict]:
    enc = get_encoder()
    model.eval()
    ctx = _autocast(device, dtype)
    out = []
    for prompt in prompts:
        ids = enc.encode_ordinary(prompt) or [EOT]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with ctx:
            gen = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        text = enc.decode(gen[0].tolist())
        out.append({"prompt": prompt, "completion": text})
    return out


def evaluate_run(run_dir, data_dir=None, device="cuda", dtype="bfloat16",
                 batch_size=32, split="val", prompts=None, max_new_tokens=200) -> dict:
    """Evaluate one training run directory: load best.pt, compute quality metrics."""
    run_dir = Path(run_dir)
    cfg_json = json.loads((run_dir / "config.json").read_text())
    if data_dir is None:
        data_dir = cfg_json["train"]["data_dir"]
    block_size = cfg_json["model"]["block_size"]
    td = _DTYPES[dtype]

    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    vl = full_val_loss(model, data_dir, block_size, device, td, batch_size, split)
    n_bytes = val_byte_count(data_dir, split)
    n_tokens = dataset_num_tokens(data_dir, split)
    bpb = bits_per_byte(vl["loss"], n_tokens, n_bytes)

    result = {
        "run": str(run_dir),
        "attn_type": cfg_json["model"]["attn_type"],
        "params": cfg_json.get("params"),
        "kv_bytes_per_token": cfg_json.get("kv_bytes_per_token"),
        "ckpt_iter": ckpt.get("iter"),
        "val_loss": vl["loss"],
        "perplexity": vl["ppl"],
        "bits_per_byte": bpb,
        "eval_tokens": vl["eval_tokens"],
    }
    if prompts:
        result["samples"] = generate_samples(model, prompts, device, td, max_new_tokens=max_new_tokens)
    return result

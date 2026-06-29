"""Recall-capacity evaluation: the metrics aggregate perplexity hides.

Three probes, all reusing the trained checkpoints:

* ``recall_accuracy`` -- exact-match accuracy at the answer positions of a
  synthetic task (MQAR / copy). This is the headline metric; cross-entropy is only
  a training proxy.
* ``difficulty_sweep`` -- accuracy as the task gets harder. Evaluating a model
  *beyond* its trained difficulty is the train-short / test-long generalization
  probe.
* ``position_wise_loss`` -- natural-text CE bucketed by token position. The bridge
  back to Paper #1: the same averaged loss, decomposed, shows whether a mechanism
  pays for KV sharing specifically at the later positions that need retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .eval import _DTYPES, _autocast, load_checkpoint
from .synthetic import build_batch, sequence_len

# Difficulty grids for the extrapolation sweep (filtered to fit block_size).
_AXIS = {"mqar": "num_pairs", "copy": "num_tokens", "selective_copy": "num_tokens"}
_SWEEP_VALUES = {
    "mqar": [16, 32, 48, 64, 96, 128, 192, 256, 384, 480],
    "copy": [16, 32, 48, 64, 96, 128, 192, 256, 384, 512],
    # x length = seq_len + num_tokens = 4*num_tokens (seq_len defaults to 3*K),
    # so 256 is the largest that fits block_size 1024; bigger values auto-dropped.
    "selective_copy": [16, 32, 48, 64, 96, 128, 192, 256],
}


@torch.no_grad()
def recall_accuracy(model, task, vocab_size, params, device="cuda",
                    dtype=torch.bfloat16, batch_size=128, num_batches=8, seed=0) -> dict:
    """Exact-match accuracy over answer positions, averaged over fresh batches."""
    model.eval()
    ctx = _autocast(device, dtype)
    correct = total = 0
    for b in range(num_batches):
        x, y = build_batch(task, batch_size, vocab_size, seed=seed + b, **params)
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _ = model(x, y)
        pred = logits.argmax(dim=-1)
        mask = y != -1
        correct += int((pred[mask] == y[mask]).sum())
        total += int(mask.sum())
    return {"accuracy": correct / max(1, total), "answers": total,
            "seq_len": sequence_len(task, **params)}


def difficulty_sweep(model, task, vocab_size, base_params, block_size, device="cuda",
                     dtype=torch.bfloat16, batch_size=128, num_batches=8, seed=0) -> list[dict]:
    """Accuracy vs difficulty for one model (values that overflow block_size dropped)."""
    axis = _AXIS[task]
    out = []
    for v in _SWEEP_VALUES[task]:
        p = dict(base_params)
        p[axis] = v
        if task == "mqar":  # queries can't exceed stored pairs
            p["num_queries"] = min(base_params.get("num_queries", v), v)
        if sequence_len(task, **p) > block_size:
            continue
        r = recall_accuracy(model, task, vocab_size, p, device, dtype,
                            batch_size, num_batches, seed)
        out.append({"difficulty": v, **r})
    return out


@torch.no_grad()
def position_wise_loss(model, data_dir, block_size, device="cuda", dtype=torch.bfloat16,
                       batch_size=16, split="val", num_batches=50, n_buckets=16, seed=0) -> dict:
    """Mean CE (nats) per token position over random windows, plus bucketed means."""
    model.eval()
    ctx = _autocast(device, dtype)
    data = np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(seed)
    sums = torch.zeros(block_size, dtype=torch.float64)
    n = 0
    for _ in range(num_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=g)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _ = model(x, y)
        lt = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             reduction="none").reshape(x.shape)
        sums += lt.sum(dim=0).double().cpu()
        n += x.shape[0]
    per_pos = (sums / max(1, n))
    edges = np.linspace(0, block_size, n_buckets + 1, dtype=int)
    buckets = [{"start": int(edges[i]), "end": int(edges[i + 1]),
                "center": int((edges[i] + edges[i + 1]) // 2),
                "loss": float(per_pos[edges[i]:edges[i + 1]].mean())}
               for i in range(n_buckets) if edges[i + 1] > edges[i]]
    return {"per_position": per_pos.tolist(), "buckets": buckets}


def evaluate_recall_run(run_dir, device="cuda", dtype="bfloat16", batch_size=128,
                        num_batches=8, seed=0) -> dict:
    """Load one synthetic run and compute in-distribution accuracy + a difficulty sweep."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    sp = run_dir / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    task = cfg["train"]["task"]
    params = cfg["train"]["task_params"]
    vocab = cfg["model"]["vocab_size"]
    block = cfg["model"]["block_size"]
    td = _DTYPES[dtype]

    model, _ = load_checkpoint(run_dir / "best.pt", device)
    indist = recall_accuracy(model, task, vocab, params, device, td, batch_size, num_batches, seed)
    sweep = difficulty_sweep(model, task, vocab, params, block, device, td,
                             batch_size, num_batches, seed)
    return {
        "run": str(run_dir),
        "attn_type": cfg["model"]["attn_type"],
        "task": task,
        "trained_difficulty": params[_AXIS[task]],
        "train_seed": cfg["train"].get("seed"),
        "params": cfg.get("params"),
        "kv_bytes_per_token": cfg.get("kv_bytes_per_token"),
        "accuracy": indist["accuracy"],
        "answers": indist["answers"],
        "iters_to_grok": summary.get("iters_to_grok"),
        "groked": summary.get("groked"),
        "final_iter": summary.get("final_iter"),
        "best_val_loss": summary.get("best_val_loss"),
        "sweep": sweep,
    }

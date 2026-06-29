"""Train one attention variant from a base YAML config.

One base config trains every variant; only --attn changes, keeping the study
controlled. Example::

    uv run python scripts/train.py --config configs/tinystories_base.yaml --attn mla --name ts_mla
    uv run python scripts/train.py --config configs/tinystories_base.yaml --attn gqa --set train.max_iters=2000
"""

from __future__ import annotations

import os

# Full-vocab logits dominate memory on 16GB cards; reduce allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from mla_gpt.config import GPTConfig
from mla_gpt.training import TrainConfig, run_training


def _coerce(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for ov in overrides:
        key, val = ov.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:                 # walk/create nested dicts
            d = d.setdefault(p, {})
        d[parts[-1]] = _coerce(val)          # e.g. train.task_params.num_pairs=128


def _filter(d: dict, dc_type) -> dict:
    valid = {f.name for f in fields(dc_type)}
    unknown = set(d) - valid
    if unknown:
        raise SystemExit(f"unknown {dc_type.__name__} fields: {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in valid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attn", choices=["mha", "mqa", "gqa", "mla"], default=None)
    ap.add_argument("--name", default=None, help="run name -> runs/<name>")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override, e.g. --set train.max_iters=2000 --set model.kv_lora_rank=128")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    if args.attn:
        cfg["model"]["attn_type"] = args.attn
    _apply_overrides(cfg, args.overrides)

    attn = cfg["model"].get("attn_type", "mha")
    name = args.name or f"{Path(args.config).stem}_{attn}"
    # RUNS_DIR lets a sweep write to non-synced scratch (e.g. /tmp) instead of the
    # OneDrive-synced repo; defaults to "runs" so single runs are unaffected.
    cfg["train"]["out_dir"] = f"{os.environ.get('RUNS_DIR', 'runs')}/{name}"

    model_cfg = GPTConfig(**_filter(cfg["model"], GPTConfig))
    train_cfg = TrainConfig(**_filter(cfg["train"], TrainConfig))
    run_training(model_cfg, train_cfg)


if __name__ == "__main__":
    main()

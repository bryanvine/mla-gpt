"""Prepare TinyStories as tokenized uint16 shards for fast dev iteration.

    uv run python scripts/prepare_tinystories.py [--num-proc N]

Writes data/tinystories/{train.bin,val.bin,meta.json}.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

from mla_gpt.data import tokenize_split_to_bin, write_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-proc", type=int, default=min(8, (os.cpu_count() or 4)))
    ap.add_argument("--out", type=str, default="data/tinystories")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading roneneldan/TinyStories ...")
    ds = load_dataset("roneneldan/TinyStories")
    text_col = "text"

    n_train = tokenize_split_to_bin(ds["train"], text_col, out_dir / "train.bin", args.num_proc)
    n_val = tokenize_split_to_bin(ds["validation"], text_col, out_dir / "val.bin", args.num_proc)

    write_meta(out_dir, dataset="roneneldan/TinyStories", train_tokens=n_train, val_tokens=n_val)
    print(f"Done. train={n_train:,} tokens  val={n_val:,} tokens -> {out_dir}")


if __name__ == "__main__":
    main()

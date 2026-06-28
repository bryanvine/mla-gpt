"""Prepare the FineWeb-Edu 10B sample as tokenized uint16 shards (headline corpus).

    uv run python scripts/prepare_fineweb.py [--num-proc N] [--val-tokens 50_000_000]

Streams HuggingFaceFW/fineweb-edu (sample-10BT), tokenizes with GPT-2 BPE, and
writes data/fineweb_edu_10b/{train.bin,val.bin,meta.json}. ~10B tokens ~= 20 GB
on disk; ensure adequate space before running.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

from mla_gpt.data import tokenize_split_to_bin, write_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-proc", type=int, default=min(16, (os.cpu_count() or 4)))
    ap.add_argument("--out", type=str, default="data/fineweb_edu_10b")
    ap.add_argument("--val-tokens", type=int, default=50_000_000, help="approx held-out token budget")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HuggingFaceFW/fineweb-edu (sample-10BT) ...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    # Carve a small held-out validation set from the front, train on the rest.
    # ~ val_tokens / avg_doc_len docs; FineWeb-Edu averages ~1.1k tokens/doc.
    n_val_docs = max(1, args.val_tokens // 1100)
    val_ds = ds.select(range(n_val_docs))
    train_ds = ds.select(range(n_val_docs, len(ds)))
    text_col = "text"

    n_val = tokenize_split_to_bin(val_ds, text_col, out_dir / "val.bin", args.num_proc)
    n_train = tokenize_split_to_bin(train_ds, text_col, out_dir / "train.bin", args.num_proc)

    write_meta(
        out_dir,
        dataset="HuggingFaceFW/fineweb-edu:sample-10BT",
        train_tokens=n_train,
        val_tokens=n_val,
    )
    print(f"Done. train={n_train:,} tokens  val={n_val:,} tokens -> {out_dir}")


if __name__ == "__main__":
    main()

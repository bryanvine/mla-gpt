"""Tokenization and data loading.

Token streams are stored as flat ``uint16`` ``.bin`` files (GPT-2 BPE, vocab
50257 < 2**16) with documents separated by the end-of-text token. Batches are
sampled as random contiguous windows, nanoGPT-style, via a fresh memmap per call
to avoid leaking page cache.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import torch

EOT = 50256  # GPT-2 <|endoftext|>


@functools.lru_cache(maxsize=1)
def get_encoder():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def _process_example(example: dict, text_col: str) -> dict:
    enc = get_encoder()
    ids = enc.encode_ordinary(example[text_col])
    ids.append(EOT)
    return {"ids": ids, "len": len(ids)}


def tokenize_split_to_bin(ds, text_col: str, out_path: str | Path, num_proc: int = 8) -> int:
    """Tokenize a HuggingFace dataset split and write a flat uint16 .bin. Returns token count."""
    from tqdm import tqdm

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized = ds.map(
        _process_example,
        fn_kwargs={"text_col": text_col},
        remove_columns=ds.column_names,
        desc=f"tokenizing -> {out_path.name}",
        num_proc=num_proc,
    )
    arr_len = int(np.sum(tokenized["len"], dtype=np.uint64))
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(arr_len,))
    n_shards = 1024
    idx = 0
    for i in tqdm(range(n_shards), desc=f"writing {out_path.name}"):
        batch = tokenized.shard(num_shards=n_shards, index=i, contiguous=True).with_format("numpy")
        if len(batch) == 0:
            continue
        chunk = np.concatenate(batch["ids"])
        arr[idx : idx + len(chunk)] = chunk
        idx += len(chunk)
    arr.flush()
    return arr_len


def write_meta(out_dir: str | Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    meta = {"vocab_size": 50257, "eot": EOT, "dtype": "uint16", **kwargs}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def read_meta(data_dir: str | Path) -> dict:
    p = Path(data_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def make_get_batch(data_dir: str | Path, block_size: int, device: str = "cuda"):
    """Return a get_batch(split, batch_size) closure over memmapped .bin files."""
    data_dir = Path(data_dir)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def get_batch(split: str, batch_size: int):
        path = data_dir / f"{split}.bin"
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch


def dataset_num_tokens(data_dir: str | Path, split: str) -> int:
    path = Path(data_dir) / f"{split}.bin"
    return int(os.path.getsize(path) // 2)  # uint16 = 2 bytes

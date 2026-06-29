"""Synthetic capacity probes: associative recall and selective copy.

Paper #1 measured *aggregate* next-token quality (perplexity / bits-per-byte),
which is the metric least sensitive to how many KV heads a mechanism keeps. These
tasks isolate the capability MQA's single shared KV head actually sacrifices:
holding many key->value associations and retrieving them on demand.

Every task emits the same contract as ``data.make_get_batch``::

    get_batch(split, batch_size) -> (x, y)

with ``y`` masked to ``-1`` everywhere except the answer positions, so the model's
existing ``cross_entropy(..., ignore_index=-1)`` scores recall only. The attention
mechanism is still the sole independent variable; only the *data* changes.

Tasks
-----
* ``mqar``  : multi-query associative recall (Arora et al., "Zoology", 2023). A
              run of ``num_pairs`` (key, value) tokens, then ``num_queries`` of the
              keys re-presented; predict each queried key's value.
* ``copy``  : plain copy (control). ``num_tokens`` content symbols, a separator, then
              the same content again; predict the second copy. A fixed positional
              offset solves it, so it groks almost immediately -- a non-discriminating
              baseline kept to show that mere copying is *not* what stresses the cache.
* ``selective_copy`` : the real probe (Gu & Dao, "Mamba", 2023). ``num_tokens`` content
              symbols are scattered at random positions among blank filler, then a
              separator; the model must emit them in order. The random scatter forces
              *content-based* in-order selection, the capability a shrunk KV cache
              should struggle to retain.
"""

from __future__ import annotations

from typing import Callable

import torch

# A task builder maps (batch_size, vocab_size, generator, **params) -> (x, y).
Builder = Callable[..., tuple[torch.Tensor, torch.Tensor]]

IGNORE = -1  # label value skipped by cross_entropy(ignore_index=-1)


def _mqar(batch_size: int, vocab_size: int, g: torch.Generator, *,
          num_pairs: int, num_queries: int, num_symbols: int | None = None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-query associative recall.

    Sequence = [k1 v1 k2 v2 ... kN vN | kq1 vq1 ... kqQ vqQ]. Keys are distinct
    within a sequence; the query block re-presents Q of them in random order. The
    label at each query-key position is that key's stored value; everything else
    is masked. Difficulty scales with ``num_pairs`` (associations to hold).
    """
    S = vocab_size if num_symbols is None else num_symbols
    if not 0 < num_pairs <= S:
        raise ValueError(f"need 0 < num_pairs ({num_pairs}) <= num_symbols ({S})")
    if not 0 < num_queries <= num_pairs:
        raise ValueError(f"need 0 < num_queries ({num_queries}) <= num_pairs ({num_pairs})")
    if S > vocab_size:
        raise ValueError(f"num_symbols ({S}) exceeds vocab_size ({vocab_size})")

    B = batch_size
    # Distinct keys per row: a partial random permutation of the symbol set.
    keys = torch.rand(B, S, generator=g).argsort(dim=1)[:, :num_pairs]
    vals = torch.randint(0, S, (B, num_pairs), generator=g)

    store = torch.empty(B, 2 * num_pairs, dtype=torch.long)
    store[:, 0::2] = keys
    store[:, 1::2] = vals

    q_idx = torch.rand(B, num_pairs, generator=g).argsort(dim=1)[:, :num_queries]
    query = torch.empty(B, 2 * num_queries, dtype=torch.long)
    query[:, 0::2] = torch.gather(keys, 1, q_idx)
    query[:, 1::2] = torch.gather(vals, 1, q_idx)

    tokens = torch.cat([store, query], dim=1)          # (B, 2N + 2Q)
    x = tokens[:, :-1].contiguous()
    y = torch.full_like(x, IGNORE)
    cols = 2 * num_pairs + 2 * torch.arange(num_queries)  # query-key positions
    y[:, cols] = tokens[:, cols + 1]                    # = the queried values
    return x, y


def _copy(batch_size: int, vocab_size: int, g: torch.Generator, *,
          num_tokens: int, num_symbols: int | None = None
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective copy.

    Sequence = [c1 ... cK <SEP> c1 ... cK]; predict the second copy. ``<SEP>`` is
    symbol id ``num_symbols`` (so ``vocab_size`` must exceed the content alphabet).
    Difficulty scales with ``num_tokens`` (how much must be carried across <SEP>).
    """
    S = vocab_size - 1 if num_symbols is None else num_symbols
    if num_tokens <= 0:
        raise ValueError(f"need num_tokens > 0, got {num_tokens}")
    if S + 1 > vocab_size:
        raise ValueError(f"num_symbols ({S}) + 1 separator exceeds vocab_size ({vocab_size})")

    B, K, sep = batch_size, num_tokens, S
    content = torch.randint(0, S, (B, K), generator=g)
    tokens = torch.cat([content, torch.full((B, 1), sep, dtype=torch.long), content], dim=1)
    x = tokens[:, :-1].contiguous()                    # (B, 2K)
    y = torch.full_like(x, IGNORE)
    cols = torch.arange(K, 2 * K)                       # <SEP> + output region
    y[:, cols] = tokens[:, cols + 1]                   # = the content tokens
    return x, y


def _selective_copy(batch_size: int, vocab_size: int, g: torch.Generator, *,
                    num_tokens: int, seq_len: int | None = None,
                    num_symbols: int | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective copy (Gu & Dao, "Mamba", 2023).

    Sequence = [ region of length L with K content tokens scattered at random
    positions among ``blank`` filler | <SEP> | the K content tokens in order ].
    Unlike plain ``copy`` (a fixed positional offset suffices), the random scatter
    forces content-based selection: the model must find the content among the blanks
    and emit it in order, which is what a shrunk KV cache should struggle to retain.

    Vocab layout: content ``0..S-1``, ``blank = S``, ``<SEP> = S+1`` (so
    ``vocab_size`` must exceed the alphabet by 2). ``seq_len`` (L) defaults to
    ``3 * num_tokens``; difficulty scales with ``num_tokens`` (items to select).
    """
    S = vocab_size - 2 if num_symbols is None else num_symbols
    K = num_tokens
    L = 3 * K if not seq_len else seq_len
    if K <= 0:
        raise ValueError(f"need num_tokens > 0, got {K}")
    if L < K:
        raise ValueError(f"need seq_len ({L}) >= num_tokens ({K})")
    if S + 2 > vocab_size:
        raise ValueError(f"num_symbols ({S}) + blank + separator exceeds vocab_size ({vocab_size})")

    B, blank, sep = batch_size, S, S + 1
    content = torch.randint(0, S, (B, K), generator=g)
    # K distinct positions per row, sorted so the content keeps left->right order
    pos = torch.rand(B, L, generator=g).argsort(dim=1)[:, :K].sort(dim=1).values
    region = torch.full((B, L), blank, dtype=torch.long)
    region.scatter_(1, pos, content)                   # drop content into the blanks
    sep_col = torch.full((B, 1), sep, dtype=torch.long)
    tokens = torch.cat([region, sep_col, content], dim=1)   # (B, L + 1 + K)
    x = tokens[:, :-1].contiguous()                    # (B, L + K)
    y = torch.full_like(x, IGNORE)
    cols = torch.arange(L, L + K)                       # <SEP> + output region
    y[:, cols] = tokens[:, cols + 1]                   # = the content, in order
    return x, y


_TASKS: dict[str, Builder] = {"mqar": _mqar, "copy": _copy,
                              "selective_copy": _selective_copy}


def sequence_len(task: str, **params) -> int:
    """Length of ``x`` (model input) for a task's parameters, for config sizing."""
    if task == "mqar":
        return 2 * params["num_pairs"] + 2 * params["num_queries"] - 1
    if task == "copy":
        return 2 * params["num_tokens"]
    if task == "selective_copy":
        K = params["num_tokens"]
        L = params["seq_len"] if params.get("seq_len") else 3 * K
        return L + K
    raise ValueError(f"unknown task: {task}")


def build_batch(task: str, batch_size: int, vocab_size: int, *, seed: int, **params
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic single batch (CPU). Used by evaluation for reproducible sweeps."""
    if task not in _TASKS:
        raise ValueError(f"unknown task: {task} (have {sorted(_TASKS)})")
    g = torch.Generator().manual_seed(seed)
    return _TASKS[task](batch_size, vocab_size, g, **params)


def make_synthetic_get_batch(task: str, block_size: int, vocab_size: int,
                             device: str = "cuda", *, seed: int = 1337, **params):
    """Return a ``get_batch(split, batch_size) -> (x, y)`` over a synthetic task.

    Mirrors ``data.make_get_batch`` so ``run_training`` is agnostic to the source.
    ``train`` and ``val`` use separate persistent generators; ``val`` advances
    deterministically (seed + call count) so its loss is reproducible across runs.
    """
    if task not in _TASKS:
        raise ValueError(f"unknown task: {task} (have {sorted(_TASKS)})")
    builder = _TASKS[task]
    gens = {"train": torch.Generator().manual_seed(seed),
            "val": torch.Generator().manual_seed(seed + 1)}
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def get_batch(split: str, batch_size: int):
        x, y = builder(batch_size, vocab_size, gens[split], **params)
        if x.shape[1] > block_size:
            raise ValueError(f"{task} sequence len {x.shape[1]} > block_size {block_size}")
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch

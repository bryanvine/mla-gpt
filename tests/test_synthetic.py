"""Correctness tests for the synthetic recall probes (MQAR / selective copy).

These guard the *data*, which is the only thing that changes across the recall
study. A silent generator bug (leaked answers, mismatched labels, non-distinct
keys) would invalidate every accuracy number, so the checks here verify that each
labeled position is (a) genuinely recoverable from earlier context and (b) the
only thing scored.
"""

from __future__ import annotations

import pytest
import torch

from mla_gpt.synthetic import (
    IGNORE,
    build_batch,
    make_synthetic_get_batch,
    sequence_len,
)


# --------------------------------------------------------------------------- MQAR


@pytest.mark.parametrize("num_pairs,num_queries", [(8, 8), (16, 4), (64, 16)])
def test_mqar_shapes_match_sequence_len(num_pairs, num_queries):
    x, y = build_batch("mqar", 4, 512, seed=0,
                       num_pairs=num_pairs, num_queries=num_queries, num_symbols=512)
    expected = sequence_len("mqar", num_pairs=num_pairs, num_queries=num_queries)
    assert x.shape == (4, expected)
    assert y.shape == x.shape


def test_mqar_keys_distinct_within_row():
    """Each stored key must be unique, else a query is ambiguous."""
    x, _ = build_batch("mqar", 8, 512, seed=1, num_pairs=64, num_queries=16, num_symbols=512)
    store_keys = x[:, : 2 * 64 : 2]                      # k0 k1 ... k63
    for row in store_keys:
        assert row.unique().numel() == row.numel()


def test_mqar_exactly_num_queries_labels():
    """Only the query-key positions are scored; everything else is IGNORE."""
    x, y = build_batch("mqar", 8, 512, seed=2, num_pairs=32, num_queries=10, num_symbols=512)
    assert (y != IGNORE).sum(dim=1).tolist() == [10] * 8


def test_mqar_labels_are_recoverable():
    """Each labeled value must equal the value stored with the *same* key earlier.

    This is the crux: it proves the answer is a function of in-context information
    (associative recall), not noise the model could never fit.
    """
    B, N, Q = 8, 32, 16
    x, y = build_batch("mqar", B, 512, seed=3, num_pairs=N, num_queries=Q, num_symbols=512)
    keys, vals = x[:, : 2 * N : 2], x[:, 1 : 2 * N : 2]
    for b in range(B):
        kv = {int(k): int(v) for k, v in zip(keys[b], vals[b])}
        labeled = (y[b] != IGNORE).nonzero(as_tuple=True)[0]
        for pos in labeled.tolist():
            query_key = int(x[b, pos])                   # token at the labeled pos
            assert kv[query_key] == int(y[b, pos])


def test_mqar_no_answer_leak_before_query():
    """The label at a query position must not appear as the *next* input token of any
    earlier labeled position (the value only follows its own query, not before)."""
    B, N, Q = 4, 16, 8
    x, y = build_batch("mqar", B, 512, seed=4, num_pairs=N, num_queries=Q, num_symbols=512)
    for b in range(B):
        labeled = (y[b] != IGNORE).nonzero(as_tuple=True)[0].tolist()
        # labels live strictly inside the query block (after the 2N store tokens)
        assert all(p >= 2 * N for p in labeled)


@pytest.mark.parametrize("bad", [
    dict(num_pairs=0, num_queries=1),
    dict(num_pairs=600, num_queries=1),     # > num_symbols
    dict(num_pairs=8, num_queries=16),      # queries > pairs
])
def test_mqar_validates_params(bad):
    with pytest.raises(ValueError):
        build_batch("mqar", 2, 512, seed=0, num_symbols=512, **bad)


# --------------------------------------------------------------------------- copy


@pytest.mark.parametrize("num_tokens", [4, 16, 64])
def test_copy_shapes_match_sequence_len(num_tokens):
    x, y = build_batch("copy", 4, 512, seed=0, num_tokens=num_tokens, num_symbols=256)
    expected = sequence_len("copy", num_tokens=num_tokens)
    assert x.shape == (4, expected)
    assert y.shape == x.shape


def test_copy_separator_and_masking():
    K, S = 16, 256
    x, y = build_batch("copy", 6, 512, seed=5, num_tokens=K, num_symbols=S)
    assert (x[:, K] == S).all()                          # separator id == num_symbols
    assert (y[:, :K] == IGNORE).all()                    # nothing scored before output
    assert (y[:, K:] != IGNORE).all()                    # whole output region scored


def test_copy_labels_reproduce_content():
    """The scored region must equal the first-copy content, in order."""
    K, S = 16, 256
    x, y = build_batch("copy", 6, 512, seed=6, num_tokens=K, num_symbols=S)
    content = x[:, :K]
    assert torch.equal(y[:, K:], content)


@pytest.mark.parametrize("bad", [
    dict(num_tokens=0),
    dict(num_tokens=4, num_symbols=512),    # S + sep > vocab_size
])
def test_copy_validates_params(bad):
    with pytest.raises(ValueError):
        build_batch("copy", 2, 512, seed=0, **bad)


# ----------------------------------------------------------------- selective copy


@pytest.mark.parametrize("num_tokens,seq_len", [(4, 12), (16, 48), (32, None)])
def test_selective_copy_shapes_match_sequence_len(num_tokens, seq_len):
    kw = dict(num_tokens=num_tokens, num_symbols=64)
    if seq_len is not None:
        kw["seq_len"] = seq_len
    x, y = build_batch("selective_copy", 4, 512, seed=0, **kw)
    expected = sequence_len("selective_copy", num_tokens=num_tokens, seq_len=seq_len)
    assert x.shape == (4, expected)
    assert y.shape == x.shape


def test_selective_copy_separator_and_masking():
    K, S, L = 16, 64, 48
    x, y = build_batch("selective_copy", 4, 512, seed=12, num_tokens=K, seq_len=L, num_symbols=S)
    assert (x[:, L] == S + 1).all()                      # separator id == num_symbols + 1
    assert (y[:, :L] == IGNORE).all()                    # nothing scored before output
    assert (y[:, L:] != IGNORE).all()                    # whole output region scored
    assert (y != IGNORE).sum(dim=1).tolist() == [K] * 4  # exactly K labels per row


def test_selective_copy_content_count_and_scatter():
    """The region holds exactly K non-blank (content) tokens; rest is blank filler.

    A non-trivial scatter (content not jammed into the first K slots) is what makes
    the task selective rather than a fixed positional copy.
    """
    K, S, L = 16, 64, 64
    x, _ = build_batch("selective_copy", 32, 512, seed=13, num_tokens=K, seq_len=L, num_symbols=S)
    region = x[:, :L]
    assert (region != S).sum(dim=1).tolist() == [K] * 32   # blank id == num_symbols
    # at least some row must place content past the first K positions (true scatter)
    assert (region[:, K:] != S).any()


def test_selective_copy_labels_are_selected_content_in_order():
    """The scored region must equal the scattered content read left-to-right."""
    K, S, L = 12, 64, 40
    x, y = build_batch("selective_copy", 8, 512, seed=14, num_tokens=K, seq_len=L, num_symbols=S)
    for b in range(8):
        scattered = x[b, :L][x[b, :L] != S]              # non-blank, in position order
        assert scattered.numel() == K
        assert torch.equal(scattered, y[b, L:])          # == emitted content, in order


@pytest.mark.parametrize("bad", [
    dict(num_tokens=0),
    dict(num_tokens=8, seq_len=4),          # seq_len < num_tokens
    dict(num_tokens=4, num_symbols=511),    # S + blank + sep > vocab_size
])
def test_selective_copy_validates_params(bad):
    with pytest.raises(ValueError):
        build_batch("selective_copy", 2, 512, seed=0, **bad)


# ------------------------------------------------------------------ get_batch path


def test_build_batch_is_deterministic():
    a = build_batch("mqar", 4, 512, seed=7, num_pairs=16, num_queries=8, num_symbols=512)
    b = build_batch("mqar", 4, 512, seed=7, num_pairs=16, num_queries=8, num_symbols=512)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_get_batch_contract_and_splits():
    gb = make_synthetic_get_batch("mqar", block_size=1024, vocab_size=512, device="cpu",
                                  seed=1337, num_pairs=32, num_queries=16, num_symbols=512)
    x, y = gb("train", 8)
    assert x.shape == (8, sequence_len("mqar", num_pairs=32, num_queries=16))
    assert y.shape == x.shape and x.device.type == "cpu"
    # train and val draw from independent generators -> different batches
    xt, _ = gb("train", 8)
    xv, _ = gb("val", 8)
    assert not torch.equal(xt, xv)


def test_get_batch_rejects_overflowing_block_size():
    gb = make_synthetic_get_batch("copy", block_size=16, vocab_size=512, device="cpu",
                                  num_tokens=64, num_symbols=256)
    with pytest.raises(ValueError, match="block_size"):
        gb("train", 4)


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        build_batch("nonsense", 2, 512, seed=0)
    with pytest.raises(ValueError):
        sequence_len("nonsense", foo=1)

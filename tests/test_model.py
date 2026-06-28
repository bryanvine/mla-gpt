"""Correctness tests for the GPT backbone and all attention variants.

Strict numerical checks (KV-cache equivalence, causality) run on CPU under the
math SDPA backend so they are exact and backend-independent.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mla_gpt.attention import MultiHeadLatentAttention
from mla_gpt.config import GPTConfig
from mla_gpt.model import GPT
from mla_gpt.rope import build_rope_cache

VARIANTS = ["mha", "mqa", "gqa", "mla"]


def tiny_cfg(attn_type: str, **kw) -> GPTConfig:
    base = dict(
        vocab_size=128, block_size=64, n_layer=3, n_head=4, d_model=128,
        attn_type=attn_type, n_kv_head=2,
        kv_lora_rank=64, qk_nope_head_dim=24, qk_rope_head_dim=8, v_head_dim=32,
    )
    base.update(kw)
    return GPTConfig(**base)


def incremental_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """All-position logits computed one token at a time via the KV cache."""
    B, T = idx.shape
    caches = [None] * len(model.blocks)
    outs = []
    for t in range(T):
        x = model.tok_emb(idx[:, t : t + 1])
        new = []
        for i, block in enumerate(model.blocks):
            x, c = block(x, model.cos, model.sin, past_kv=caches[i], use_cache=True)
            new.append(c)
        caches = new
        outs.append(model.lm_head(model.norm_f(x)))
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("attn", VARIANTS)
def test_forward_backward(attn):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(attn))
    idx = torch.randint(0, 128, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("attn", VARIANTS)
def test_kv_cache_equivalence(attn):
    """Incremental decode with the cache must match a single full forward pass."""
    torch.manual_seed(0)
    m = GPT(tiny_cfg(attn)).eval().to(torch.float64)
    idx = torch.randint(0, 128, (2, 24))
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        full, _ = m(idx, idx)          # (B, T, V) all positions
        incr = incremental_logits(m, idx)
    assert torch.allclose(full, incr, atol=1e-9, rtol=1e-6), (
        f"{attn}: max abs diff {(full - incr).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("attn", VARIANTS)
def test_causal_no_future_leak(attn):
    """Changing a future token must not affect earlier-position logits."""
    torch.manual_seed(0)
    m = GPT(tiny_cfg(attn)).eval().to(torch.float64)
    idx = torch.randint(0, 128, (1, 20))
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        a, _ = m(idx, idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % 128       # perturb only the last token
        b, _ = m(idx2, idx2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-9)


@pytest.mark.parametrize("attn", VARIANTS)
def test_generate_shape(attn):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(attn))
    idx = torch.randint(0, 128, (2, 5))
    out = m.generate(idx, max_new_tokens=7, top_k=10)
    assert out.shape == (2, 12)


def test_kv_bytes_ordering():
    """At the headline 124M shape, KV cache should shrink MHA > GQA > MLA, MQA smallest."""
    cfgs = {a: GPTConfig(attn_type=a) for a in VARIANTS}
    b = {a: GPT(cfgs[a]).kv_bytes_per_token() for a in VARIANTS}
    assert b["mha"] > b["gqa"] > b["mqa"]
    assert b["mha"] > b["mla"]
    # default kv_lora_rank=256 vs MHA 2*768; MLA must be a large reduction
    assert b["mla"] < 0.4 * b["mha"]


def _mla_module(cfg):
    m = MultiHeadLatentAttention(cfg).double().eval()
    cos, sin = build_rope_cache(cfg.block_size, cfg.qk_rope_head_dim, cfg.rope_theta, dtype=torch.float64)
    return m, cos, sin


def test_mla_absorbed_equals_naive():
    """Weight-absorbed attention must equal the naive up-projected path (reassociation)."""
    torch.manual_seed(0)
    cfg = tiny_cfg("mla")
    m, cos, sin = _mla_module(cfg)
    x = torch.randn(2, 16, cfg.d_model, dtype=torch.float64)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        naive, _ = m(x, cos, sin, absorbed=False)
        absorbed, _ = m(x, cos, sin, absorbed=True)
    assert torch.allclose(naive, absorbed, atol=1e-9), (naive - absorbed).abs().max().item()


def test_mla_absorbed_incremental_matches_naive():
    """Absorbed incremental decode (the benchmark path) must match a naive full forward."""
    torch.manual_seed(0)
    cfg = tiny_cfg("mla")
    m, cos, sin = _mla_module(cfg)
    x = torch.randn(2, 12, cfg.d_model, dtype=torch.float64)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        full, _ = m(x, cos, sin, absorbed=False)
        cache, outs = None, []
        for t in range(x.shape[1]):
            o, cache = m(x[:, t : t + 1], cos, sin, past_kv=cache, use_cache=True, absorbed=True)
            outs.append(o)
        incr = torch.cat(outs, dim=1)
    assert torch.allclose(full, incr, atol=1e-9), (full - incr).abs().max().item()


def test_param_counts_124m():
    counts = {a: GPT(GPTConfig(attn_type=a)).num_params() for a in VARIANTS}
    for a, n in counts.items():
        assert 110e6 < n < 130e6, f"{a} has {n/1e6:.1f}M params, outside 124M target band"

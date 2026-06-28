"""Pluggable causal self-attention variants.

All variants share the same RoPE positional encoding and the same I/O contract::

    out, new_cache = attn(x, cos, sin, past_kv=None, use_cache=False)

so the GPT block is identical regardless of attention type, and the attention
mechanism is the only independent variable in the study.

Variants
--------
* MHA  : standard multi-head attention (n_kv_head == n_head).
* MQA  : single shared key/value head (n_kv_head == 1).
* GQA  : grouped key/value heads (n_kv_head groups), RoPE on full head_dim.
* MLA  : DeepSeek-V2 Multi-head Latent Attention -- low-rank joint KV
         compression (the latent is what gets cached) plus a decoupled RoPE key.

KV-cache element counts per token (the inference-memory story):
    MHA: 2 * n_head    * head_dim
    GQA: 2 * n_kv_head * head_dim
    MQA: 2 * 1         * head_dim
    MLA: kv_lora_rank + qk_rope_head_dim
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .layers import RMSNorm
from .rope import apply_rope

# A KV cache entry is a tuple of tensors whose exact shapes depend on the variant.
KVCache = tuple[torch.Tensor, torch.Tensor]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, hd) -> (B, n_kv * n_rep, T, hd) by repeating each KV head."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


def _causal_mask(q_pos: torch.Tensor, k_len: int, device) -> torch.Tensor:
    """Boolean keep-mask (True = attend) of shape (T_q, k_len) for cached decode."""
    k_pos = torch.arange(k_len, device=device)
    return k_pos[None, :] <= q_pos[:, None]


class GroupedQueryAttention(nn.Module):
    """Unified MHA / MQA / GQA. Set n_kv_head = n_head (MHA), 1 (MQA), or g (GQA)."""

    def __init__(self, config: GPTConfig, n_kv_head: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        d = config.d_model
        self.q_proj = nn.Linear(d, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, d, bias=False)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, absorbed=False):
        # `absorbed` is accepted for a uniform interface but has no effect here:
        # there is no latent to absorb in MHA/MQA/GQA.
        B, T, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].shape[2]
        positions = torch.arange(past_len, past_len + T, device=x.device)

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        # The cache stores the *unexpanded* KV heads (this is the memory win);
        # expansion to n_head happens transiently for the SDPA kernel.
        kx = repeat_kv(k, self.n_rep)
        vx = repeat_kv(v, self.n_rep)

        dp = self.dropout if self.training else 0.0
        if past_kv is None:
            out = F.scaled_dot_product_attention(q, kx, vx, is_causal=True, dropout_p=dp)
        else:
            mask = _causal_mask(positions, kx.shape[2], x.device)
            out = F.scaled_dot_product_attention(q, kx, vx, attn_mask=mask, dropout_p=dp)

        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.o_proj(out), new_cache

    def kv_bytes_per_token(self, dtype_size: int = 2) -> int:
        return 2 * self.n_kv_head * self.head_dim * dtype_size

    def init_cache(self, batch, length, device, dtype) -> KVCache:
        """Synthesize a populated KV cache of the given context length (for benchmarking)."""
        shape = (batch, self.n_kv_head, length, self.head_dim)
        return (torch.randn(shape, device=device, dtype=dtype), torch.randn(shape, device=device, dtype=dtype))


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2 MLA.

    KV is compressed to a latent ``c_kv`` of width ``kv_lora_rank``; a separate
    decoupled key ``k_rope`` of width ``qk_rope_head_dim`` carries RoPE and is
    shared across heads. Only (c_kv, k_rope) are cached. At each forward the
    latent is up-projected to per-head content keys and values; the query's
    per-head dim is split into a non-RoPE content part and a RoPE part.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.nope = config.qk_nope_head_dim
        self.rope = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora = config.kv_lora_rank
        self.q_lora = config.q_lora_rank
        self.qk_head_dim = self.nope + self.rope
        self.dropout = config.dropout
        d = config.d_model

        # Query path (optionally low-rank compressed to save activation memory).
        if self.q_lora > 0:
            self.q_down = nn.Linear(d, self.q_lora, bias=False)
            self.q_norm = RMSNorm(self.q_lora, eps=config.norm_eps)
            self.q_up = nn.Linear(self.q_lora, self.n_head * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(d, self.n_head * self.qk_head_dim, bias=False)

        # KV down-projection -> [compressed latent | decoupled rope key].
        self.kv_down = nn.Linear(d, self.kv_lora + self.rope, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora, eps=config.norm_eps)
        # Up-projection of the latent -> per-head [content key | value].
        self.kv_up = nn.Linear(self.kv_lora, self.n_head * (self.nope + self.v_head_dim), bias=False)

        self.o_proj = nn.Linear(self.n_head * self.v_head_dim, d, bias=False)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, absorbed=False):
        B, T, _ = x.shape
        past_len = 0 if past_kv is None else past_kv[0].shape[1]
        positions = torch.arange(past_len, past_len + T, device=x.device)

        # ---- queries ----
        if self.q_lora > 0:
            q = self.q_up(self.q_norm(self.q_down(x)))
        else:
            q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin, positions)

        # ---- KV down-projection ----
        kv = self.kv_down(x)
        c_kv, k_rope = kv.split([self.kv_lora, self.rope], dim=-1)
        c_kv = self.kv_norm(c_kv)                                  # (B, T, kv_lora)  [cached]
        k_rope = k_rope.view(B, T, 1, self.rope).transpose(1, 2)   # (B, 1, T, rope)
        k_rope = apply_rope(k_rope, cos, sin, positions)           # [cached, rope applied]

        # ---- cache (store compressed latent + decoupled rope key) ----
        if past_kv is not None:
            past_ckv, past_krope = past_kv
            c_kv = torch.cat([past_ckv, c_kv], dim=1)
            k_rope = torch.cat([past_krope, k_rope], dim=2)
        new_cache = (c_kv, k_rope) if use_cache else None

        S = c_kv.shape[1]

        if absorbed:
            out = self._attend_absorbed(q_nope, q_rope, c_kv, k_rope, positions, S)
        else:
            out = self._attend_naive(q_nope, q_rope, c_kv, k_rope, positions, S, past_kv is None)
        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.v_head_dim)
        return self.o_proj(out), new_cache

    def _attend_naive(self, q_nope, q_rope, c_kv, k_rope, positions, S, is_prefill):
        """Reference path: up-project the latent to per-head K/V, then SDPA."""
        B = c_kv.shape[0]
        kv_up = self.kv_up(c_kv).view(B, S, self.n_head, self.nope + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv_up.split([self.nope, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(B, self.n_head, S, self.rope)], dim=-1)
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        dp = self.dropout if self.training else 0.0
        if is_prefill:
            return F.scaled_dot_product_attention(q_full, k, v, is_causal=True, dropout_p=dp)
        mask = _causal_mask(positions, S, q_nope.device)
        return F.scaled_dot_product_attention(q_full, k, v, attn_mask=mask, dropout_p=dp)

    def _attend_absorbed(self, q_nope, q_rope, c_kv, k_rope, positions, S):
        """Weight-absorbed path: attend directly against the cached latent.

        Mathematically identical to the naive path, but never materializes the
        per-head content keys/values over the full context -- this is MLA's true
        inference-time win (memory bandwidth + activation footprint).
        """
        w = self.kv_up.weight.view(self.n_head, self.nope + self.v_head_dim, self.kv_lora)
        w_uk = w[:, : self.nope, :]                       # (H, nope, r)
        w_uv = w[:, self.nope :, :]                       # (H, v, r)
        q_abs = torch.einsum("bhtn,hnr->bhtr", q_nope, w_uk)              # (B,H,T,r)
        s_nope = torch.einsum("bhtr,bsr->bhts", q_abs, c_kv)             # (B,H,T,S)
        s_rope = torch.einsum("bhtr,bsr->bhts", q_rope, k_rope.squeeze(1))
        scores = (s_nope + s_rope) / math.sqrt(self.qk_head_dim)
        mask = _causal_mask(positions, S, c_kv.device)                   # (T,S) keep-mask
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhts,bsr->bhtr", attn, c_kv)                # (B,H,T,r)
        return torch.einsum("bhtr,hvr->bhtv", ctx, w_uv)                # (B,H,T,v)

    def kv_bytes_per_token(self, dtype_size: int = 2) -> int:
        return (self.kv_lora + self.rope) * dtype_size

    def init_cache(self, batch, length, device, dtype) -> KVCache:
        """Synthesize a populated latent KV cache of the given length (for benchmarking)."""
        c_kv = torch.randn(batch, length, self.kv_lora, device=device, dtype=dtype)
        k_rope = torch.randn(batch, 1, length, self.rope, device=device, dtype=dtype)
        return (c_kv, k_rope)


def build_attention(config: GPTConfig) -> nn.Module:
    config.validate()
    if config.attn_type == "mha":
        return GroupedQueryAttention(config, n_kv_head=config.n_head)
    if config.attn_type == "mqa":
        return GroupedQueryAttention(config, n_kv_head=1)
    if config.attn_type == "gqa":
        return GroupedQueryAttention(config, n_kv_head=config.n_kv_head)
    if config.attn_type == "mla":
        return MultiHeadLatentAttention(config)
    raise ValueError(f"unknown attn_type: {config.attn_type}")

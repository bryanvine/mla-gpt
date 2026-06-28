"""Decoder-only GPT backbone with a pluggable attention mechanism.

Everything here is shared across the MHA/MQA/GQA/MLA variants; the only thing
that changes is which attention module ``build_attention`` returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention
from .config import GPTConfig
from .layers import RMSNorm, SwiGLU
from .rope import build_rope_cache


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = build_attention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden_mult, config.dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, absorbed=False):
        attn_out, new_cache = self.attn(self.norm1(x), cos, sin, past_kv, use_cache, absorbed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE cache: dim depends on variant (full head_dim, or the decoupled
        # rope sub-dim for MLA). One cache serves every block of this model.
        rope_dim = config.qk_rope_head_dim if config.attn_type == "mla" else config.head_dim
        cos, sin = build_rope_cache(config.block_size, rope_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual output projections (GPT-2 / nanoGPT trick).
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x, _ = block(x, self.cos, self.sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss
        # inference: only the last position is needed
        logits = self.lm_head(x[:, -1:, :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation using per-layer KV caches."""
        self.eval()
        caches = [None] * len(self.blocks)
        cur = idx
        for _ in range(max_new_tokens):
            assert idx.shape[1] < self.config.block_size
            x = self.drop(self.tok_emb(cur))
            new_caches = []
            for i, block in enumerate(self.blocks):
                x, c = block(x, self.cos, self.sin, past_kv=caches[i], use_cache=True)
                new_caches.append(c)
            caches = new_caches
            logits = self.lm_head(self.norm_f(x[:, -1:, :])).squeeze(1) / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            cur = nxt  # subsequent steps feed only the new token; cache holds the rest
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=use_fused)

    def kv_bytes_per_token(self, dtype_size: int = 2) -> int:
        """Total KV-cache bytes per token across all layers (the inference-memory metric)."""
        return self.config.n_layer * self.blocks[0].attn.kv_bytes_per_token(dtype_size)

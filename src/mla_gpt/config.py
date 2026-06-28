"""Model configuration.

A single GPTConfig drives every attention variant. The backbone (depth, width,
MLP, normalization, RoPE) is identical across variants; only `attn_type` and the
variant-specific fields below change. This keeps the attention mechanism the
sole independent variable in the study.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

AttnType = Literal["mha", "mqa", "gqa", "mla"]


@dataclass
class GPTConfig:
    # --- vocab / sequence ---
    vocab_size: int = 50304          # GPT-2 BPE (50257) padded to a multiple of 64
    block_size: int = 1024           # max context length

    # --- backbone (held constant across variants) ---
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    mlp_hidden_mult: float = 8 / 3   # SwiGLU expansion (Llama-style); ~4x equiv params
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    # --- attention selection ---
    attn_type: AttnType = "mha"

    # GQA: number of key/value heads (must divide n_head). MQA is gqa with 1.
    n_kv_head: int = 4

    # --- MLA (DeepSeek-V2 style) ---
    # KV is jointly compressed to a latent of width kv_lora_rank (this is what is
    # cached). Queries optionally compressed to q_lora_rank (0 disables -> direct
    # projection). The per-head query/key dim is split into a non-RoPE "content"
    # part and a decoupled RoPE part; their sum defaults to head_dim so MLA's
    # per-head compute matches MHA in the headline config.
    kv_lora_rank: int = 256
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 48
    qk_rope_head_dim: int = 16
    v_head_dim: int = 64

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head

    def validate(self) -> None:
        assert self.attn_type in ("mha", "mqa", "gqa", "mla"), self.attn_type
        assert self.d_model % self.n_head == 0
        if self.attn_type == "gqa":
            assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        if self.attn_type == "mla":
            assert self.kv_lora_rank > 0
            assert self.qk_rope_head_dim > 0
            assert self.qk_rope_head_dim % 2 == 0, "RoPE dim must be even"

    def to_dict(self) -> dict:
        return asdict(self)

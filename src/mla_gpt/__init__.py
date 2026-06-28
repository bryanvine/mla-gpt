"""mla_gpt: a controlled study of attention KV-cache variants on a small GPT.

Variants: MHA (standard / FlashAttention), MQA, GQA, and MLA (Multi-head
Latent Attention, DeepSeek-V2 style). All share the same backbone, RoPE
positional encoding, and training recipe so the attention mechanism is the
only independent variable.
"""

__version__ = "0.1.0"

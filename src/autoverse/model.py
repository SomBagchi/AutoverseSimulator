"""Transformer configuration and op-graph construction.

Day 0: :class:`TransformerConfig` dataclass + a reference ``LLAMA_1B`` constant.
Day 1 fills in :func:`build_op_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autoverse.ops import Op

#: Forward-pass mode. ``"prefill"`` processes seq_len tokens at once;
#: ``"decode"`` processes one token given a KV cache of length seq_len.
Mode = Literal["prefill", "decode"]


@dataclass(frozen=True)
class TransformerConfig:
    """Dimensional parameters of a decoder-only transformer.

    Attributes:
        name: Short identifier.
        d_model: Model (residual-stream) width.
        n_layers: Number of transformer blocks.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads (equal to ``n_heads`` for vanilla MHA; fewer
            for grouped-query attention).
        d_head: Per-head embedding dimension. ``d_model == n_heads * d_head``.
        d_ffn: MLP inner dimension (the Gate/Up width; the Down output is ``d_model``).
        vocab_size: Vocabulary size (embedding + LM head).
        dtype: Activation/weight dtype for the simulated run.
    """

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ffn: int
    vocab_size: int
    dtype: str = "bf16"


# Reference configuration. Calibration and all Tier-1 validation runs target this.
LLAMA_1B = TransformerConfig(
    name="llama-3.2-1b",
    d_model=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,  # GQA: 4 Q-heads per KV-head
    d_head=64,
    d_ffn=8192,
    vocab_size=128256,
    dtype="bf16",
)


def build_op_graph(cfg: TransformerConfig, seq_len: int, mode: Mode) -> list[Op]:
    """Build a flat op graph for one forward pass of ``cfg`` at the given mode.

    Day 1 implements this. The graph order reflects execution order — the simulator
    consumes it sequentially.
    """
    raise NotImplementedError("Implemented at Tier 0 (Day 1). See CLAUDE.md checkpoint 1B.")

"""Operator representations — shape + FLOP/byte accounting.

Each concrete op (to be added at Tier 0 / Day 1) is a frozen dataclass that knows
its own FLOP count and the bytes it reads and writes. The analytical cost model
(:mod:`autoverse.cost`) uses these to produce latency estimates.

Day 0 provides the :class:`Op` protocol + dtype utilities; Day 1 fills in concrete
op types (``MatMul``, ``Attention``, ``RMSNorm``, …).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

#: Byte widths per supported dtype. Add entries here when a new precision is modeled.
DTYPE_BYTES: dict[str, int] = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp8": 1,
}


@runtime_checkable
class Op(Protocol):
    """An operator contributes FLOPs and bytes-moved to a workload.

    Concrete ops are frozen dataclasses; see Day-1 commits for the set implemented
    at Tier 0.
    """

    name: str
    dtype: str

    def flops(self) -> int:
        """Total multiply-add-equivalent FLOPs for this op."""
        ...

    def bytes_read(self) -> int:
        """Bytes read from HBM (before accounting for L2 hits)."""
        ...

    def bytes_written(self) -> int:
        """Bytes written to HBM."""
        ...


def dtype_bytes(dtype: str) -> int:
    """Return the byte width of a named dtype. Raises :class:`KeyError` if unknown."""
    return DTYPE_BYTES[dtype]

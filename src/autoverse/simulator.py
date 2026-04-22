"""Top-level simulator: compose per-op timings into workload latency."""

from __future__ import annotations

from dataclasses import dataclass, field

from autoverse.cost import OpTiming
from autoverse.hardware import HardwareSpec
from autoverse.ops import Op


@dataclass(frozen=True)
class SimResult:
    """Outcome of simulating a workload.

    Attributes:
        total_ms: End-to-end latency estimate (sum of per-op ``effective_ms``).
        per_op: List of ``(op_name, OpTiming)`` in execution order. Useful for
            per-op breakdown tables and timeline plots.
    """

    total_ms: float
    per_op: list[tuple[str, OpTiming]] = field(default_factory=list)


def simulate(ops: list[Op], spec: HardwareSpec) -> SimResult:
    """Simulate the workload defined by ``ops`` on ``spec``.

    Tier 0 implementation: iterate ops sequentially, estimate each, sum the
    effective times. No overlap across ops yet — that is a Tier-2 refinement.
    """
    raise NotImplementedError("Implemented at Tier 0 (Day 1). See CLAUDE.md checkpoint 1D.")

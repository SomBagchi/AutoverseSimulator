"""Top-level simulator: compose per-op timings into workload latency."""

from __future__ import annotations

from dataclasses import dataclass, field

from autoverse.cost import OpTiming, estimate
from autoverse.hardware import HardwareSpec
from autoverse.ops import Op


@dataclass(frozen=True)
class SimResult:
    """Outcome of simulating a workload.

    Attributes:
        total_ms: End-to-end latency estimate (sum of per-op ``effective_ms``).
        per_op: ``(op_name, OpTiming)`` pairs in execution order. Drives per-op
            breakdown tables and timeline plots in later tiers.
    """

    total_ms: float
    per_op: list[tuple[str, OpTiming]] = field(default_factory=list)


def simulate(ops: list[Op], spec: HardwareSpec) -> SimResult:
    """Simulate the workload defined by ``ops`` on ``spec``.

    Tier 0: strictly sequential — no cross-op overlap. Each op's effective time
    is added to a running total. Inter-op overlap (kernel dispatch pipelining)
    is a Tier-2 refinement if we have budget.
    """
    per_op: list[tuple[str, OpTiming]] = []
    total_ms = 0.0
    for op in ops:
        timing = estimate(op, spec)
        per_op.append((op.name, timing))
        total_ms += timing.effective_ms
    return SimResult(total_ms=total_ms, per_op=per_op)

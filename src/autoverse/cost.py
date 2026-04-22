"""Per-op analytical cost model.

Tier 0 (Day 1): pure roofline — ``t_effective = max(flops/peak_compute, bytes/peak_bw)``.
Tier 2 (Day 3) refines with wave quantization, compute-memory overlap, and L2-hit modeling.

See ``../03_autoverse_end_product.md`` §8 for the pinned modeling decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoverse.hardware import HardwareSpec
from autoverse.ops import Op


@dataclass(frozen=True)
class OpTiming:
    """Result of costing a single op.

    Attributes:
        compute_ms: Time if purely compute-bound (``flops / peak_compute``).
        memory_ms: Time if purely memory-bound (``bytes / peak_bw``).
        effective_ms: Combined model output. Tier 0: ``max(compute_ms, memory_ms)``.
            Later tiers add overlap: ``max(...) + (1 - alpha) * min(...)``.
    """

    compute_ms: float
    memory_ms: float
    effective_ms: float


def estimate(op: Op, spec: HardwareSpec) -> OpTiming:
    """Estimate the effective latency of ``op`` on ``spec``.

    Day 1 implements Tier-0 pure roofline. Day 3 refines with wave quantization
    (for small GEMMs), compute-memory overlap, and L2 hit-rate effects.
    """
    raise NotImplementedError("Implemented at Tier 0 (Day 1). See CLAUDE.md checkpoint 1C.")

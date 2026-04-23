"""Per-op analytical cost model.

Tier 0 (Day 1): pure roofline — the effective time of an op is
``max(flops / peak_compute, bytes / peak_bw)``, with an additive per-op launch
overhead. The ``max`` encodes **perfect compute-memory overlap within a single
op** — ALU and HBM run fully in parallel; the op finishes when the slower
phase finishes. This is a lower bound on real per-op latency; reality sits
above it because overlap is imperfect, nominal peaks are unachievable, and
small kernels under-utilise the machine.

Tier 2 (Day 3) will refine with:
  - wave quantisation for small GEMMs,
  - a calibratable overlap model ``t = max(t_c, t_m) + (1 - alpha) * min(t_c, t_m)``
    (alpha=1 recovers Tier 0 / perfect overlap; alpha=0 is strict serialisation),
  - an L2 hit-rate heuristic.

See ``../03_autoverse_end_product.md`` §8 for the pinned modelling decisions.
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
        effective_ms: Combined roofline output. Tier 0: ``max(t_c, t_m)`` plus
            per-op launch overhead. Later tiers add overlap.
    """

    compute_ms: float
    memory_ms: float
    effective_ms: float


def _peak_tflops_for_dtype(spec: HardwareSpec, dtype: str) -> float:
    """Pick the relevant peak compute throughput for the op's dtype.

    Tier 0 is coarse: BF16/FP16 use the BF16 tensor-core peak; FP32 uses the
    non-tensor-core peak; other dtypes (FP8, future precisions) fall back to
    the BF16 peak and get refined later.
    """
    if dtype in ("bf16", "fp16"):
        return spec.peak_bf16_tflops
    if dtype == "fp32":
        return spec.peak_fp32_tflops
    return spec.peak_bf16_tflops


def estimate(op: Op, spec: HardwareSpec) -> OpTiming:
    """Estimate the effective latency of ``op`` on ``spec``.

    Pure roofline with a launch-overhead term:

        t_c = flops / peak_compute
        t_m = (bytes_read + bytes_written) / peak_bandwidth
        t_eff = max(t_c, t_m) + per_op_overhead

    Returns times in **milliseconds**. The overhead field starts at zero and is
    populated by Tier-1 calibration.
    """
    peak_flops_per_s = _peak_tflops_for_dtype(spec, op.dtype) * 1e12
    peak_bytes_per_s = spec.hbm_gbps * 1e9
    bytes_moved = op.bytes_read() + op.bytes_written()

    compute_s = op.flops() / peak_flops_per_s if peak_flops_per_s > 0 else 0.0
    memory_s = bytes_moved / peak_bytes_per_s if peak_bytes_per_s > 0 else 0.0

    compute_ms = compute_s * 1e3
    memory_ms = memory_s * 1e3
    effective_ms = max(compute_ms, memory_ms) + spec.per_op_overhead_us / 1e3

    return OpTiming(
        compute_ms=compute_ms,
        memory_ms=memory_ms,
        effective_ms=effective_ms,
    )

"""Hardware specifications for accelerators we simulate.

A :class:`HardwareSpec` captures the parameters an analytical (roofline++) cost model
needs: effective compute throughput, memory bandwidth, memory-hierarchy sizes, and a
handful of calibrated overhead parameters populated by :mod:`autoverse.calibrate`.

All "effective" numbers are targets for calibration. The nominal peaks in the shipped
spec constants below are starting points from vendor specsheets; real workloads rarely
hit peak, and the calibration step (Tier 1) adjusts them to match measured behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    """An accelerator's roofline-level capabilities.

    Attributes:
        name: Human-readable identifier (e.g., ``"H100-SXM"``).
        peak_bf16_tflops: Effective BF16 tensor-core throughput in TFLOP/s.
            Calibration typically reduces this from the vendor nominal peak.
        peak_fp32_tflops: Effective FP32 (non-tensor-core) throughput in TFLOP/s.
        n_sm: Streaming-multiprocessor count. Used for wave-quantization modeling
            in Tier 2.
        hbm_gbps: Effective HBM bandwidth in GB/s.
        hbm_capacity_gb: HBM capacity in GB.
        l2_mb: L2 cache capacity in MB.
        l2_gbps: Effective L2 bandwidth in GB/s (rough; refined on calibration).
        smem_kb_per_sm: SM-private shared-memory capacity per SM, in KB.
        sm_clock_ghz: Nominal SM clock in GHz.
        per_op_overhead_us: Calibrated constant per-op overhead (kernel launch +
            dispatch) in microseconds. Zero until calibrated.
    """

    name: str
    peak_bf16_tflops: float
    peak_fp32_tflops: float
    n_sm: int
    hbm_gbps: float
    hbm_capacity_gb: int
    l2_mb: int
    l2_gbps: float
    smem_kb_per_sm: int
    sm_clock_ghz: float
    per_op_overhead_us: float = 0.0


# Starting-guess spec for H100-SXM. Peak numbers are vendor nominal; calibration will
# refine them to effective throughputs observed on real workloads.
H100_SXM = HardwareSpec(
    name="H100-SXM",
    peak_bf16_tflops=989.0,
    peak_fp32_tflops=67.0,
    n_sm=132,
    hbm_gbps=3350.0,
    hbm_capacity_gb=80,
    l2_mb=50,
    l2_gbps=12000.0,  # rough estimate; refined during calibration
    smem_kb_per_sm=228,
    sm_clock_ghz=1.98,
)

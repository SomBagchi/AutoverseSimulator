"""Calibration: fit :class:`HardwareSpec` parameters to real measurements.

Tier 1 (Day 2). Given (op, measured_latency_ms) pairs, fit effective peak TFLOPs,
effective HBM bandwidth, and a per-op overhead constant so that the sim matches
measurements as closely as possible in log-latency. Uses SciPy ``least_squares``.
"""

from __future__ import annotations


def calibrate() -> None:
    """Placeholder. Implemented at Tier 1 (Day 2)."""
    raise NotImplementedError("Implemented at Tier 1 (Day 2). See CLAUDE.md checkpoint 2C.")

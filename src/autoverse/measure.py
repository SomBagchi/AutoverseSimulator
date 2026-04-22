"""Real-hardware measurement harness.

Requires ``torch`` (optional dependency, install with ``uv sync --extra measure``).
Tier 1 (Day 2): timing primitives for matmul, attention, RMSNorm, etc., using
``torch.cuda.Event``-based measurement with warm-up and repeat counts.

This module is intentionally quarantined — the rest of the library is CPU-only
and never imports ``torch``, so CI stays fast.
"""

from __future__ import annotations


def measure() -> None:
    """Placeholder. Implemented at Tier 1 (Day 2)."""
    raise NotImplementedError("Implemented at Tier 1 (Day 2). See CLAUDE.md checkpoint 2A.")

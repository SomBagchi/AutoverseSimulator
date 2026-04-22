"""Autoverse — analytical performance model for transformer inference on GPU-like accelerators."""

from autoverse.hardware import H100_SXM, HardwareSpec
from autoverse.model import LLAMA_1B, TransformerConfig

__version__ = "0.1.0"
__all__ = [
    "H100_SXM",
    "LLAMA_1B",
    "HardwareSpec",
    "TransformerConfig",
    "__version__",
]

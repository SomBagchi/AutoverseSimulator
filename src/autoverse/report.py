"""Placeholder for in-package report-generation utilities.

Currently empty: the actual reports under ``reports/`` are produced by
``scripts/calibrate.py``, ``scripts/make_validation_plot.py``, and
``scripts/whatif.py``. This module is reserved for any future API that
needs to be importable from the library rather than invoked as a script.
"""

from __future__ import annotations


def report() -> None:
    """Reserved. Use ``scripts/`` for now."""
    raise NotImplementedError(
        "Use scripts/calibrate.py, scripts/make_validation_plot.py, "
        "or scripts/whatif.py — there is no in-package report API yet."
    )

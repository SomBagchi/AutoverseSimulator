"""Smoke tests — verify imports, the H100 spec, the Llama-1B config, and the CLI shell."""

from __future__ import annotations

import pytest


def test_import_package() -> None:
    import autoverse

    assert autoverse.__version__ == "0.1.0"


def test_h100_sxm_spec_shape() -> None:
    from autoverse import H100_SXM

    assert H100_SXM.name == "H100-SXM"
    assert H100_SXM.n_sm == 132
    assert H100_SXM.hbm_gbps > 0
    assert H100_SXM.peak_bf16_tflops > 0
    assert H100_SXM.per_op_overhead_us == 0.0  # vendor-nominal default; calibration overrides


def test_llama_1b_config_consistency() -> None:
    from autoverse import LLAMA_1B

    assert LLAMA_1B.name == "llama-3.2-1b"
    assert LLAMA_1B.n_heads * LLAMA_1B.d_head == LLAMA_1B.d_model
    assert LLAMA_1B.n_kv_heads <= LLAMA_1B.n_heads
    assert LLAMA_1B.n_heads % LLAMA_1B.n_kv_heads == 0  # GQA head grouping


def test_cli_help_exits_clean() -> None:
    from autoverse.cli import main

    # --help invokes argparse's internal print-and-exit path.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_simulate_prints_total_latency(capsys: pytest.CaptureFixture[str]) -> None:
    from autoverse.cli import main

    rc = main(["simulate", "--model", "llama1b", "--mode", "decode", "--seq-len", "1024"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "simulate" in out
    assert "llama1b" in out
    assert "total latency" in out
    assert "bare roofline" in out


def test_cli_simulate_breakdown_flag(capsys: pytest.CaptureFixture[str]) -> None:
    from autoverse.cli import main

    rc = main(
        ["simulate", "--model", "llama1b", "--mode", "prefill", "--seq-len", "512", "--breakdown"]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "per-op-family breakdown" in out
    # The top contenders should show up in the breakdown.
    assert "mlp_down" in out or "mlp_up" in out or "attn_prefill" in out

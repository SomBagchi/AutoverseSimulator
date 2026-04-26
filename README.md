# Autoverse Simulator

> An analytical performance model for transformer inference on GPU-like accelerators.
>
> Named after the *Autoverse* in Greg Egan's **Permutation City** — a simulation whose value is in the counterfactual questions you can ask of it, not how fast it runs.

## What this is

Autoverse takes two inputs:

1. A **hardware spec** — SM count, tensor-core TFLOPs, HBM bandwidth, L2 capacity, etc.
2. A **transformer configuration** — d_model, n_layers, n_heads, etc.

…and predicts per-op latency for prefill and decode. Predictions are calibrated against real H100 measurements on Llama-3.2-1B. Once validated, the model answers *"what if HBM bandwidth doubled?"* or *"what if L2 were bigger?"* — counterfactuals that would cost thousands of dollars of real-hardware time to answer by measurement.

## Status

**Tier 1 — validated against real H100.** The pipeline now fits three calibration
scalars (`peak_bf16_tflops`, `hbm_gbps`, `per_op_overhead_us`) to 530 measurements
collected on one NVIDIA H100 80GB HBM3, achieving **MAPE 20.2% on a 30% held-out
split** — under the Tier-1 ≤ 30 % target. Big ops (MatMul 9 %, AttentionPrefill 8 %)
modelled well; the remaining error concentrates on small overhead-dominated ops
(RoPE, Residual) and is queued for the per-op-family overhead refinement in Tier 2.

Full numbers, residual plot, and honest analysis of failure modes:
[`reports/01_validation.md`](./reports/01_validation.md).

| | fitted | vendor (H100-SXM) |
|---|---|---|
| `peak_bf16_tflops` | 1138.5 | 989 |
| `hbm_gbps` | 5374.8 | 3350 |
| `per_op_overhead_us` | 17.40 | — |

(Both throughputs land *above* vendor nominal — the optimiser is compensating for
L2-hit-rate effects in tight measurement loops and imperfect compute/memory
overlap. Both are named Tier-2 refinements; see the report.)

Sample output for one-token decode at `ctx_len=1024` on the **fitted** spec:

```
[autoverse] simulate: model=llama1b mode=decode seq_len=1024
  ops simulated: 227
  total latency: 0.749 ms  (uncalibrated Tier-0 roofline)
  per-op-family breakdown (effective_ms):
    mlp_gate        0.160 ms  ( 21.4%, 16 ops)
    mlp_up          0.160 ms  ( 21.4%, 16 ops)
    mlp_down        0.160 ms  ( 21.4%, 16 ops)
    lm_head         0.157 ms  ( 21.0%,  1 op)
    q_proj          0.040 ms  (  5.4%, 16 ops)
    out_proj        0.040 ms  (  5.4%, 16 ops)
    ...
```

See [`CLAUDE.md`](./CLAUDE.md) for the per-day roadmap and pinned modelling decisions.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
# Simulate Llama-1B decode — total latency only
uv run python -m autoverse simulate --model llama1b --mode decode --seq-len 1024

# …or with a per-op-family breakdown
uv run python -m autoverse simulate --model llama1b --mode prefill --seq-len 1024 --breakdown
```

Additional commands (from the Makefile):

```bash
make test       # pytest (66 tests, CPU-only)
make lint       # ruff + mypy
make calibrate  # Tier 1: refit (F, B, O) on the committed H100 JSON
make validate   # Tier 1: calibrate + regenerate residual plot
make measure    # Tier 1: collect a fresh H100 sweep (needs CUDA + uv sync --extra measure)
make whatif     # Tier 3+: run counterfactual experiments
```

To collect a fresh measurement on an H100 box:

```bash
uv sync --extra measure
MEASUREMENT_OUT=measurements/h100_sxm/run_$(date -u +%Y%m%d_%H%M%S).json make measure
```

Then update `MEASUREMENTS = ...` at the top of the `Makefile` and rerun
`make validate`.

## Why this project

Built as a code sample for the [Anthropic Fellows program](https://job-boards.greenhouse.io/anthropic/jobs/5023394008) — specifically the ML Systems & Performance track, which lists "CPU simulators for accelerator workloads" as an example project shape.

## License

MIT — see [`LICENSE`](./LICENSE).

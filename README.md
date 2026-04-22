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

**Tier 0 — end-to-end skeleton.** The pipeline (op-graph → roofline → total latency)
runs end-to-end for Llama-3.2-1B on a nominal H100-SXM spec. Numbers are **uncalibrated**;
use them for ratios and shape, not absolute accuracy. Tier 1 (calibration against
real H100 measurements) lands next.

Sample output for one-token decode at `ctx_len=1024` (nominal H100 peak):

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
make test       # pytest
make lint       # ruff + mypy
make validate   # Tier 1+: calibrate against real measurements, generate report
make whatif     # Tier 3+: run counterfactual experiments
```

## Why this project

Built as a code sample for the [Anthropic Fellows program](https://job-boards.greenhouse.io/anthropic/jobs/5023394008) — specifically the ML Systems & Performance track, which lists "CPU simulators for accelerator workloads" as an example project shape.

## License

MIT — see [`LICENSE`](./LICENSE).

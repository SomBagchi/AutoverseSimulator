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

**Day 0 — scaffolding.** Interfaces drafted; CI green; measurement & calibration TBD.

See [`CLAUDE.md`](./CLAUDE.md) for the per-day roadmap.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
# (Tier 0+) Simulate Llama-1B decode — prints a per-op breakdown
uv run python -m autoverse simulate --model llama1b --mode decode --seq-len 1024
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

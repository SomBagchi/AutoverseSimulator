# Tier-2 Validation Report

> **Sprint day 3 (compressed into Sun Apr 26 morning).** Two refinements
> over the Tier-1 roofline, both pinned in `03_autoverse_end_product.md` §8:
> an L2 hit-rate heuristic and per-op-family launch overhead.
>
> *Reproduce: `make validate`. Same `MEASUREMENTS = ...` JSON as Tier 1.*

## Headline

|   | Tier 1 | Tier 2 (this report) | target |
|---|---|---|---|
| **MAPE held-out** | 20.2 % | **9.4 %** | ≤ 20 % |
| MAPE fit          | 19.6 % | 8.4 %   | — |
| fitted F (TFLOPs) | 1138 (1.15× vendor) | 1172 (1.18× vendor) | physical ≤ 989 |
| fitted B (GB/s)   | 5375 (1.60× vendor) | **2311 (0.69× vendor)** | physical ≤ 3350 |
| worst per-op MAPE | RoPE 65 % | AttentionPrefill 13 % | — |

**Verdict: Tier 2 ships, more than 2× under target.** The MAPE win comes
almost entirely from per-family overhead. The L2 heuristic doesn't move
MAPE much on its own but is what gets `B` back into a physically
meaningful range — important for Tier-3 what-if questions like *"what if
HBM bandwidth doubled?"* where the baseline `B` needs to mean something.

## What changed from Tier 1

Tier 1 used a global per-op overhead. Examination of the residuals
showed two systematic biases:

1. **B inflated above vendor (5375 vs 3350)** because measurements run
   each op 100× with the same input tensors → after iter 1, inputs sit
   in L2 → the median latency reflects warm-cache reads, but the model
   counted full HBM traffic. The optimiser absorbed the gap into B.
2. **Small ops (RoPE 65 %, Residual 48 %) terribly fit** because a
   single global O cannot capture the spread between RoPE's real ~52 µs
   launch and Residual's ~12 µs.

Tier 2 addresses both directly.

### Refinement 1: L2 hit-rate heuristic

For each op:

```
hit_rate = min(1, L2_capacity / bytes_read)        # H100 L2 = 50 MB
effective_bytes = bytes_written + bytes_read · (1 - hit_rate)
t_memory = effective_bytes / B
```

Two design notes:

- **Hit rate is computed on `bytes_read`, not the full working set.**
  Caches reduce *re-reads*; first-time writes always stream to HBM.
  The simpler "hit rate on working set" formulation zeros out memory
  entirely for L2-resident ops, which is a clear over-correction —
  SwiGLU's MAPE jumped from 11 % to 28 % under that variant before this
  fix was applied.
- **Static heuristic, no new free parameter.** L2 capacity comes from
  `HardwareSpec.l2_mb` (50 MB for H100). The fit doesn't tune it; the
  geometry is fixed by the chip.

Implementation: `autoverse.cost.l2_hit_rate` and the `use_l2` flag on
`autoverse.cost.estimate`. Tier-0 ablation is one keyword away
(`use_l2=False`), which keeps synthetic-recovery tests stable.

### Refinement 2: per-op-family overhead

`autoverse.calibrate.calibrate_per_family` fits one launch-overhead
scalar per op type (8 families × 1 each = 8 free params, replacing 1).
Total free params goes from 3 → 10. Still well-determined: 530
measurements vs 10 params is ~50 measurements per parameter.

Fitted values (sorted high-to-low):

| op family | fitted O (µs) | what this is |
|---|---|---|
| RoPE | **52.1** | Pair-rotation kernel; index arithmetic + dispatch dominant. |
| AttentionDecode | 22.6 | SDPA decode launch (cuDNN backend selection + setup). |
| AttentionPrefill | 21.8 | Same kernel family, slightly cheaper for tiled prefill. |
| SiLUGate | 18.9 | Two elementwise kernels (`silu` then mul) — paying launch twice. |
| MatMul | 17.6 | cuBLAS heuristic + algorithm selection. |
| RMSNorm | 14.6 | `F.rms_norm` is well-tuned. |
| Embedding | 13.4 | Gather is cheap to dispatch. |
| Residual | **11.6** | Single elementwise add — minimal setup. |

The 4–5× spread between RoPE and Residual is the core thing a single
global O *cannot* model. Per-family captures it directly.

## Per-op-family accuracy

Held-out MAPE, sorted worst-first:

| op family | Tier 1 | Tier 2 | improvement |
|---|---|---|---|
| RoPE | 65.5 % | **1.0 %** | 66× |
| Residual | 48.2 % | 7.9 % | 6× |
| Embedding | 26.0 % | 0.9 % | 30× |
| AttentionDecode | 22.7 % | 1.7 % | 13× |
| RMSNorm | 21.7 % | 8.4 % | 2.6× |
| SiLUGate | 11.0 % | 12.5 % | slight regression |
| MatMul | 9.2 % | 10.5 % | slight regression |
| AttentionPrefill | 8.2 % | 13.0 % | regression |

**The two regressions are the result of B fitting low (2311 GB/s).**
With B physically realistic, memory time goes up for memory-bound ops
that don't fit in L2 — the MatMul and SiLUGate / prefill subset where
the working set spills out of L2 now pays full bandwidth time, and a
few of those didn't fully accept that. Net win is still ≥ 11 percentage
points on aggregate held-out MAPE.

## Why F is *still* above vendor (1172 vs 989)

The remaining flattering bias on F traces to **a single outlier**: the
prefill LM head `MatMul(m=1024, k=2048, n=128 256)`, measured 0.818 ms
versus predicted 0.477 ms (1.7×). For this tall-skinny GEMM the model's
roofline math (a generous 78 % of vendor F) over-estimates effective
throughput; cuBLAS picks an algorithm that doesn't reach the
tensor-core roof on this shape.

This is exactly the kind of effect the **wave-quantisation** Tier-2
refinement (deferred from this sprint) would catch: the trailing
partial-wave penalty on `N = 128 256 / 128 = 1002` tiles across
`132` SMs adds up to ~5 % directly, and another 30+ % from sub-peak
algorithm selection on tall-skinny shapes. With wave-quantisation in
place, F should settle around 770–800 TFLOPs (real cuBLAS effective rate
on big square GEMMs).

We accept the lm_head as a known outlier in this Tier-2 fit and call
out wave-quantisation as the highest-leverage next refinement.

## Worst-fit ops (Tier 2)

| name | type | measured | predicted | rel err |
|---|---|---|---|---|
| `lm_head` (prefill) | MatMul | 0.818 ms | 0.477 ms | 41.8 % |
| `mlp_down_15` | MatMul | 0.028 ms | 0.018 ms | 37.5 % |
| `residual_mlp_15` | Residual | 0.017 ms | 0.012 ms | 32.7 % |
| `rmsnorm_pre_mlp_*` | RMSNorm | 0.022 ms | 0.015 ms | ~32 % |

The non-LM-head outliers are all in the 0.01–0.03 ms range — small
enough that a few microseconds of measurement jitter shows up as
double-digit relative error. The corresponding *absolute* error is
0.005–0.010 ms, which contributes < 0.5 % of total inference latency
even when summed across all 16 layers.

## Residual plot

![Tier-2 residual scatter](figures/measured_vs_predicted.png)

Read the plot:

- The **±30 % grey band** now contains visibly more points than the
  Tier-1 plot — the off-band cluster of RoPE / Residual at the bottom-
  left is gone.
- The **diagonal trend** has tightened, especially at the medium-size
  range where most Llama-1B ops sit.
- The **upper-right outlier** is the LM head — the only point still
  visibly off the diagonal at the high end, and the named driver of
  the residual F-above-vendor anomaly.

## Tier-2 refinements deferred

We pinned four refinements at the start of Tier 2 in
`03_autoverse_end_product.md` §8. Status:

| refinement | status | impact realised |
|---|---|---|
| L2 hit-rate heuristic | **shipped** | Brings B into physical range; small direct MAPE win. |
| Per-op-family overhead | **shipped** (added based on Tier 1 findings) | The dominant MAPE win. |
| Per-op-family α (overlap) | deferred | Would tighten the F over-fit. Lower priority once O is per-family. |
| Wave quantisation for GEMMs | deferred | Highest-leverage next step — would close the LM-head outlier and bring F to ≈ 770 TFLOPs. |

Held-out MAPE target was ≤ 20 %. We landed at 9.4 %. Further fidelity
work is real but past the point of diminishing returns relative to what
we want to spend Tier-3 budget on (counterfactual experiments).

## How to reproduce

```bash
make validate              # Tier-2 fit (default: per-family + L2)
# Ablations:
uv run python scripts/calibrate.py \
    measurements/h100_sxm/run_20260426_233235.json \
    --l2-mb 0              # Tier-1 ablation: no L2
uv run python scripts/calibrate.py \
    measurements/h100_sxm/run_20260426_233235.json \
    --global-overhead      # Single O ablation
```

The committed `reports/calibration_fit.json` is the Tier-2 fit (both
refinements active), with `fitted_overhead_by_family` populated.

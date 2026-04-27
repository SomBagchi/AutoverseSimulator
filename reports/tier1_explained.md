# Tier 1 — what we did, and what the numbers mean

> Pedagogical companion to [`01_validation.md`](./01_validation.md). That
> file is the formal report; this one walks through what each number
> means and resolves a genuinely confusing result (fitted throughputs
> above vendor peaks).

## What were we trying to do?

The Tier-0 simulator already predicts a per-op latency given a
`HardwareSpec`. But every number in `HardwareSpec` is a *vendor nominal* —
it's what the chip would do under ideal conditions, not what cuBLAS and
the runtime actually deliver in practice. **Tier 1's job is to replace the
vendor numbers with measured-effective ones**, so that
`simulate(model, hw)` predicts something close to wall-clock reality.

The pipeline:

```
                                    ┌──────────────────────────────┐
        ┌─────────────────┐         │ scripts/collect_measurements │
        │ HardwareSpec    │         │   (runs on H100, 530 ops)    │
        │ (vendor peaks)  │         └──────────────┬───────────────┘
        └────────┬────────┘                        │ measured_ms per op
                 │                                 ▼
                 │                       ┌──────────────────┐
                 ├──────────────────────►│ calibrate.py     │
                 │                       │ (least-squares)  │
                 │                       └────────┬─────────┘
                 │                                │ fitted (F, B, O)
                 │                                ▼
                 │                       ┌──────────────────┐
                 └──────────────────────►│ HardwareSpec     │
                                         │ (calibrated)     │
                                         └──────────────────┘
```

So the deliverable is three numbers — **F**, **B**, **O** — and a
single accuracy headline (**MAPE**). Once those are good, every
downstream "what-if" experiment in Tier 3 inherits the calibration.

## What is MAPE?

**MAPE = Mean Absolute Percentage Error.** For each measurement `m_i`
and the model's prediction `p_i`:

```
            1     n   |p_i − m_i|
MAPE   =   ───   ∑   ─────────────
            n   i=1     m_i
```

Read in plain English: *"on average, how far off (as a percentage of the
true value) are the predictions?"*

- 0 % MAPE = the model nails every measurement exactly.
- 20 % MAPE = on average, predictions are off by 20 % of the measured value
  (in either direction).
- 100 %+ MAPE = the model is barely better than guessing.

Why MAPE rather than RMSE or MSE? Because latencies in this dataset span
**three orders of magnitude** (≈ 0.012 ms to ≈ 0.82 ms). An absolute-error
loss would let one big-op error dwarf hundreds of small-op errors. We
care about *relative* accuracy — getting a tiny op wrong by 0.01 ms is a
~50 % miss; getting the LM head wrong by 0.01 ms is < 1 %. MAPE encodes
that asymmetry directly.

We report two MAPEs:

- **MAPE on the fit set** (70 % of measurements). The optimiser has seen
  these — they're how it picked F, B, O. Low MAPE here just means the
  model can *fit* the data.
- **MAPE on the held-out set** (30 %). The optimiser has *not* seen these.
  This is the honest accuracy number — it shows how the model
  *generalises* to ops it wasn't trained on.

## What are F, B, and O?

Three scalars in our roofline. Predicted latency for op `i`:

```
                ┌  flops_i      bytes_i  ┐                 ms
   p_i  =  max  │ ─────────  ,  ──────── │   +   O · 10⁻³
                └  F · 10¹²    B · 10⁹  ┘
                  ──────────  ──────────
                  t_compute    t_memory
                  (seconds)    (seconds)
```

| symbol | unit | physical meaning |
|---|---|---|
| **F** | TFLOPs/s | Effective BF16 tensor-core throughput. Sets `t_compute = flops / F`. |
| **B** | GB/s | Effective HBM bandwidth. Sets `t_memory = bytes / B`. |
| **O** | µs | Per-op launch overhead — the cost of dispatching *any* kernel, no matter how trivial. Added once per op. |

For Llama-1B on H100 these would naively be the vendor numbers from
`HardwareSpec`: F = 989, B = 3350, O = 0. **Calibration replaces them
with whatever values make the model best match what the chip actually
does**, so that downstream simulations reflect deliverable performance,
not specsheet performance.

Why these three and not more? The roofline equation has *exactly* three
free knobs. Any other refinement (overlap, L2 hit rate, wave
quantisation) changes the *shape* of the equation, not its scalars —
those are Tier 2 work.

## What the optimiser actually does

Given 530 `(op, measured_ms)` pairs:

```python
loss(F, B, O) = Σ ( log p_i(F, B, O)  −  log m_i )²
```

SciPy's `least_squares` (Trust-Region Reflective with positivity bounds)
finds the (F, B, O) that minimises this loss. **Log-space** because we
care about relative error — `log(p) − log(m) = log(p/m)`, so the loss
penalises ratios, which matches MAPE's relative-error notion.

The split: 70 % of ops are used for the fit, 30 % held out. The split is
deterministic (seeded shuffle). Held-out MAPE is the headline.

## The headline result

```
  peak_bf16_tflops    : 1138.51   (vendor 989)
  hbm_gbps            : 5374.8    (vendor 3350)
  per_op_overhead_us  :   17.40   (—)

  MAPE  fit          : 19.61%   (n=371)
  MAPE  held-out     : 20.20%   (n=159)
```

**This is where the confusion starts.** F = 1138 > vendor 989, and
B = 5375 > vendor 3350. *You cannot run faster than the chip can run.*
So why does the fit say we are?

## Why fitted numbers exceed vendor — and why physics is fine

**Short answer: it's not the chip exceeding peak; it's our *model* over-
counting bytes, and the optimiser compensating.**

Long answer in four steps.

### Step 1: how the measurement actually works

When we measure the latency of an op, we don't do it once. We:

1. Allocate the input tensors **once** on the GPU.
2. Run the op **10 times** to warm up (compile, JIT, populate caches).
3. Run the op **100 more times** with `cudaEventRecord` brackets,
   capturing per-iter latency.
4. Take the **median**.

Crucially: we use the **same input tensors** for all 110 iters. We never
reallocate, never zero-out caches between iters.

### Step 2: where the data lives between iters

H100 has a **50 MB L2 cache**, which is *huge* by historical standards.
After iter 1, the input tensors are in L2 (or part of them are, if they
exceed 50 MB). Iter 2 doesn't have to fetch them from HBM again — the
loads are served from L2.

So:
- **Iter 1** pays the full HBM round-trip.
- **Iters 2–100** mostly hit L2 for inputs.

The **median** across all 100 iters is dominated by the warm-cache time,
not the cold-cache time. That's actually what we want for predicting
real-world inference latency *within* a layer or *within* a token, where
caches similarly stay warm. But it means our *measurements* reflect a
different cost than our *model* assumes.

### Step 3: where our model is over-counting

`Op.bytes_read()` and `Op.bytes_written()` assume **every byte goes to
HBM**, every time. For a `MatMul(M, K, N)` in BF16:

```python
bytes_read    = (M·K  +  K·N) · 2
bytes_written =  M·N · 2
```

But on iter 50, the activation `(M, K)` and weight `(K, N)` tensors are
sitting in L2. The actual HBM traffic on that iter is *much* smaller.

So our model claims "this op moves 30 MB from HBM" but the chip actually
moved maybe 5 MB. To make `t_memory = bytes / B = 30 MB / B` come out
equal to the small measured time, the optimiser must inflate `B`. The
math is forced.

### Step 4: empirical proof

If the L2-caching story is correct, we'd predict: **calibrating only on
ops too large to fit in L2 should give B ≤ vendor**, because there's no
cache effect for the optimiser to absorb.

Done. The numbers:

|                                | n   | fitted F (TFLOPs) | fitted B (GB/s) | held-out MAPE |
|---|---|---|---|---|
| **All 530 ops** (full fit)     | 530 | 1138 (1.15× vendor) | 5375 (1.60× vendor) | 20.2 % |
| **Subset: working set > L2**   |  54 |  797 (0.81× vendor) | 3111 (0.93× vendor) |  3.4 % |
| **Subset: working set < 5 MB** | 162 |  989 (= vendor)*    | 3350 (= vendor)*    | 25.4 % |

*\* On the 5 MB subset, the optimiser doesn't move from its initial
guess (989, 3350) because every op is overhead-dominated — neither F nor
B affects predicted time, so the gradient is zero. That row tells us
nothing about F or B; it tells us the small-op MAPE is bad because a
single global O can't capture per-op-family overhead variation.*

The middle row is the clincher:

- **F = 797 TFLOPs** is 81 % of vendor 989 — exactly the realistic cuBLAS
  efficiency we already saw on the 4096³ GEMM (78 %).
- **B = 3111 GB/s** is 93 % of vendor 3350 — typical effective HBM3
  performance with normal access patterns.
- **MAPE 3.4 %** says the roofline + global overhead nails these big ops
  to within a few percent.

So **the chip behaves exactly as physics requires** — 81 % of compute
peak, 93 % of bandwidth peak. The full-dataset fit's above-vendor
numbers are not the chip overachieving; they are the optimiser papering
over an L2-shaped hole in our model.

### Why we report the above-vendor fit anyway

We could just calibrate on the 54 large-only ops and quote MAPE 3.4 %.
But that would be a lie of omission: the model would still mispredict
real Llama-1B forward passes (which are dominated by *medium*-sized ops
that fit nicely in L2), because we'd have ignored that regime.

The honest summary is: **our current Tier-0 model has no concept of L2.
The full-dataset fit shows what happens when you ask three scalars to
do a job that needs four.** Tier 2 adds the fourth (an L2 hit-rate
heuristic), at which point the fitted F and B should drop back into
their physical ranges.

## Per-op MAPE breakdown

Held-out MAPE, sorted worst-first:

| op family | held-out MAPE | what's happening |
|---|---|---|
| RoPE | 65 % | Single global O = 17 µs, but RoPE's real launch cost is ≈ 50 µs. Per-family O fixes this. |
| Residual | 48 % | Same: real launch ≈ 12 µs, model says 17 µs — the global O is being pulled high by RoPE-like ops, hurting genuinely cheap ops. |
| Embedding | 26 % | Same overhead-fitting issue, plus the gather pattern is hard to model. |
| AttentionDecode | 23 % | Memory-bound; affected by L2 inflation. |
| RMSNorm | 22 % | Mostly memory-bound; small + frequent. |
| SiLUGate | 11 % | Bigger ops; cleaner fit. |
| **MatMul** | **9 %** | The bread-and-butter. Within target. |
| **AttentionPrefill** | **8 %** | Best-fit family. |

The big ops are nailed; the small ones aren't. **This is fine for now**:
the small ops contribute < 5 % of total Llama-1B latency, so even a
50 %-relative miss on them changes the total by < 2.5 %.

## What Tier 2 will fix

The Tier-1 measurements *confirm* the refinements pinned in
`03_autoverse_end_product.md` §8 are the right ones. Concretely:

1. **L2 hit-rate heuristic** — `hit_rate = min(1, L2_cap / working_set)`.
   Will pull fitted B back to ≈ 3350 GB/s and tighten MAPE on
   medium-sized ops (the ones currently inflated).
2. **Per-op-family α (compute / memory overlap)** — collapses the
   small over-fit on F.
3. **Per-op-family overhead** — collapses the RoPE/Residual MAPE column.
4. **Wave quantisation** — fixes the LM-head outlier (1.73× off in the
   current fit).

Held-out MAPE target after Tier 2: **≤ 20 %**.

## TL;DR

- We measured 530 ops on a real H100, fit three scalars (F, B, O),
  and got **20 % held-out MAPE** — under the ≤ 30 % Tier-1 target.
- The fitted F and B *appear* to exceed vendor peaks, but the chip is
  not breaking physics. Restricting the fit to ops that don't fit in L2
  recovers F = 797, B = 3111 — both *below* vendor — with MAPE 3.4 %.
- The above-vendor full-fit is the optimiser absorbing L2-caching
  effects into the throughput scalars. This is exactly the gap the
  Tier-2 L2 hit-rate heuristic closes.
- Big ops (MatMul 9 %, prefill attention 8 %) modelled well already.
  Small overhead-dominated ops (RoPE, Residual) are the remaining
  weak spot, queued for the per-op-family overhead refinement.

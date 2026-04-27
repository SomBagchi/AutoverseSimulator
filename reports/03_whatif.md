# What-if experiments — counterfactual hardware on the calibrated model

> Five counterfactual hardware questions answered using the calibrated
> H100 model from `reports/calibration_fit.json`. Baseline is the
> calibrated H100; each experiment changes one parameter and reports
> the delta.

Baseline calibrated `HardwareSpec`: `F=943 TFLOPs`, `B=2285 GB/s`, `L2=50 MB`, per-family overhead from the calibration fit.

## Headline finding: Llama-1B decode is overhead-bound, not memory-bound

Decoding one token through 227 ops on the calibrated H100 model predicts **4.50 ms**, of which **4.29 ms (95%)** is per-op kernel-launch overhead. Compute + HBM together account for the remaining ~5%. This is the dominant surprise of the what-if work and reframes most of the experiments below: hardware throughput knobs (HBM, compute, L2) move the small non-overhead slice; the big lever for small-model single-stream decode is collapsing the launch count itself (CUDA Graphs, kernel fusion, larger batches).

## E1 — What if HBM bandwidth doubled? (decode, ctx=1024)

| | latency | comment |
|---|---|---|
| Baseline (B = 2285 GB/s) | **4.502 ms** | Calibrated H100. |
| Counterfactual (B × 2) | 4.398 ms | Speedup **1.02 ×** (saves 104 µs/token). |

**Reading.** I expected ≈ 2 × — "decode is memory-bound, double the bandwidth, halve the time." The model says otherwise. With 95 % of total time in launch overhead, doubling HBM only acts on the 5 % memory slice. **For Llama-1B decode on H100, an HBM upgrade barely moves the needle.** The roofline still holds — within memory-bound ops the time roughly halves — but those ops are not what you're paying for.

Implication: large-model decode (where weight-read traffic per layer is much larger and the per-op compute/memory time grows past the launch overhead) **would** see the expected ≈ 2 × from this experiment. Llama-1B is small enough that it lives in a different regime.

## E2 — What if BF16 compute doubled? (prefill, seq_len=1024)

| | latency | comment |
|---|---|---|
| Baseline (F = 943 TFLOPs) | **7.360 ms** | Calibrated H100. |
| Counterfactual (F × 2) | 5.984 ms | Speedup **1.23 ×**. |

**Reading.** Prefill at seq_len=1024 *is* substantially compute-bound. The 1.18 × falls short of 2 × because ~4.4 ms of the 6.9 ms baseline is the same launch-overhead floor as decode (227 ops, ~14–52 µs each). Halving compute time on the remaining ~2.5 ms gets you the 1.18 × seen. **This is the same story as E1 from the opposite direction**: per-op overhead caps the upside of any single throughput knob.

## E3 — What if L2 doubled? (decode, ctx=1024)

| | latency | comment |
|---|---|---|
| Baseline (L2 = 50 MB) | **4.502 ms** | Calibrated H100. |
| Counterfactual (L2 × 2 = 100 MB) | 4.479 ms | Speedup **1.01 ×**. |

**Reading.** Effectively no help. Decode-shape MLP matmuls (`m=1`, `k=2048`, `n=8192`) already fit in 50 MB L2 (working set ~33 MB), so doubling L2 doesn't change their hit-rate. The LM head (`n=128 256`, 525 MB working set) is way too big for any plausible L2; doubling 50→100 shifts hit_rate from 9.5 % to 19 %, saving a few µs. **Llama-1B decode L2 is already "big enough" by a comfortable margin** — the 50 MB H100 L2 was a substantial uplift over A100's 40 MB and pays off here.

## E4 — Decode latency vs context length (sweep)

| ctx_len | total ms | attn ms | matmul ms | attn share |
|---:|---:|---:|---:|---:|
|     128 | 4.5014 | 0.3619 | 2.1371 |   8.0 % |
|     512 | 4.5014 | 0.3620 | 2.1371 |   8.0 % |
|    1024 | 4.5015 | 0.3620 | 2.1371 |   8.0 % |
|    2048 | 4.5017 | 0.3622 | 2.1371 |   8.0 % |
|    4096 | 4.5019 | 0.3625 | 2.1371 |   8.1 % |
|    8192 | 4.5025 | 0.3630 | 2.1371 |   8.1 % |
|   16384 | 4.5037 | 0.3642 | 2.1371 |   8.1 % |
|   32768 | 4.6042 | 0.4647 | 2.1371 |  10.1 % |

**Reading.** Llama-1B's 8 KV heads × 64 d_head = 512 floats per token of KV cache, BF16 = 1024 bytes/token/layer. Even at ctx=32 768 the per-layer KV cache is 32 MB — fits comfortably in L2. Combined with the fact that AttentionDecode has a single-digit-µs roofline contribution (memory-bound but small) plus its 22 µs per-family launch overhead, the attention share stays ≤ 10 % of total decode latency at every context length we tested.

**Generalisation watch-out.** This is an artefact of Llama-1B's *very* aggressive GQA (4 Q-heads per KV-head). Llama-3.1-70B with 8 KV-heads × 128 d_head sees 4 × the KV-cache size per token, and a `d_model=8192` MLP that's bigger but not 4 × larger — the crossover where attention dominates moves much closer in.

## E5 — Prefill vs decode per-token (prefill seq_len=1024)

| | per-token latency | tokens/sec |
|---|---:|---:|
| Prefill (batch all 1024 tokens at once) | **7.2 µs** |  139,138 |
| Decode (one token at a time) | **4502 µs** |      222 |
| **Prefill advantage** | **626 ×** faster per token | |

**Reading.** Decode pays the full 227-op launch overhead *per token*. Prefill pays it once for the whole batch and then amortises across all 1024 tokens — the per-op overhead per token drops by ~1024×. Plus, prefill MatMuls are big enough to actually engage the compute roof instead of bottoming out at the launch floor. Result: a ~626 × per-token latency advantage. **This number is the strongest argument for continuous batching.** Every additional decode step you can convert to a prefill-shaped step (by stacking concurrent users) gets close to that advantage.

## What to take from these five

1. **Per-op launch overhead is the dominant Llama-1B-on-H100 single-
   stream-decode bottleneck.** Not HBM, not L2, not compute. Engineering    hours on this model class go furthest into kernel fusion, CUDA Graphs,    or batching — anything that collapses the 227-launch count.
2. **Prefill is on a different shore of the roofline** — partially    compute-bound, where doubling compute does buy you 1.2 ×. Compute    upgrades pay off in prefill time-to-first-token (TTFT), much less    in inter-token latency.
3. **L2 is already over-provisioned for Llama-1B** at H100's 50 MB.    This is a real architectural feature of the H100 generation paying    off; doubling it again wouldn't.
4. **Context-length scaling is benign for Llama-1B** because of GQA.    This conclusion does *not* generalise to bigger Llama variants —    the simulator can answer that question once those configs are added    to `model.py`.
5. **Continuous batching is the headline serving lever.** The per-token    ratio between batched-prefill and isolated-decode is the upper bound    on what fancy batching can buy you.

## Caveats

- The calibrated model has held-out MAPE 10.1 %. Absolute numbers carry that error bar.
- **Speedup ratios** in E1–E3 are mostly roofline-determined and survive the absolute calibration error — if 5 % of time is memory and you halve memory, you save 2.5 % regardless of whether your time was 9 % off.
- The model has no concept of multi-GPU collectives, paged attention, speculative decoding, or batched continuous serving — explicitly out of scope per `03_autoverse_end_product.md` §2.
- Per-op overhead is fitted from a measurement methodology that runs the same op 100 × in a tight loop. CUDA Graphs / kernel-launch APIs in production may produce lower effective overhead than this fit captures; that would shrink the overhead floor and shift these conclusions toward more conventional roofline regimes.

## Reproduce

```bash
make whatif        # regenerates this report from the calibration fit
```

Or for a one-off: `uv run python scripts/whatif.py --out -` prints to stdout.

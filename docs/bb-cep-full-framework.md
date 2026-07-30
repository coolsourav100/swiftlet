# BB-CEP: Bandwidth-Balanced Concurrent Expert Placement
### A task-adaptive framework for CPU/GPU LLM inference on unified-memory architectures

**Author:** Sourav Sarkar
**Status:** Proposed framework — theoretical model + practical validation plan. Not yet empirically confirmed; Section 7 gives the exact steps to validate it.
**Target system:** Apple Silicon (M-series), tested against Qwen3-30B-A3B (MoE)

---

## 1. Problem statement

Existing Mixture-of-Experts (MoE) CPU/GPU offloading techniques (e.g. `llama.cpp`'s `--n-cpu-moe`) were designed for **discrete GPU systems**, where GPU VRAM and CPU system RAM are physically separate, connected by a comparatively slow PCIe bus. The established strategy there is to keep frequently-active weights in VRAM and offload rarely-active experts to system RAM, paying a PCIe transfer cost only when an offloaded expert fires.

**Apple Silicon has no such split.** CPU and GPU read from one shared unified memory pool over one memory controller — there is no PCIe transfer to avoid. This means the standard offloading rationale doesn't transfer directly, and the actual question becomes:

> Given one shared memory bus, can concurrent CPU + GPU execution extract more useful throughput per second than GPU alone, and if so, how should model components be partitioned to do that — and does the answer change by task/phase?

BB-CEP is a proposed answer to that question.

---

## 2. Related work

| Work | What it does | How BB-CEP differs |
|---|---|---|
| **llama.cpp `--n-cpu-moe` / MoE offload guides** (Doctor Shotgun & Geechan, 2026) | Static placement of routed experts to CPU RAM vs. GPU VRAM on discrete-GPU systems, to avoid re-transferring rarely-used experts over PCIe | BB-CEP targets unified-memory systems where there's no PCIe cost to avoid — the goal is concurrent throughput, not transfer avoidance |
| **HeterPS** (heterogeneous parameter server work) | Uses reinforcement learning to learn a per-layer CPU-vs-GPU placement policy, in a GPU-cluster + CPU-aggregation training context | BB-CEP uses a closed-form roofline + scheduling-theory calculation instead of a learned policy, and targets single-device inference rather than distributed training |
| **NUMA-Caffe** (Roy et al., TACO 2018) | NUMA-aware placement across CPU sockets/nodes to minimize remote memory access during training | Not directly applicable — Apple Silicon is single-SoC, no NUMA nodes exist. Referenced here only to explicitly rule out relevance, since it's easy to mistakenly assume NUMA techniques transfer |
| **Roofline model** (Williams et al., 2009) | General HPC performance model: execution time is the max of compute-bound and memory-bound time, given arithmetic intensity | BB-CEP's core cost model *is* a roofline model — this is the established piece being applied to a new setting (concurrent CPU/GPU on shared memory), not reinvented |
| **Two-machine makespan scheduling** (classical scheduling theory) | Partitioning jobs across two parallel machines to minimize the time the slower branch finishes | BB-CEP's partitioning algorithm (Section 4) is a direct application of this — specifically longest-processing-time-first (LPT), which is provably within 4/3 of optimal for two machines |

BB-CEP's contribution is not a new primitive — it's the **specific combination** of roofline modeling + LPT scheduling + phase-awareness (prefill vs. decode), applied to a hardware topology (Apple unified memory) that the existing MoE-offload literature doesn't address.

---

## 3. Cost model

For layer or expert `e`:
- `C_e` — compute required (FLOPs)
- `S_e` — weight size to read from memory (bytes)
- `F_gpu`, `F_cpu` — compute throughput of each engine (FLOPs/s)
- `B_gpu`, `B_cpu` — **effective, contended** achievable bandwidth each engine sustains when both are active on the shared bus (not the theoretical bus max — see Section 7 for how to measure this)

Roofline execution time on each engine:
```
T_gpu(e) = max( C_e / F_gpu , S_e / B_gpu )
T_cpu(e) = max( C_e / F_cpu , S_e / B_cpu )
```

**Two operating regimes:**

- **Decode (batch size 1, per-token generation):** low arithmetic intensity → memory-bound → `T(e) ≈ S_e / B`. This is where CPU/GPU concurrency can help, because performance depends on aggregate bandwidth, not aggregate FLOPs.
- **Prefill (large batch of prompt tokens processed together):** high arithmetic intensity (weight reuse across the batch) → compute-bound → `T(e) ≈ C_e / F`. GPU's FLOPs/s advantage dominates; adding CPU here typically *hurts*, since CPU's much lower FLOPs/s becomes a straggler in any concurrent split.

This is the central reason a **static** split (one `--n-cpu-moe` value for the whole run) is provably suboptimal across a real workload that mixes both phases.

---

## 4. Partitioning algorithm (decode phase)

Given `k` active experts for the current token, partition into GPU-set `G` and CPU-set `C`, executed concurrently, to minimize:
```
T_parallel = max( Σ(S_e, e∈G) / B_gpu ,  Σ(S_e, e∈C) / B_cpu )
```
subject to `B_gpu + B_cpu ≤ B_total_effective` (shared-bus ceiling).

**Algorithm (Longest Processing Time first, adapted):**
```
1. Sort experts by S_e descending
2. Maintain running load estimates: load_G = 0, load_C = 0
3. For each expert e (largest first):
     if (load_G / B_gpu) <= (load_C / B_cpu):
         assign e to G;  load_G += S_e
     else:
         assign e to C;  load_C += S_e
4. Return (G, C)
```
This greedy approach is within 4/3 of the optimal partition for two machines — sufficient given the noise already present in estimating `B_gpu`/`B_cpu` under real contention.

**Phase gate:** only invoke this partitioning during decode. During prefill, route everything to GPU — adding CPU to a compute-bound stage adds a straggler, not a speedup.

---

## 5. Expected speedup

```
Theoretical speedup ceiling = (B_gpu + B_cpu) / B_gpu_max
```
where `B_gpu_max` is GPU-alone achievable bandwidth (the baseline). Since GPU cores have far higher memory-level parallelism than CPU cores, `B_cpu` is expected to be a modest fraction of total bus capacity — realistic expectation is a **15-25% decode-phase speedup**, not a multiplier. This is a testable prediction (Section 7), not an assumption to take on faith.

---

## 6. Target model for validation: Qwen3-30B-A3B

Chosen specifically because:
- MoE architecture (30B total, ~3B active per token) — has genuine "experts" to partition, unlike a dense model
- Fits comfortably in unified memory at 4-bit quantization (~17-18GB), well within reach of an 24GB+ Mac
- Actively maintained GGUF quantizations available (Unsloth Dynamic 2.0 quants)
- Small enough to iterate on quickly; large enough that the memory-bound decode regime is realistic and representative

---

## 7. Validation plan (how to turn this from theory into a measured result)

### 7.1 Bandwidth calibration (hardware-specific, model-independent)
```python
# calibrate_bandwidth.py
import time
import numpy as np
import mlx.core as mx

SIZE_GB = 4
N = int(SIZE_GB * 1e9 / 4)

def bench_gpu(n, iters=20):
    a = mx.random.normal((n,)); mx.eval(a)
    start = time.perf_counter()
    for _ in range(iters):
        b = a * 2.0; mx.eval(b)
    elapsed = time.perf_counter() - start
    return (n * 4 * iters) / elapsed / 1e9  # GB/s

def bench_cpu(n, iters=20):
    a = np.random.randn(n).astype(np.float32)
    start = time.perf_counter()
    for _ in range(iters):
        b = a * 2.0
    elapsed = time.perf_counter() - start
    return (n * 4 * iters) / elapsed / 1e9

print(f"GPU-alone: {bench_gpu(N):.1f} GB/s")
print(f"CPU-alone: {bench_cpu(N):.1f} GB/s")
# Run both concurrently (separate processes) for the CONTENDED figures —
# those, not the solo numbers above, feed Section 3's B_gpu / B_cpu.
```

### 7.2 Model-level sweep and automated comparison
```bash
#!/bin/bash
# bbcep_sweep.sh — sweep --n-cpu-moe, capture llama-bench results, build comparison table
MODEL=./models/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
OUT=bbcep_results.csv

echo "n_cpu_moe,pp512_tps,tg128_tps" > "$OUT"

for n in 0 4 8 12 16 20 24; do
  echo "=== n-cpu-moe=$n ==="
  result=$(./llama-bench -m "$MODEL" --n-gpu-layers 99 --n-cpu-moe "$n" -p 512 -n 128 2>/dev/null)

  pp=$(echo "$result" | grep "pp512" | awk '{print $(NF-1)}')
  tg=$(echo "$result" | grep "tg128" | awk '{print $(NF-1)}')

  echo "$n,$pp,$tg" >> "$OUT"
done

echo "Results written to $OUT"
```

```python
# analyze_bbcep.py — turn the sweep into the theoretical-vs-measured comparison
import pandas as pd

df = pd.read_csv("bbcep_results.csv")
baseline_tg = df.loc[df.n_cpu_moe == 0, "tg128_tps"].values[0]

df["measured_speedup"] = df["tg128_tps"] / baseline_tg
best_row = df.loc[df["measured_speedup"].idxmax()]

print(f"Baseline (GPU-only) decode speed: {baseline_tg:.1f} tok/s")
print(f"Best split found: n_cpu_moe={int(best_row.n_cpu_moe)}, "
      f"measured speedup = {best_row.measured_speedup:.2f}x")
print("Compare this to the theoretical ceiling from Section 5 "
      "using your calibrated B_gpu / B_cpu from step 7.1.")

print(df.to_string(index=False))
```

### 7.3 What "success" looks like
- If measured speedup lands near the theoretical 1.15-1.25x ceiling → the model's core assumption (decode is memory-bound, concurrent CPU/GPU adds bandwidth) is validated on this hardware.
- If measured speedup is much lower → likely means `B_cpu` under real contention is smaller than assumed, or thread/scheduling overhead (context switches, synchronization between the two result streams) is eating the gain — both are useful, specific findings, not failures.
- If measured speedup is *higher* than predicted → worth double-checking the baseline run wasn't itself bandwidth-starved for an unrelated reason (e.g. thermal throttling during the GPU-only baseline).

Either outcome is a legitimate, reportable result — the value of doing this validation step is having an actual number instead of an assumption.

---

## 8. Honest scope and limitations

- This is a **single-node, single-SoC inference framework** — it does not address multi-machine clusters, NUMA (irrelevant on Apple Silicon by design), or training-time gradient synchronization.
- The partitioning algorithm assumes bandwidth contention is roughly additive and boundable by a single `B_total_effective` figure — real memory controllers may show non-linear contention behavior under load, which Section 7.1's concurrent-contended benchmark is specifically designed to catch rather than assume away.
- Section 5's speedup ceiling is a **prediction to test**, not a guarantee. Report the measured number, not the predicted one, once you've run Section 7.

---

## 9. Suggested next artifact
Once Section 7 has been run once, the natural output is a short results write-up: theoretical ceiling vs. measured speedup, the winning `n-cpu-moe` value for Qwen3-30B-A3B specifically, and one paragraph on which of the "what success looks like" outcomes (Section 7.3) actually occurred. That result — not this framework document alone — is what belongs in the Jarvis README and in an interview answer.

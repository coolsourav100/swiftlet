<p align="center">
  <img src="assets/swiftlet-logo.png" width="350" alt="swiftlet — fast local MoE inference">
</p>

<p align="center">
  <b>An adaptive tuning layer for fast local MoE inference on Apple Silicon.</b><br>
  Learns your workload, gets faster the more you use it.
</p>

<p align="center">
  <a href="https://github.com/coolsourav100/swiftlet"><b>GitHub</b></a> ·
  <a href="docs/bb-cep-full-framework.md">BB-CEP Framework</a> ·
  <a href="docs/validation.md">Validation</a>
</p>

<p align="center">
  <img src="assets/demo.svg" width="600" alt="Swiftlet Animated Demo">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/status-Experimental-orange" alt="Experimental">
</p>

**Static configurations don't fit dynamic workloads.** Explore large Mixture-of-Experts (MoE) models on Apple Silicon and unified-memory hardware using a transparent proxy that dynamically shifts computation between your CPU and GPU based on the exact shape of your prompt.

> **swiftlet doesn't replace `llama.cpp`'s inference engine — it sits on top of it.**
> Large models have different bottlenecks depending on what they are doing. A long document Q&A (prefill) is compute-bound and wants GPU dominance. A short chat turn (decode) is memory-bound and benefits from a CPU/GPU split. swiftlet dynamically classifies requests, routes them to optimally tuned `llama-server` instances, and remembers what worked best for next time.

```bash
$ python -m swiftlet.app
```
*Launch the rich terminal UI to chat with your model and monitor live CPU core utilization.*

## The Idea: Where swiftlet fits

swiftlet combines two powerful ideas for a use-case neither fully covers on its own:

1. **BB-CEP (Bandwidth-Balanced Concurrent Expert Placement)**: A roofline-based model demonstrating that MoE expert computation should be split across CPU and GPU on unified-memory hardware, and that this split must be phase-aware (prefill vs. decode).
2. **Colibrì's Learning Cache**: The idea that an inference engine should treat memory as a hierarchy and *learn* from real usage over time.

While colibrì streams models larger than RAM from disk, **swiftlet targets models that already fit in your Mac's unified memory** (like Qwen3-30B-A3B). What these resident models still desperately need is **smarter, learned scheduling of where computation happens**. `llama.cpp`'s `--n-cpu-moe` is chosen once at launch. swiftlet makes it dynamic.

## Architecture & Data Flow

swiftlet transparently intercepts OpenAI-compatible API traffic, deciding *how* to configure the engine for each request.

```text
                    ┌─────────────────────┐
  Request  ────────▶│  Workload Classifier │  buckets by (prompt_len, expected_gen_len)
                    └──────────┬──────────┘
                               │ workload signature
                               ▼
                    ┌─────────────────────┐
                    │   Learned Config     │  signature → best known (n_gpu_layers,
                    │       Store          │  n_cpu_moe, batch_size), persisted to disk
                    └──────────┬──────────┘
                               │ config (or "explore": try an untested config)
                               ▼
                    ┌─────────────────────┐
                    │     Orchestrator     │  maintains a small pool of pre-warmed
                    │  (server pool mgr)   │  llama-server instances at known-good configs
                    └──────────┬──────────┘
                               │
                               ▼
                    llama-server (llama.cpp, Metal backend)
                               │
                               ▼
                    Response + measured tok/s ──▶ fed back into Learned Config Store
```

## Core Techniques and Measured Results

- **Workload Phase Classification:** Automatically categorizes traffic into `PREFILL_HEAVY`, `DECODE_HEAVY`, or `BALANCED`. A 5,000 token document summary gets a different hardware split than a 10 token chat reply.
- **Epsilon-Greedy Config Store:** Records real `tok/s` metrics directly from `llama-server` streams. It exploits the best-known hardware config 85% of the time, and explores untested configurations 15% of the time, perpetually optimizing for your exact machine.
- **Memory-Aware Orchestration:** Operates a bounded `ServerPool`. By default (`max_size=1`), it aggressively evicts old configurations before cold-starting new ones, preventing OOM crashes on 16GB Macs when exploring new MoE splits.
- **Thread & Inference Tuning:** Bypasses default thread caps to fully saturate performance cores, and natively strips out hidden reasoning overhead (like Qwen3's chain-of-thought) when it jeopardizes your generation token budgets.

### Live Measured Performance (M5 Mac, 16GB)

The following is a **Multi-Dimensional Use-Case Profile** based on real data recorded by `swiftlet` on an Apple M5 Mac (16GB). 

*See our [Performance Evaluation Methodology](docs/research_matrix.md) for full details on how we measure Prefill (TTFT) vs Decode (TPOT) speeds.*

It demonstrates how the absolute best hardware configuration (CPU/GPU split) completely changes depending on the size of your input prompt vs your expected output generation. Swiftlet learns these profiles and automatically routes your requests to the optimal split.

<p align="center">
  <img src="assets/performance_matrix.svg" width="800" alt="Performance Matrix">
</p>

## Open Hypotheses and Experiments

swiftlet treats every optimization as a hypothesis until an end-to-end A/B test proves it.

| Hypothesis | Current Evidence | Required Experiment |
|---|---|---|
| CPU/GPU splits benefit decode phases | Confirmed: pure GPU struggles with memory-bound decode | Test across different unified memory bandwidths (M3 vs M5 Max) |
| Thread starvation masks split benefits | Confirmed: `n_threads` capping hides `--n-cpu-moe` impact | Controlled sweep with fixed threads vs automatic threads |
| Qwen3 reasoning limits generation budget | Confirmed: 512 token limits truncate code generation | Measure `tok/s` overhead of `reasoning_content` vs pure generation |
| Fast eviction is better than multi-residence | Confirmed: `max_size=3` causes OOM on 16GB during exploration | Test `max_size=3` on 64GB+ Mac Studios to measure cold-start latency drops |

## Getting Started (For Mac Users)

We have built a frictionless, "1-Click" setup process specifically designed so that **anyone** can run this locally without needing to configure complex CLI arguments or terminal windows.

### 1. Download Swiftlet
1. Download this repository (either `git clone` or Download ZIP and extract it).
2. Open the `swiftlet` folder in your Mac's Finder.

### 2. Run the 1-Click Installer
1. Double-click the `install.command` file. 
   *(Note: If macOS says "unidentified developer", right-click it, select **Open**, and click Open again).*
2. A terminal window will appear. It will automatically:
   - Check if you have [Ollama](https://ollama.com/) installed (and prompt you if not).
   - Find your downloaded AI models.
   - Install all required Python dependencies securely in a virtual environment.
   - Automatically configure your `.env` file perfectly for your Mac.

### 3. Launch the App
1. Once installation is complete, double-click the `start.command` file.
2. It will automatically open two terminal tabs:
   - **The Engine:** One tab runs the background proxy that transparently optimizes your hardware splits.
   - **The Chat UI:** The other tab opens the sleek Swiftlet Chat interface where you can talk to your model!


## Credits & License

Built on ideas from **[colibrì](https://github.com/JustVugg/colibri)** (JustVugg) and **llama.cpp** (ggml-org). This project doesn't reimplement either — it's a thin, honest orchestration layer designed to extract maximum efficiency from `llama.cpp`'s Metal backend on Apple Silicon.

**License:** Apache 2.0

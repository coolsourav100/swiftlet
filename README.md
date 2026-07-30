# swiftlet

**An adaptive tuning layer for fast local MoE inference on Apple Silicon — learns your workload, gets faster the more you use it.**

swiftlet doesn't replace `llama.cpp`'s inference engine — it sits on top of it, deciding *how* to configure it for each request based on two ideas borrowed from existing work, combined for a case neither fully covers on its own:

- **BB-CEP** (bandwidth-balanced concurrent expert placement — see [`docs/bb-cep-full-framework.md`](docs/bb-cep-full-framework.md)): a roofline-based model for splitting MoE expert computation across CPU and GPU on unified-memory hardware, phase-aware (prefill vs. decode need different splits).
- **[colibrì](https://github.com/JustVugg/colibri)**: pioneered treating VRAM/RAM/disk as one managed hierarchy for MoE models too large to fit in memory, and — the idea swiftlet actually borrows — a **learning cache** that records real usage and gets faster over time, plus router-lookahead prefetching.

## Where swiftlet fits that neither source covers

colibrì's disk-streaming tier solves "the model is bigger than your RAM." That's not your problem if you're running Qwen3-30B-A3B or similarly-sized MoE models on a 16GB+ Mac — they already fit. What colibrì-scale models *don't* need, small resident models still benefit from: **smarter, learned scheduling of where computation happens**, not where data lives. `llama.cpp`'s own `--n-cpu-moe` is a single static value chosen once at process launch — it can't adapt between a short chat reply and a long document-processing prefill in the same session, and it doesn't remember what worked best last time.

swiftlet's job: **classify each request by workload shape, look up (or learn) the best CPU/GPU configuration for that shape, and route the request to a correctly-tuned `llama-server` instance** — improving automatically as it sees more of your real traffic, the way colibrì's usage-learning cache improves expert placement over a session.

## Architecture

```
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

## What's implemented vs. what's designed but not built

Being direct about scope, the same way the BB-CEP document was:

**Implemented and working end-to-end with llama-server:**
- Workload classifier (`swiftlet/classifier.py`)
- Learned config store with epsilon-greedy exploration (`swiftlet/config_store.py`)
- Orchestrator's decision logic (`swiftlet/orchestrator.py`) — process pool management and request routing
- HTTP Proxying (`swiftlet/cli.py`) — transparently proxies OpenAI-compatible `/v1/chat/completions` traffic, extracting actual `tok/s` measurements directly from `llama-server` streams to feed back into the learned store.
- Interactive Chat CLI (`chat_cli.py`) — a sample frontend demonstrating streaming responses, performance threshold tracking, and graceful handling of model reasoning.

## Quick start

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your environment:**
   Instead of passing long paths every time, you can configure swiftlet using a `.env` file. Copy the example and edit it to match your paths:
   ```bash
   cp .env.example .env
   ```
   *Note: If you use Ollama on macOS, your `llama-server` binary is usually at `/Applications/Ollama.app/Contents/Resources/llama-server`, and models are stored in `~/.ollama/models/blobs/`.*

3. **Start the swiftlet proxy:**
   ```bash
   python -m swiftlet.cli
   ```
   *(Alternatively, you can still pass `--model` and `--llama-server` via command line arguments if you prefer).*

4. **Chat with your model:**
   In a separate terminal, run the chat client:
   ```bash
   python chat_cli.py
   ```
   As you chat, the proxy will transparently launch `llama-server` instances with different CPU/GPU splits, measure the token generation speed, and learn which hardware configuration is optimal for different types of prompts (e.g., short chats vs. long coding tasks).

## Credits

Built on ideas from [colibrì](https://github.com/JustVugg/colibri) (JustVugg) and `llama.cpp` (ggml-org). This project doesn't reimplement either — it's a thin, honest orchestration layer that assumes you already have `llama.cpp` built with Metal support.

## License
Apache 2.0

# Wiring swiftlet to a real llama-server + validating it actually helps

Everything in `swiftlet/` is real, tested Python — the classifier, the learned
config store, and the pool/routing logic all pass their test suites without
needing a model. What's *not* yet done is connecting that logic to an actual
running `llama-server`, and confirming the learned configs genuinely beat a
static baseline on real hardware. This doc is the honest checklist for that.

## 1. Implement a real launcher

`ServerPool` takes a `launcher` callable. Replace the fake one from the demo
with something like:

```python
import subprocess
import httpx
import time

def real_launcher(config, port) -> ServerHandle:
    proc = subprocess.Popen([
        "./llama-server",
        "-m", "/path/to/Qwen3-30B-A3B-UD-Q4_K_XL.gguf",
        "--n-gpu-layers", str(config.n_gpu_layers),
        "--n-cpu-moe", str(config.n_cpu_moe),
        "--batch-size", str(config.batch_size),
        "--port", str(port),
    ])

    # Wait for it to actually be ready before returning
    for _ in range(60):
        try:
            httpx.get(f"http://localhost:{port}/health", timeout=1.0)
            break
        except httpx.ConnectError:
            time.sleep(1)
    else:
        raise RuntimeError(f"llama-server on port {port} didn't come up in time")

    return ServerHandle(config=config, port=port, started_at=time.time())
```

## 2. Implement real request proxying + tok/s measurement

Forward the actual chat request to the routed server's `/v1/chat/completions`
(or `/completion`), and compute tok/s from the response's usage stats
(`llama-server` reports timing info you can use directly — check its
`/completion` response for `timings.predicted_per_second`, or compute it
yourself from token count / wall-clock time).

## 3. Run the actual experiment

- Fixed baseline: force `n_cpu_moe=0` for every request for a session, log tok/s
- swiftlet-managed: let the orchestrator route and learn for an equivalent session
- Compare mean tok/s per workload phase between the two

## 4. What "success" looks like here

This is the same discipline as BB-CEP Section 7.3: report what you actually
measure, not what the design predicts.

- If swiftlet's learned configs beat the static baseline on decode-heavy
  workloads specifically (where BB-CEP predicts a real but modest gain),
  that's the result to write up — with the actual number, not "should be faster."
- If gains are negligible or negative, worth checking: is the exploration
  epsilon too high for your usage volume to converge? Is process-relaunch
  overhead (each `get_or_launch` miss costs a full model load) eating any
  gains from a better runtime config? That relaunch cost is a real, currently
  unaddressed weakness of the v1 design — worth measuring explicitly rather
  than assuming it's negligible.

## Known limitation worth stating plainly

Every distinct config in the learned store potentially means a **full model
reload** the first time it's tried (a `llama-server` restart, not a cheap
operation for a 30B model). The `ServerPool`'s bounded pool + LRU eviction
mitigates this only if your traffic revisits the same few workload shapes
repeatedly — for a wildly varied workload with no repeating shapes, swiftlet's
learning loop won't have time to pay for itself. This is the honest tradeoff
to measure in step 3, not something the design already solved.

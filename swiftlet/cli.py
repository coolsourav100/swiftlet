"""
CLI entrypoint. NOTE: end-to-end request proxying (actually forwarding a
chat request to the routed llama-server and streaming the response back)
is not implemented here — that requires your real llama-server binary and
model path to build and test against. What IS implemented and runnable
right now is the routing decision itself, shown via --demo.
"""

import argparse
from pathlib import Path

from .classifier import classify
from .config_store import LearnedConfigStore
from .orchestrator import Orchestrator, ServerPool, ServerHandle


def demo(store_path: str):
    """
    Runs a handful of representative requests through the real classifier +
    config store + pool logic, using a fake launcher (no actual model load),
    so you can see the routing decisions and learning behavior directly.
    """
    store = LearnedConfigStore(store_path, seed=42)

    def fake_launcher(config, port):
        print(f"  [would launch] llama-server --n-gpu-layers {config.n_gpu_layers} "
              f"--n-cpu-moe {config.n_cpu_moe} --batch-size {config.batch_size} --port {port}")
        return ServerHandle(config=config, port=port, started_at=0.0)

    pool = ServerPool(max_size=3, launcher=fake_launcher)
    orch = Orchestrator(store, pool)

    scenarios = [
        ("short chat turn", 80, 150),
        ("long document Q&A", 4000, 100),
        ("open-ended long generation", 100, 1500),
        ("short chat turn (again)", 90, 140),
    ]

    for name, prompt_tokens, gen_tokens in scenarios:
        sig, config, handle, exploring = orch.route(prompt_tokens, gen_tokens)
        tag = "EXPLORE" if exploring else "EXPLOIT"
        print(f"\n[{name}] prompt={prompt_tokens} gen={gen_tokens} -> signature={sig} phase={sig.phase.value}")
        print(f"  decision: {tag} -> n_gpu_layers={config.n_gpu_layers} n_cpu_moe={config.n_cpu_moe}")

        # Simulated measurement — replace with a real tok/s reading from
        # llama-server's response once wired up to an actual instance.
        fake_measured_tps = 20 + config.n_cpu_moe * 0.3
        orch.record(sig, config, fake_measured_tps)
        print(f"  recorded (simulated) {fake_measured_tps:.1f} tok/s")

    print(f"\nLearned store saved to: {store_path}")


def main():
    parser = argparse.ArgumentParser(description="swiftlet — adaptive CPU/GPU config learning for llama.cpp")
    parser.add_argument("--model", help="Path to a GGUF model (required once real serving is wired up)")
    parser.add_argument("--store", default="swiftlet_learned_config.json", help="Path to the learned config store")
    parser.add_argument("--demo", action="store_true", help="Run the routing/learning demo with a fake launcher")
    args = parser.parse_args()

    if args.demo or not args.model:
        if not args.demo:
            print("No --model given and real serving isn't wired up yet — running --demo instead.\n")
        demo(args.store)
        return

    raise NotImplementedError(
        "Real llama-server launching isn't implemented yet — see docs/validation.md "
        "for how to wire ServerPool's launcher to an actual `llama-server` process."
    )


if __name__ == "__main__":
    main()

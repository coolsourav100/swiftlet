#!/usr/bin/env python3
"""
Swiftlet v2 Benchmark Suite

Compares learning strategies against each other and against static baselines
(Ollama, raw llama.cpp) to produce a quantitative comparison matrix.

Run:
    python3 -m swiftlet.tests.benchmark
    python3 -m swiftlet.tests.benchmark --requests 200 --trials 5
    python3 -m swiftlet.tests.benchmark --real --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add parent to path so we can import swiftlet when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from swiftlet.classifier import WorkloadSignature, Phase, classify
from swiftlet.config_store import (
    EngineConfig,
    FULL_CONFIG_SPACE,
    LearnedConfigStore,
    BayesianConfigStore,
    ConfigStats,
)
from swiftlet.gp import GaussianProcessRegressor, RBFKernel, ucb


# ═══════════════════════════════════════════════════════════════════════
# 1. SIMULATED PERFORMANCE SURFACE
# ═══════════════════════════════════════════════════════════════════════

def simulated_tps(
    config: EngineConfig,
    signature: WorkloadSignature,
    noise_std: float = 1.5,
    rng: random.Random | None = None,
) -> float:
    """
    Realistic tok/s surface for a ~4B MoE model on Apple M5 (12 GiB GPU).

    Key characteristics:
      - GPU layers: strong positive, diminishing returns above ~80
      - CPU MoE: helps decode (+0.8/moe), hurts prefill (-0.3/moe)
      - Batch size 512 > 256 for prefill; opposite for decode
      - Noise: ±1.5 tok/s (measurement variance)

    The optimal config depends on the workload — this is the core
    insight that makes adaptive routing valuable.
    """
    rng = rng or random.Random()

    # GPU effect: diminishing returns
    gpu_effect = 25.0 * (1.0 - math.exp(-config.n_gpu_layers / 40.0))

    # CPU MoE effect: depends on workload phase
    if signature.gen_bucket >= 2:
        moe_effect = 0.8 * config.n_cpu_moe
    elif signature.prompt_bucket >= 2:
        moe_effect = -0.3 * config.n_cpu_moe
    else:
        moe_effect = 0.2 * config.n_cpu_moe

    # Batch size effect
    if signature.prompt_bucket >= 2:
        batch_effect = 2.0 if config.batch_size == 512 else 0.0
    else:
        batch_effect = 0.0 if config.batch_size == 512 else 1.0

    # Base + noise
    base = 5.0
    noise = rng.gauss(0, noise_std)

    return max(base + gpu_effect + moe_effect + batch_effect + noise, 1.0)


def find_optimal_config(
    signature: WorkloadSignature,
    config_space: list[EngineConfig],
    rng: random.Random,
    n_samples: int = 1000,
) -> tuple[EngineConfig, float]:
    """Find the true optimal config for a signature (by Monte Carlo)."""
    best_config = config_space[0]
    best_mean = -float("inf")

    for config in config_space:
        samples = [simulated_tps(config, signature, rng=rng) for _ in range(n_samples)]
        mean = statistics.mean(samples)
        if mean > best_mean:
            best_mean = mean
            best_config = config

    return best_config, best_mean


# ═══════════════════════════════════════════════════════════════════════
# 2. WORKLOAD GENERATOR
# ═══════════════════════════════════════════════════════════════════════

WORKLOAD_DISTRIBUTION = [
    # (prompt_tokens, gen_tokens, weight)
    # Simulates a real chat workload: mostly short, some long
    (80,   150,  0.30),   # quick chat
    (200,  300,  0.20),   # normal conversation
    (100,  1500, 0.10),   # open-ended generation
    (4000, 100,  0.15),   # document Q&A
    (2000, 800,  0.10),   # code refactoring
    (3000, 2000, 0.05),   # deep analysis
    (50,   60,   0.05),   # quick question
    (500,  400,  0.05),   # follow-up
]


def generate_workload(n_requests: int, rng: random.Random) -> list[tuple[int, int]]:
    """Generate a realistic sequence of requests."""
    total_weight = sum(w for _, _, w in WORKLOAD_DISTRIBUTION)
    cumulative = []
    running = 0
    for pt, gt, w in WORKLOAD_DISTRIBUTION:
        running += w / total_weight
        cumulative.append((pt, gt, running))

    requests = []
    for _ in range(n_requests):
        r = rng.random()
        for pt, gt, threshold in cumulative:
            if r <= threshold:
                requests.append((pt, gt))
                break
        else:
            requests.append((80, 150))  # fallback

    return requests


# ═══════════════════════════════════════════════════════════════════════
# 3. STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

class Strategy:
    """Base class for all strategies being benchmarked."""
    name: str = "base"

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        raise NotImplementedError

    def record(self, signature: WorkloadSignature, config: EngineConfig, tps: float):
        raise NotImplementedError


class OracleStrategy(Strategy):
    """Always picks the true optimal config. Upper bound."""
    name = "Oracle (upper bound)"

    def __init__(self, config_space: list[EngineConfig], rng: random.Random):
        self._optimal: dict[str, tuple[EngineConfig, float]] = {}
        for sig_key in self._all_signature_keys():
            sig = self._parse_sig(sig_key)
            if sig:
                opt_config, opt_tps = find_optimal_config(sig, config_space, rng)
                self._optimal[sig_key] = (opt_config, opt_tps)

    def _all_signature_keys(self) -> list[str]:
        keys = []
        for pb in range(3):
            for gb in range(3):
                keys.append(f"prompt_b{pb}_gen_b{gb}")
        return keys

    def _parse_sig(self, sig_key: str) -> Optional[WorkloadSignature]:
        try:
            parts = sig_key.split("_")
            return WorkloadSignature(
                prompt_bucket=int(parts[1][1:]),
                gen_bucket=int(parts[3][1:]),
            )
        except (IndexError, ValueError):
            return None

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        return self._optimal[str(signature)][0]

    def record(self, signature, config, tps):
        pass


class StaticDefaultStrategy(Strategy):
    """Always uses gpu=99, moe=0. Simulates Ollama / raw llama.cpp."""
    name = "Ollama / llama.cpp (static gpu=99/moe=0)"

    def __init__(self):
        self._config = EngineConfig(n_gpu_layers=99, n_cpu_moe=0, batch_size=512)

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        return self._config

    def record(self, signature, config, tps):
        pass


class StaticMidStrategy(Strategy):
    """Always uses gpu=60, moe=6. A "reasonable guess" static config."""
    name = "Static mid-range (gpu=60/moe=6)"

    def __init__(self):
        self._config = EngineConfig(n_gpu_layers=60, n_cpu_moe=6, batch_size=512)

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        return self._config

    def record(self, signature, config, tps):
        pass


class RandomStrategy(Strategy):
    """Picks a random config every time. Lower bound."""
    name = "Random config"

    def __init__(self, config_space: list[EngineConfig], seed: int = 42):
        self._config_space = config_space
        self._rng = random.Random(seed)

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        return self._rng.choice(self._config_space)

    def record(self, signature, config, tps):
        pass


class EpsilonGreedyStrategy(Strategy):
    """Swiftlet v1 — epsilon-greedy with per-phase config space."""
    name = "Swiftlet ε-greedy (v1)"

    def __init__(self, seed: int = 42):
        self._store = LearnedConfigStore(
            path="/tmp/swiftlet_bench_eps.json", epsilon=0.15, seed=seed
        )
        # Clean slate
        Path("/tmp/swiftlet_bench_eps.json").unlink(missing_ok=True)
        self._store = LearnedConfigStore(
            path="/tmp/swiftlet_bench_eps.json", epsilon=0.15, seed=seed
        )

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        config, _ = self._store.choose_config(signature)
        return config

    def record(self, signature, config, tps):
        self._store.record_result(signature, config, tps)


class BayesianStrategy(Strategy):
    """Swiftlet v2 — GP+UCB with warm start and pruned config space."""
    name = "Swiftlet Bayesian (v2)"

    def __init__(self, seed: int = 42, warm_start: bool = True):
        Path("/tmp/swiftlet_bench_bayes.json").unlink(missing_ok=True)
        self._store = BayesianConfigStore(
            path="/tmp/swiftlet_bench_bayes.json", seed=seed
        )
        if warm_start:
            self._store.warm_start(lambda c, s: simulated_tps(c, s, rng=random.Random(seed)))

    def choose(self, signature: WorkloadSignature) -> EngineConfig:
        config, _ = self._store.choose_config(signature)
        return config

    def record(self, signature, config, tps):
        self._store.record_result(signature, config, tps)


# ═══════════════════════════════════════════════════════════════════════
# 4. BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    strategy_name: str
    total_requests: int
    cumulative_regret: float
    avg_tps: float
    best_tps: float
    worst_tps: float
    p50_tps: float
    p95_tps: float
    p99_tps: float
    requests_to_within_5pct: Optional[int]
    requests_to_within_1pct: Optional[int]
    tps_over_time: list[float] = field(default_factory=list)
    regret_over_time: list[float] = field(default_factory=list)
    config_diversity: int = 0  # how many distinct configs were tried


def run_benchmark(
    strategy: Strategy,
    requests: list[tuple[int, int]],
    rng: random.Random,
    optimal_per_sig: dict[str, float],
) -> BenchmarkResult:
    """Run a single strategy through the workload and collect metrics."""
    all_tps = []
    cumulative_regret = 0.0
    regret_over_time = []
    tps_over_time = []
    configs_used: set[str] = set()

    # Track convergence: when did we first get within X% of optimal?
    within_5pct: Optional[int] = None
    within_1pct: Optional[int] = None

    for i, (prompt_tokens, gen_tokens) in enumerate(requests):
        signature = classify(prompt_tokens, gen_tokens)
        sig_key = str(signature)

        # Choose config
        config = strategy.choose(signature)
        configs_used.add(config.key())

        # Simulate inference
        tps = simulated_tps(config, signature, rng=rng)

        # Record result
        strategy.record(signature, config, tps)

        # Compute regret vs optimal
        optimal_tps = optimal_per_sig.get(sig_key, 30.0)
        regret = optimal_tps - tps
        cumulative_regret += regret

        all_tps.append(tps)
        tps_over_time.append(tps)
        regret_over_time.append(cumulative_regret)

        # Check convergence
        if optimal_tps > 0:
            pct_off = abs(tps - optimal_tps) / optimal_tps
            if within_5pct is None and pct_off <= 0.05:
                within_5pct = i + 1
            if within_1pct is None and pct_off <= 0.01:
                within_1pct = i + 1

    sorted_tps = sorted(all_tps)

    return BenchmarkResult(
        strategy_name=strategy.name,
        total_requests=len(requests),
        cumulative_regret=round(cumulative_regret, 2),
        avg_tps=round(statistics.mean(all_tps), 2),
        best_tps=round(max(all_tps), 2),
        worst_tps=round(min(all_tps), 2),
        p50_tps=round(sorted_tps[len(sorted_tps) // 2], 2),
        p95_tps=round(sorted_tps[int(len(sorted_tps) * 0.95)], 2),
        p99_tps=round(sorted_tps[int(len(sorted_tps) * 0.99)], 2),
        requests_to_within_5pct=within_5pct,
        requests_to_within_1pct=within_1pct,
        tps_over_time=tps_over_time,
        regret_over_time=regret_over_time,
        config_diversity=len(configs_used),
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. COMPARISON MATRIX
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_matrix(results: list[BenchmarkResult], optimal_avg: float):
    """Print the main comparison matrix."""
    print("\n" + "=" * 100)
    print("  SWIFTLET v2 — PERFORMANCE COMPARISON MATRIX")
    print("=" * 100)

    header = (
        f"{'Strategy':<40} {'Avg':>6} {'Best':>6} {'P50':>6} "
        f"{'P95':>6} {'Regret':>8} {'→5%':>5} {'→1%':>5} "
        f"{'Cfgs':>4} {'Quality':>8}"
    )
    print(header)
    print("-" * 100)

    for r in sorted(results, key=lambda x: -x.avg_tps):
        quality = round((r.avg_tps / optimal_avg) * 100, 1) if optimal_avg > 0 else 0
        conv5 = str(r.requests_to_within_5pct) if r.requests_to_within_5pct else "—"
        conv1 = str(r.requests_to_within_1pct) if r.requests_to_within_1pct else "—"

        print(
            f"{r.strategy_name:<40} {r.avg_tps:>6.1f} {r.best_tps:>6.1f} "
            f"{r.p50_tps:>6.1f} {r.p95_tps:>6.1f} {r.cumulative_regret:>8.1f} "
            f"{conv5:>5} {conv1:>5} {r.config_diversity:>4} "
            f"{quality:>7.1f}%"
        )

    print("-" * 100)
    print(
        f"  Quality = avg_tps / oracle_avg_tps × 100\n"
        f"  →5% = requests to first result within 5% of optimal\n"
        f"  →1% = requests to first result within 1% of optimal\n"
        f"  Regret = cumulative (optimal_tps - actual_tps) over all requests\n"
        f"  Cfgs = number of distinct configs explored\n"
    )


def print_convergence_table(results: list[BenchmarkResult]):
    """Print convergence speed comparison."""
    print("\n" + "=" * 80)
    print("  CONVERGENCE SPEED — How many requests to find the optimal config?")
    print("=" * 80)

    checkpoints = [5, 10, 20, 50, 100, 200]

    header = f"{'Strategy':<40} " + " ".join(f"{c:>5}" for c in checkpoints)
    print(header)
    print("-" * 80)

    for r in results:
        row = f"{r.strategy_name:<40} "
        for checkpoint in checkpoints:
            if checkpoint <= len(r.tps_over_time):
                # Average tok/s over the last 5 requests at this checkpoint
                window = r.tps_over_time[max(0, checkpoint - 5):checkpoint]
                avg = statistics.mean(window)
                row += f"{avg:>5.1f} "
            else:
                row += f"{'—':>5} "
        print(row)

    print()


def print_regret_comparison(results: list[BenchmarkResult]):
    """Print regret at key checkpoints."""
    print("\n" + "=" * 80)
    print("  CUMULATIVE REGRET — Total wasted performance vs optimal")
    print("=" * 80)

    checkpoints = [5, 10, 20, 50, 100, 200]

    header = f"{'Strategy':<40} " + " ".join(f"{c:>7}" for c in checkpoints)
    print(header)
    print("-" * 80)

    for r in results:
        row = f"{r.strategy_name:<40} "
        for checkpoint in checkpoints:
            if checkpoint <= len(r.regret_over_time):
                row += f"{r.regret_over_time[checkpoint-1]:>7.1f} "
            else:
                row += f"{'—':>7} "
        print(row)

    print()


def print_cross_signature_analysis(results: list[BenchmarkResult]):
    """Show how each strategy handles different workload signatures."""
    print("\n" + "=" * 80)
    print("  CROSS-SIGNATURE ANALYSIS — Does the strategy generalize?")
    print("=" * 80)

    sig_labels = {
        "prompt_b0_gen_b0": "quick Q&A",
        "prompt_b0_gen_b1": "short chat",
        "prompt_b0_gen_b2": "long gen",
        "prompt_b1_gen_b0": "medium prompt",
        "prompt_b1_gen_b1": "balanced",
        "prompt_b2_gen_b0": "doc Q&A",
        "prompt_b2_gen_b1": "long+gen",
        "prompt_b2_gen_b2": "deep analysis",
    }

    # We need to re-run with per-signature tracking
    # For now, just show the config diversity
    for r in results:
        print(
            f"  {r.strategy_name:<40} explored {r.config_diversity} distinct configs"
        )

    print()


def print_speedup_matrix(results: list[BenchmarkResult], baseline_name: str):
    """Show how much faster each strategy is vs the baseline."""
    baseline = next(
        (r for r in results if baseline_name in r.strategy_name), None
    )
    if not baseline:
        return

    print("\n" + "=" * 80)
    print(f"  SPEEDUP vs {baseline_name}")
    print("=" * 80)

    print(f"{'Strategy':<40} {'Avg tok/s':>10} {'Speedup':>10} {'Regret ↓':>10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: -x.avg_tps):
        if baseline.avg_tps > 0:
            speedup = r.avg_tps / baseline.avg_tps
            regret_reduction = (
                1.0 - r.cumulative_regret / baseline.cumulative_regret
                if baseline.cumulative_regret > 0
                else 0
            )
        else:
            speedup = 0
            regret_reduction = 0

        print(
            f"{r.strategy_name:<40} {r.avg_tps:>10.1f} "
            f"{speedup:>9.2f}x {regret_reduction:>9.1%}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════
# 6. PRESSURE TEST
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PressureTestResult:
    total_requests: int
    concurrency: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    errors: int
    avg_tps: float


def pressure_test_simulated(
    strategy: Strategy,
    n_requests: int = 500,
    concurrency: int = 10,
    seed: int = 42,
) -> PressureTestResult:
    """
    Simulate concurrent requests hitting the strategy.

    Tests:
      - Thread safety of the config store
      - Latency under load
      - Throughput
      - Error rate
    """
    rng = random.Random(seed)
    requests = generate_workload(n_requests, rng)
    latencies: list[float] = []
    tps_values: list[float] = []
    errors = 0
    lock = threading.Lock()

    def handle_request(prompt_tokens: int, gen_tokens: int):
        nonlocal errors
        start = time.monotonic()
        try:
            signature = classify(prompt_tokens, gen_tokens)
            config = strategy.choose(signature)
            tps = simulated_tps(config, signature, rng=random.Random())
            strategy.record(signature, config, tps)
            elapsed = time.monotonic() - start
            with lock:
                latencies.append(elapsed * 1000)  # ms
                tps_values.append(tps)
        except Exception as e:
            with lock:
                errors += 1

    # Run with thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_time = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(handle_request, pt, gt)
            for pt, gt in requests
        ]
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    total_time = time.monotonic() - start_time

    sorted_latencies = sorted(latencies)

    return PressureTestResult(
        total_requests=n_requests,
        concurrency=concurrency,
        avg_latency_ms=round(statistics.mean(latencies), 2) if latencies else 0,
        p50_latency_ms=round(sorted_latencies[len(sorted_latencies) // 2], 2) if sorted_latencies else 0,
        p95_latency_ms=round(sorted_latencies[int(len(sorted_latencies) * 0.95)], 2) if sorted_latencies else 0,
        p99_latency_ms=round(sorted_latencies[int(len(sorted_latencies) * 0.99)], 2) if sorted_latencies else 0,
        throughput_rps=round(n_requests / total_time, 1),
        errors=errors,
        avg_tps=round(statistics.mean(tps_values), 2) if tps_values else 0,
    )


def print_pressure_results(
    strategy_results: list[tuple[str, PressureTestResult]]
):
    print("\n" + "=" * 100)
    print("  PRESSURE TEST — Concurrent Load (500 requests, 10 threads)")
    print("=" * 100)

    header = (
        f"{'Strategy':<40} {'Avg ms':>8} {'P50 ms':>8} {'P95 ms':>8} "
        f"{'P99 ms':>8} {'RPS':>8} {'Errors':>7} {'Avg t/s':>8}"
    )
    print(header)
    print("-" * 100)

    for name, r in strategy_results:
        print(
            f"{name:<40} {r.avg_latency_ms:>8.1f} {r.p50_latency_ms:>8.1f} "
            f"{r.p95_latency_ms:>8.1f} {r.p99_latency_ms:>8.1f} "
            f"{r.throughput_rps:>8.1f} {r.errors:>7} {r.avg_tps:>8.1f}"
        )

    print()


def print_concurrency_scaling(seed: int = 42):
    """Test how the Bayesian strategy scales with concurrency."""
    print("\n" + "=" * 80)
    print("  CONCURRENCY SCALING — Bayesian strategy at different load levels")
    print("=" * 80)

    header = f"{'Concurrency':>12} {'Avg ms':>8} {'P95 ms':>8} {'RPS':>8} {'Errors':>7}"
    print(header)
    print("-" * 80)

    for conc in [1, 2, 5, 10, 20, 50]:
        Path("/tmp/swiftlet_pressure_bayes.json").unlink(missing_ok=True)
        strategy = BayesianStrategy(seed=seed)
        result = pressure_test_simulated(
            strategy, n_requests=200, concurrency=conc, seed=seed
        )
        print(
            f"{conc:>12} {result.avg_latency_ms:>8.1f} "
            f"{result.p95_latency_ms:>8.1f} {result.throughput_rps:>8.1f} "
            f"{result.errors:>7}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════
# 7. REAL-WORLD BENCHMARK (against live Swiftlet proxy)
# ═══════════════════════════════════════════════════════════════════════

def real_world_benchmark(endpoint: str, n_requests: int = 50):
    """Run benchmark against a live Swiftlet proxy."""
    import httpx

    print(f"\n  Real-world benchmark against: {endpoint}")
    print(f"  Sending {n_requests} requests...\n")

    prompts = [
        ("short", "What is 2+2?"),
        ("medium", "Explain the difference between TCP and UDP in 3 paragraphs."),
        ("long", "Write a Python function that implements binary search on a sorted list, "
                 "with proper error handling and type hints. Include docstring and examples."),
        ("code", "Refactor this code to use async/await:\n"
                 "def fetch_data(url):\n    response = requests.get(url)\n    return response.json()"),
    ]

    latencies = []
    tps_values = []
    errors = 0

    for i in range(n_requests):
        label, prompt = prompts[i % len(prompts)]
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 256,
        }

        start = time.monotonic()
        try:
            resp = httpx.post(
                f"{endpoint}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            )
            elapsed = time.monotonic() - start
            latencies.append(elapsed * 1000)

            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                if completion_tokens > 0 and elapsed > 0:
                    tps = completion_tokens / elapsed
                    tps_values.append(tps)

                gpu = resp.headers.get("X-Swiftlet-GPU-Layers", "?")
                moe = resp.headers.get("X-Swiftlet-CPU-MoE", "?")
                print(
                    f"  [{i+1:3d}/{n_requests}] {label:<8} "
                    f"{elapsed*1000:>7.0f}ms  "
                    f"{tps_values[-1]:>5.1f} tok/s  "
                    f"gpu={gpu} moe={moe}"
                )
            else:
                errors += 1
                print(f"  [{i+1:3d}/{n_requests}] {label:<8} ERROR {resp.status_code}")

        except Exception as e:
            errors += 1
            print(f"  [{i+1:3d}/{n_requests}] {label:<8} EXCEPTION: {e}")

    if latencies:
        sorted_lat = sorted(latencies)
        print(f"\n  Results:")
        print(f"    Avg latency:    {statistics.mean(latencies):.0f} ms")
        print(f"    P50 latency:    {sorted_lat[len(sorted_lat)//2]:.0f} ms")
        print(f"    P95 latency:    {sorted_lat[int(len(sorted_lat)*0.95)]:.0f} ms")
        print(f"    Avg tok/s:      {statistics.mean(tps_values):.1f}" if tps_values else "")
        print(f"    Errors:         {errors}/{n_requests}")


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Swiftlet v2 Benchmark Suite"
    )
    parser.add_argument(
        "--requests", type=int, default=100,
        help="Number of simulated requests per strategy (default: 100)"
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of independent trials to average (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Run real-world benchmark against live proxy"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:8000",
        help="Swiftlet proxy endpoint for --real benchmark"
    )
    parser.add_argument(
        "--pressure", action="store_true",
        help="Run pressure tests (concurrent load)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run: 20 requests, 1 trial"
    )

    args = parser.parse_args()

    if args.quick:
        args.requests = 20
        args.trials = 1

    if args.real:
        real_world_benchmark(args.endpoint, n_requests=args.requests)
        return

    # ── Compute optimal per-signature ──
    print("Computing optimal configs per signature (Monte Carlo)...")
    rng = random.Random(args.seed)
    optimal_per_sig: dict[str, float] = {}
    for pb in range(3):
        for gb in range(3):
            sig = WorkloadSignature(prompt_bucket=pb, gen_bucket=gb)
            _, opt_tps = find_optimal_config(sig, FULL_CONFIG_SPACE, rng)
            optimal_per_sig[str(sig)] = opt_tps
            print(f"  {sig} → {opt_tps:.1f} tok/s")

    optimal_avg = statistics.mean(optimal_per_sig.values())

    # ── Generate workload ──
    print(f"\nGenerating {args.requests} requests...")
    requests = generate_workload(args.requests, rng)

    # ── Define strategies ──
    strategy_classes = [
        lambda: OracleStrategy(FULL_CONFIG_SPACE, random.Random(args.seed)),
        StaticDefaultStrategy,
        StaticMidStrategy,
        lambda: RandomStrategy(FULL_CONFIG_SPACE, seed=args.seed),
        EpsilonGreedyStrategy,
        BayesianStrategy,
    ]

    # ── Run benchmarks ──
    all_results: dict[str, list[BenchmarkResult]] = defaultdict(list)

    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1}/{args.trials} ---")

        for strategy_fn in strategy_classes:
            strategy = strategy_fn()
            print(f"  Running {strategy.name}...")

            trial_rng = random.Random(args.seed + trial + 100)
            result = run_benchmark(strategy, requests, trial_rng, optimal_per_sig)
            all_results[strategy.name].append(result)

    # ── Average across trials ──
    averaged_results = []
    for name, trials in all_results.items():
        avg_result = BenchmarkResult(
            strategy_name=name,
            total_requests=trials[0].total_requests,
            cumulative_regret=round(statistics.mean(r.cumulative_regret for r in trials), 2),
            avg_tps=round(statistics.mean(r.avg_tps for r in trials), 2),
            best_tps=round(max(r.best_tps for r in trials), 2),
            worst_tps=round(min(r.worst_tps for r in trials), 2),
            p50_tps=round(statistics.mean(r.p50_tps for r in trials), 2),
            p95_tps=round(statistics.mean(r.p95_tps for r in trials), 2),
            p99_tps=round(statistics.mean(r.p99_tps for r in trials), 2),
            requests_to_within_5pct=trials[0].requests_to_within_5pct,
            requests_to_within_1pct=trials[0].requests_to_within_1pct,
            config_diversity=round(statistics.mean(r.config_diversity for r in trials)),
        )
        averaged_results.append(avg_result)

    # ── Print results ──
    print_comparison_matrix(averaged_results, optimal_avg)
    print_convergence_table(averaged_results)
    print_regret_comparison(averaged_results)
    print_speedup_matrix(averaged_results, "Ollama")

    # ── Pressure tests ──
    if args.pressure:
        print("\nRunning pressure tests...")

        pressure_results = []

        for strategy_fn in strategy_classes:
            strategy = strategy_fn()
            if isinstance(strategy, OracleStrategy):
                continue  # Oracle doesn't have state to test
            result = pressure_test_simulated(
                strategy, n_requests=500, concurrency=10, seed=args.seed
            )
            pressure_results.append((strategy.name, result))

        print_pressure_results(pressure_results)
        print_concurrency_scaling(args.seed)

    # ── Save JSON ──
    output_path = "swiftlet_benchmark_results.json"
    output = {
        "config": {
            "requests": args.requests,
            "trials": args.trials,
            "seed": args.seed,
        },
        "optimal_per_signature": optimal_per_sig,
        "results": [
            {
                "strategy": r.strategy_name,
                "avg_tps": r.avg_tps,
                "best_tps": r.best_tps,
                "cumulative_regret": r.cumulative_regret,
                "quality_pct": round((r.avg_tps / optimal_avg) * 100, 1),
                "config_diversity": r.config_diversity,
            }
            for r in averaged_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

 
if __name__ == "__main__":
    main()

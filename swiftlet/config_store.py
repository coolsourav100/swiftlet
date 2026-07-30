"""
Learned configuration store — the piece directly inspired by colibrì's
"gets faster the more you use it" learning cache, adapted from "which
experts are hot" to "which CPU/GPU split works best for this workload shape."

Persisted to a local JSON file so learning survives process restarts,
the same way colibrì persists its .coli_usage file across runs.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .classifier import WorkloadSignature, Phase


@dataclass
class EngineConfig:
    """A concrete llama.cpp launch configuration."""
    n_gpu_layers: int
    n_cpu_moe: int
    batch_size: int = 512

    def key(self) -> str:
        return f"g{self.n_gpu_layers}_m{self.n_cpu_moe}_b{self.batch_size}"


@dataclass
class ConfigStats:
    config: EngineConfig
    trials: int = 0
    total_tok_per_sec: float = 0.0

    @property
    def mean_tok_per_sec(self) -> float:
        return self.total_tok_per_sec / self.trials if self.trials else 0.0


def _default_configs_for_phase(phase: Phase) -> list[EngineConfig]:
    """
    Reasonable starting candidates before any real measurement exists,
    based on BB-CEP Section 3's regime analysis:
      - prefill is compute-bound  -> GPU-heavy, no CPU-MoE split needed
      - decode is memory-bound    -> a CPU/GPU split can help (Section 4/5)
      - balanced                  -> hedge with a mid-range split
    These are priors to explore from, not conclusions — the store is
    expected to override them once it has real data.
    """
    if phase == Phase.PREFILL_HEAVY:
        return [EngineConfig(n_gpu_layers=99, n_cpu_moe=0)]
    if phase == Phase.DECODE_HEAVY:
        return [
            EngineConfig(n_gpu_layers=99, n_cpu_moe=n)
            for n in (0, 8, 16)
        ]
    return [EngineConfig(n_gpu_layers=99, n_cpu_moe=n) for n in (0, 6, 12)]


class LearnedConfigStore:
    """
    Maps WorkloadSignature -> best known EngineConfig, with epsilon-greedy
    exploration so the store keeps discovering better configs instead of
    permanently committing to whatever performed best on a small early sample.
    """

    def __init__(self, path: str | Path, epsilon: float = 0.15, seed: int | None = None):
        self.path = Path(path)
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        # signature_str -> {config_key: ConfigStats}
        self._data: dict[str, dict[str, ConfigStats]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for sig_str, configs in raw.items():
            self._data[sig_str] = {}
            for cfg_key, stats in configs.items():
                cfg = EngineConfig(**stats["config"])
                self._data[sig_str][cfg_key] = ConfigStats(
                    config=cfg,
                    trials=stats["trials"],
                    total_tok_per_sec=stats["total_tok_per_sec"],
                )

    def save(self) -> None:
        out = {}
        for sig_str, configs in self._data.items():
            out[sig_str] = {
                cfg_key: {
                    "config": asdict(stats.config),
                    "trials": stats.trials,
                    "total_tok_per_sec": stats.total_tok_per_sec,
                }
                for cfg_key, stats in configs.items()
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(out, indent=2))

    def choose_config(self, signature: WorkloadSignature) -> tuple[EngineConfig, bool]:
        """
        Returns (config, is_exploration).
        is_exploration=True means this is a deliberate probe of an
        under-tried config, not the current best pick — useful for the
        caller to know it might be slower than usual, by design.
        """
        sig_str = str(signature)
        known = self._data.get(sig_str, {})

        candidates = _default_configs_for_phase(signature.phase)
        untried = [c for c in candidates if c.key() not in known]

        # Explore: either we have zero data, or epsilon-greedy roll says try
        # something new rather than exploit the current best.
        if untried or self._rng.random() < self.epsilon:
            pool = untried if untried else candidates
            return self._rng.choice(pool), True

        # Exploit: return the best-performing config seen so far.
        best = max(known.values(), key=lambda s: s.mean_tok_per_sec)
        return best.config, False

    def record_result(self, signature: WorkloadSignature, config: EngineConfig, tok_per_sec: float) -> None:
        sig_str = str(signature)
        self._data.setdefault(sig_str, {})
        cfg_key = config.key()
        if cfg_key not in self._data[sig_str]:
            self._data[sig_str][cfg_key] = ConfigStats(config=config)
        stats = self._data[sig_str][cfg_key]
        stats.trials += 1
        stats.total_tok_per_sec += tok_per_sec
        self.save()

    def best_known(self, signature: WorkloadSignature) -> ConfigStats | None:
        known = self._data.get(str(signature), {})
        if not known:
            return None
        return max(known.values(), key=lambda s: s.mean_tok_per_sec)

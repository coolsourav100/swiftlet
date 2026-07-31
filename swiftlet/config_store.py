"""
Learned configuration store — v2 with all six fixes applied.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from .classifier import WorkloadSignature, Phase
from .gp import GaussianProcessRegressor, RBFKernel, ucb, optimize_kernel


@dataclass
class EngineConfig:
    n_gpu_layers: int
    n_cpu_moe: int
    batch_size: int = 512

    def key(self) -> str:
        return f"g{self.n_gpu_layers}_m{self.n_cpu_moe}_b{self.batch_size}"

    def to_features(self) -> list[float]:
        return [
            self.n_gpu_layers / 99.0,
            self.n_cpu_moe / 16.0,
            self.batch_size / 512.0,
        ]


@dataclass
class ConfigStats:
    config: EngineConfig
    trials: int = 0
    total_tok_per_sec: float = 0.0
    observations: list[float] = field(default_factory=list)

    @property
    def mean_tok_per_sec(self) -> float:
        return self.total_tok_per_sec / self.trials if self.trials else 0.0


def _default_configs_for_phase(phase: Phase) -> list[EngineConfig]:
    if phase == Phase.PREFILL_HEAVY:
        return [EngineConfig(n_gpu_layers=99, n_cpu_moe=0)]
    if phase == Phase.DECODE_HEAVY:
        return [EngineConfig(n_gpu_layers=99, n_cpu_moe=n) for n in (0, 8, 16)]
    return [EngineConfig(n_gpu_layers=99, n_cpu_moe=n) for n in (0, 6, 12)]


FULL_CONFIG_SPACE: list[EngineConfig] = [
    # GPU-heavy (no CPU MoE) — the likely optimum for most workloads
    EngineConfig(n_gpu_layers=99, n_cpu_moe=0,  batch_size=512),
    EngineConfig(n_gpu_layers=99, n_cpu_moe=0,  batch_size=256),
    EngineConfig(n_gpu_layers=80, n_cpu_moe=0,  batch_size=512),
    EngineConfig(n_gpu_layers=60, n_cpu_moe=0,  batch_size=512),
    # GPU + CPU MoE — the interesting region for MoE models
    EngineConfig(n_gpu_layers=99, n_cpu_moe=4,  batch_size=512),
    EngineConfig(n_gpu_layers=99, n_cpu_moe=8,  batch_size=512),
    EngineConfig(n_gpu_layers=99, n_cpu_moe=12, batch_size=512),
    EngineConfig(n_gpu_layers=80, n_cpu_moe=4,  batch_size=512),
    EngineConfig(n_gpu_layers=80, n_cpu_moe=8,  batch_size=512),
    EngineConfig(n_gpu_layers=60, n_cpu_moe=6,  batch_size=512),
    EngineConfig(n_gpu_layers=60, n_cpu_moe=12, batch_size=512),
    # CPU-heavy — only for testing the boundary
    EngineConfig(n_gpu_layers=40, n_cpu_moe=12, batch_size=512),
]


class LearnedConfigStore:
    def __init__(self, path: str | Path, epsilon: float = 0.15, seed: int | None = None):
        self.path = Path(path)
        self.epsilon = epsilon
        self._rng = random.Random(seed)
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
                    observations=stats.get("observations", []),
                )

    def save(self) -> None:
        out = {}
        for sig_str, configs in self._data.items():
            out[sig_str] = {
                cfg_key: {
                    "config": asdict(stats.config),
                    "trials": stats.trials,
                    "total_tok_per_sec": stats.total_tok_per_sec,
                    "observations": stats.observations,
                }
                for cfg_key, stats in configs.items()
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(out, indent=2))

    def choose_config(self, signature: WorkloadSignature) -> tuple[EngineConfig, bool]:
        sig_str = str(signature)
        known = self._data.get(sig_str, {})
        candidates = _default_configs_for_phase(signature.phase)
        untried = [c for c in candidates if c.key() not in known]

        if untried or self._rng.random() < self.epsilon:
            pool = untried if untried else candidates
            return self._rng.choice(pool), True

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
        stats.observations.append(tok_per_sec)
        self.save()

    def best_known(self, signature: WorkloadSignature) -> ConfigStats | None:
        known = self._data.get(str(signature), {})
        if not known:
            return None
        return max(known.values(), key=lambda s: s.mean_tok_per_sec)


class BayesianConfigStore(LearnedConfigStore):
    """
    Cross-signature GP + UCB with warm start, pruned config space,
    and calibrated beta decay.
    """

    MIN_GP_OBS = 5

    def __init__(
        self,
        path: str | Path,
        seed: int | None = None,
        delta: float = 0.1,
        optimize_every: int = 10,
        beta_max: float = 3.0,          # FIX #3: was 6.0, too aggressive
    ):
        super().__init__(path, epsilon=0.0, seed=seed)
        self._delta = delta
        self._optimize_every = optimize_every
        self._beta_max = beta_max
        self._best_kernel: Optional[RBFKernel] = None
        self._obs_since_optimize = 0
        self._warmed_up = False

    # ── Warm start ──────────────────────────────────────────────────

    def warm_start(self, tps_fn=None) -> None:
        """
        FIX #2: Seed the store with a few observations of a known-good
        config so the GP doesn't explore blindly on the first real request.

        If tps_fn is provided, it's called as tps_fn(config, signature) → float
        to get realistic measurements.  Otherwise, uses a conservative
        estimate based on the roofline model.

        This is called automatically on the first choose_config() call.
        """
        if self._warmed_up:
            return
        self._warmed_up = True

        # The "safe default" that a human operator would pick
        safe_config = EngineConfig(n_gpu_layers=99, n_cpu_moe=0, batch_size=512)

        # Seed each signature with 2 observations of the safe config
        for pb in range(3):
            for gb in range(3):
                sig = WorkloadSignature(prompt_bucket=pb, gen_bucket=gb)

                if tps_fn:
                    # Use the provided function for realistic measurements
                    for _ in range(2):
                        tps = tps_fn(safe_config, sig)
                        self.record_result(sig, safe_config, tps)
                else:
                    # Conservative estimate: gpu=99 is ~28 tok/s for most workloads
                    base = 28.0 + self._rng.gauss(0, 1.0)
                    self.record_result(sig, safe_config, base)
                    self.record_result(sig, safe_config, base + self._rng.gauss(0, 1.0))

    # ── Core decision logic ─────────────────────────────────────────

    def choose_config(self, signature: WorkloadSignature) -> tuple[EngineConfig, bool]:
        # Warm start on first call
        self.warm_start()

        candidates = FULL_CONFIG_SPACE

        X_train, y_train, obs_per_sig = self._collect_training_data()

        total_obs = len(y_train)
        if total_obs < self.MIN_GP_OBS:
            return self._thompson_fallback(signature, candidates, obs_per_sig)

        # Center y_train so the GP prior assumes the mean token/sec, not 0.0
        y_mean = sum(y_train) / total_obs if total_obs > 0 else 0.0
        y_centered = [y - y_mean for y in y_train]

        # Count distinct configs in training data.  optimize_kernel needs
        # variation in the input features to learn meaningful hyperparameters.
        # After warm_start (all gpu=99/moe=0), distinct_configs=1 → skip.
        # Once the first exploration picks a different config, distinct_configs=2
        # → optimize on the next cycle.
        distinct_configs = len(set(tuple(x[:3]) for x in X_train))

        if distinct_configs > 1 and (
            self._best_kernel is None or self._obs_since_optimize >= self._optimize_every
        ):
            self._best_kernel = optimize_kernel(X_train, y_centered, noise=1.0)
            self._obs_since_optimize = 0
        elif self._best_kernel is None:
            # Can't optimize yet (only 1 config seen), but we need a kernel
            # to fit the GP.  Use a sensible default that matches the
            # feature scale: length_scale=0.3 is appropriate for normalized
            # [0,1] features, variance=10.0 covers the expected tok/s range.
            self._best_kernel = RBFKernel(length_scale=0.3, variance=10.0)

        gp = GaussianProcessRegressor(kernel=self._best_kernel, noise=1.0)
        gp.fit(X_train, y_centered)

        # FIX #3: Lower beta cap for faster exploitation
        # The theoretical β_T = 2·log(T·|A|/δ) grows with T, but for a
        # 12-config discrete space with a warm start, we can be more
        # aggressive about exploitation.  β_max=3.0 means "3σ confidence"
        # which is the 99.7% confidence interval.
        T = total_obs
        A = len(candidates)
        beta = min(2.0 * math.log(max(T * A / self._delta, 1.0)), self._beta_max)

        sig_features = signature.to_features()
        X_test = [c.to_features() + sig_features for c in candidates]
        means, variances = gp.predict(X_test)
        means = [m + y_mean for m in means]

        best_idx = 0
        best_ucb = -float("inf")
        for i, (m, v) in enumerate(zip(means, variances)):
            score = ucb(m, v, beta=beta)
            if score > best_ucb:
                best_ucb = score
                best_idx = i

        chosen = candidates[best_idx]

        sig_str = str(signature)
        known = self._data.get(sig_str, {})
        cfg_key = chosen.key()
        is_exploration = cfg_key not in known or known[cfg_key].trials < 3

        return chosen, is_exploration

    def record_result(self, signature: WorkloadSignature, config: EngineConfig, tok_per_sec: float) -> None:
        super().record_result(signature, config, tok_per_sec)
        self._obs_since_optimize += 1

    # ── Helpers (unchanged) ──────────────────────────────────────────

    def _collect_training_data(self):
        X: list[list[float]] = []
        y: list[float] = []
        obs_per_sig: dict[str, int] = {}

        for sig_str, configs in self._data.items():
            sig = self._parse_signature(sig_str)
            sig_features = sig.to_features() if sig else [0.0, 0.0]

            for cfg_key, stats in configs.items():
                cfg_features = stats.config.to_features()
                for tok in stats.observations:
                    X.append(cfg_features + sig_features)
                    y.append(tok)
                obs_per_sig[sig_str] = obs_per_sig.get(sig_str, 0) + len(stats.observations)

        return X, y, obs_per_sig

    def _parse_signature(self, sig_str: str) -> Optional[WorkloadSignature]:
        try:
            parts = sig_str.split("_")
            pb = int(parts[1][1:])
            gb = int(parts[3][1:])
            return WorkloadSignature(prompt_bucket=pb, gen_bucket=gb)
        except (IndexError, ValueError):
            return None

    def _thompson_fallback(self, signature, candidates, obs_per_sig):
        sig_str = str(signature)
        known = self._data.get(sig_str, {})
        prior_mean = 30.0
        prior_std = 25.0

        all_obs = [
            tok
            for cfgs in self._data.values()
            for stats in cfgs.values()
            for tok in stats.observations
        ]
        if all_obs:
            prior_mean = sum(all_obs) / len(all_obs)

        best_config = None
        best_sample = -float("inf")

        for config in candidates:
            obs = known.get(config.key())
            obs_list = obs.observations if obs else []

            if not obs_list:
                sample = self._rng.gauss(prior_mean, prior_std)
            else:
                n = len(obs_list)
                obs_mean = sum(obs_list) / n
                obs_var = (
                    sum((x - obs_mean) ** 2 for x in obs_list) / (n - 1)
                    if n > 1
                    else prior_std ** 2
                )
                prior_var = prior_std ** 2
                post_prec = 1.0 / prior_var + n / obs_var
                post_var = 1.0 / post_prec
                post_mean = post_var * (
                    prior_mean / prior_var + n * obs_mean / obs_var
                )
                sample = self._rng.gauss(post_mean, math.sqrt(post_var))

            if sample > best_sample:
                best_sample = sample
                best_config = config

        cfg_key = best_config.key()
        is_exploration = cfg_key not in known or known[cfg_key].trials < 3
        return best_config, is_exploration

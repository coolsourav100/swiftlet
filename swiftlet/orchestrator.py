"""
Orchestrator — manages a small pool of pre-warmed llama-server processes at
learned-good configs, and routes each incoming request to the right one
based on the classifier + config store decision.

This is the part that requires a real llama-server binary + model to run
end-to-end. The pool-management and routing *logic* below is fully testable
without one (see tests/test_orchestrator.py) — what's NOT included here is
an actual subprocess launch of llama-server, which needs your real model
path and hardware. See docs/validation.md for wiring this to a live run.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

from .classifier import classify, WorkloadSignature
from .config_store import LearnedConfigStore, EngineConfig


@dataclass
class ServerHandle:
    """Represents a (real or simulated) running llama-server instance."""
    config: EngineConfig
    port: int
    started_at: float
    process: subprocess.Popen | None = None  # set by a real launcher; None for fakes/tests


class ServerPool:
    """
    Bounded pool of server handles, keyed by config. Bounded because each
    llama-server instance holds a model copy in memory — you can't just
    spin up one per distinct config on a 16GB Mac. When the pool is full,
    the least-recently-used handle is evicted to make room.
    """

    def __init__(self, max_size: int, launcher=None):
        """
        launcher: callable(EngineConfig, port) -> ServerHandle. Injected so
        tests can use a fake launcher instead of actually spawning
        llama-server. Defaults to a stub that raises, forcing callers to be
        explicit about the fact that no real launch is wired up yet.
        """
        self.max_size = max_size
        self._launcher = launcher or self._unimplemented_launcher
        self._pool: dict[str, ServerHandle] = {}
        self._last_used: dict[str, float] = {}
        self._next_port = 8080
        self._lock = threading.Lock()

    @staticmethod
    def _unimplemented_launcher(config: EngineConfig, port: int) -> ServerHandle:
        raise NotImplementedError(
            "No real llama-server launcher configured. Provide one that "
            "spawns `llama-server -m <model> --n-gpu-layers N --n-cpu-moe M "
            "--port <port>` and confirms it's ready before returning."
        )

    def get_or_launch(self, config: EngineConfig) -> ServerHandle:
        """
        Thread-safe: the whole check-launch-insert sequence holds the lock,
        so two concurrent requests for the same new config can't both miss
        the cache and race to launch duplicate servers on the same port.
        The launcher call itself (which may block for tens of seconds on a
        cold model load) happens WHILE the lock is held — a deliberate
        trade-off: a second thread wanting a *different* config will wait
        rather than race, at the cost of the pool being briefly unusable
        during any cold start. Acceptable for a single-Mac deployment;
        would need a per-key lock instead of one global lock to scale further.
        """
        with self._lock:
            key = config.key()
            self._last_used[key] = time.time()

            if key in self._pool:
                return self._pool[key]

            if len(self._pool) >= self.max_size:
                lru_key = min(
                    (k for k in self._pool),
                    key=lambda k: self._last_used.get(k, float("inf")),
                )
                self._terminate_handle(self._pool[lru_key])
                del self._pool[lru_key]

            handle = self._launcher(config, self._next_port)
            self._next_port += 1
            self._pool[key] = handle
            return handle

    @staticmethod
    def _terminate_handle(handle: ServerHandle) -> None:
        if handle.process is None:
            return  # fake/test handle, nothing real to terminate
        handle.process.terminate()
        try:
            handle.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.process.kill()
            handle.process.wait(timeout=5)

    def shutdown(self) -> None:
        """Terminate every real llama-server process this pool launched."""
        with self._lock:
            for handle in self._pool.values():
                self._terminate_handle(handle)
            self._pool.clear()

    def active_configs(self) -> list[EngineConfig]:
        return [h.config for h in self._pool.values()]


class Orchestrator:
    """
    Ties classifier + config store + server pool together into the
    end-to-end decision: given a request, which server should handle it,
    and what should we record afterward.
    """

    def __init__(self, store: LearnedConfigStore, pool: ServerPool):
        self.store = store
        self.pool = pool

    def route(self, prompt_tokens: int, expected_gen_tokens: int) -> tuple[WorkloadSignature, EngineConfig, ServerHandle, bool]:
        signature = classify(prompt_tokens, expected_gen_tokens)
        config, is_exploration = self.store.choose_config(signature)
        handle = self.pool.get_or_launch(config)
        return signature, config, handle, is_exploration

    def record(self, signature: WorkloadSignature, config: EngineConfig, tok_per_sec: float) -> None:
        self.store.record_result(signature, config, tok_per_sec)

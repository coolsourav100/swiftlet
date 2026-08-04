import httpx
import os
import subprocess
import time

from swiftlet.backends.base import BackendLauncher, BackendHandle
from swiftlet.config_store import EngineConfig
from swiftlet.logging_config import get_logger

_log = get_logger("llamacpp")

class LlamaCppLauncher(BackendLauncher):
    def __init__(self, model_path: str, bin_path: str, ctx_size: int, threads: int, startup_timeout: int = 300, cache_dir: str = ".swiftlet_cache"):
        self.model_path = model_path
        self.bin_path = bin_path
        self.ctx_size = ctx_size
        self.threads = threads
        self.startup_timeout = startup_timeout
        self.cache_dir = cache_dir

    def launch(self, config: EngineConfig, port: int) -> BackendHandle:
        _log.info(
            f"[launching llamacpp] {self.bin_path} --n-gpu-layers {config.n_gpu_layers} "
            f"--n-cpu-moe {config.n_cpu_moe} --batch-size {config.batch_size} --threads {self.threads} --port {port}"
        )

        proc = subprocess.Popen([
            self.bin_path,
            "-m", self.model_path,
            "--n-gpu-layers", str(config.n_gpu_layers),
            "--n-cpu-moe", str(config.n_cpu_moe),
            "--batch-size", str(config.batch_size),
            "--threads", str(self.threads),
            "-np", "1",
            "-c", str(self.ctx_size),
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--slot-save-path", self.cache_dir,
            "--port", str(port),
        ])

        poll_interval = 2
        elapsed = 0

        while elapsed < self.startup_timeout:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server on port {port} exited early (code {proc.returncode}) "
                    f"during startup — check its logs above for the actual error."
                )

            try:
                res = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if res.status_code == 200:
                    _log.info(f"[ready] llamacpp on port {port} came up after {elapsed}s")
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RequestError):
                pass

            time.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed % 20 == 0:
                _log.info(f"[waiting] still loading on port {port}... ({elapsed}s elapsed)")
        else:
            proc.terminate()
            raise RuntimeError(
                f"llama-server on port {port} didn't come up within {self.startup_timeout}s."
            )

        return BackendHandle(config=config, port=port, started_at=time.time(), process=proc)

import httpx
import subprocess
import time
import sys

from swiftlet.backends.base import BackendLauncher, BackendHandle
from swiftlet.config_store import EngineConfig
from swiftlet.logging_config import get_logger

_log = get_logger("mlx")

class MLXLauncher(BackendLauncher):
    def __init__(self, model_path: str, startup_timeout: int = 300):
        self.model_path = model_path
        self.startup_timeout = startup_timeout

    def launch(self, config: EngineConfig, port: int) -> BackendHandle:
        # Note: MLX dynamically handles unified memory scheduling automatically.
        # It doesn't accept n_gpu_layers or n_cpu_moe as explicit CLI parameters.
        # Swiftlet will still route the request and record tok/s, but MLX makes the final hardware split.
        _log.info(
            f"[launching mlx] python3 -m mlx_lm.server --model {self.model_path} --port {port}"
        )

        proc = subprocess.Popen([
            sys.executable,
            "-m", "mlx_lm.server",
            "--model", self.model_path,
            "--port", str(port),
        ])

        poll_interval = 2
        elapsed = 0

        while elapsed < self.startup_timeout:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"mlx_lm.server on port {port} exited early (code {proc.returncode}) "
                    f"during startup — check its logs above for the actual error."
                )

            try:
                # mlx_lm.server responds to /v1/models (which is OpenAI compatible)
                res = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
                if res.status_code == 200:
                    _log.info(f"[ready] mlx on port {port} came up after {elapsed}s")
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
                f"mlx_lm.server on port {port} didn't come up within {self.startup_timeout}s."
            )

        return BackendHandle(config=config, port=port, started_at=time.time(), process=proc)

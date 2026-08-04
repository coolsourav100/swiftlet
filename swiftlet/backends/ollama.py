"""
Ollama backend — talks to a running Ollama instance instead of launching
a raw llama-server. Swiftlet still handles workload classification and
routing, but delegates model management to Ollama.
"""

import httpx
import time

from swiftlet.backends.base import BackendLauncher, BackendHandle
from swiftlet.config_store import EngineConfig


class OllamaLauncher(BackendLauncher):
    """
    Unlike LlamaCppLauncher, this doesn't spawn a new process per config.
    Ollama manages its own model loading. We point at the running Ollama
    API and let Swiftlet's classifier + store still learn from the results.
    
    The n_gpu_layers/n_cpu_moe config values are passed as Ollama
    'options' when possible, though Ollama may ignore some of them.
    """

    def __init__(
        self,
        model_name: str,
        ollama_host: str = "http://localhost:11434",
        startup_timeout: int = 60,
    ):
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.startup_timeout = startup_timeout

    def launch(self, config: EngineConfig, port: int) -> BackendHandle:
        """
        Ensure the model is loaded in Ollama. We don't actually use
        the 'port' parameter since Ollama runs on its own port —
        but we return it for compatibility with ServerPool.
        """
        from swiftlet.logging_config import get_logger
        log = get_logger("ollama")

        log.info(
            f"[ollama] Ensuring model '{self.model_name}' is loaded "
            f"(gpu_layers={config.n_gpu_layers}, host={self.ollama_host})"
        )

        # Preload model by sending a keep-alive request
        try:
            resp = httpx.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "",
                    "keep_alive": "10m",
                    "options": {
                        "num_gpu": config.n_gpu_layers,
                        "num_thread": 8,
                    },
                },
                timeout=self.startup_timeout,
            )
            if resp.status_code != 200:
                log.warning(f"[ollama] Preload returned {resp.status_code}: {resp.text[:200]}")
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_host}. "
                f"Is Ollama running? Error: {e}"
            )

        # Wait for health
        elapsed = 0
        while elapsed < self.startup_timeout:
            try:
                res = httpx.get(f"{self.ollama_host}/api/tags", timeout=5.0)
                if res.status_code == 200:
                    models = [m["name"] for m in res.json().get("models", [])]
                    # Check model exists (handle both "name" and "name:tag" formats)
                    base_name = self.model_name.split(":")[0]
                    if any(base_name in m for m in models):
                        log.info(f"[ollama] Model '{self.model_name}' ready")
                        break
                    else:
                        log.warning(
                            f"[ollama] Model '{self.model_name}' not found. "
                            f"Available: {models}. Attempting pull..."
                        )
                        self._pull_model(log)
                        break
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass

            time.sleep(2)
            elapsed += 2

        # Ollama uses its own port — the 'port' param maps to it for ServerPool compat
        # We use a special port value to indicate it's an Ollama backend
        return BackendHandle(
            config=config,
            port=self._extract_port(),
            started_at=time.time(),
            process=None,  # Ollama manages its own process
        )

    def _extract_port(self) -> int:
        """Extract port from ollama_host URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.ollama_host)
            return parsed.port or 11434
        except Exception:
            return 11434

    def _pull_model(self, log) -> None:
        """Attempt to pull the model from Ollama registry."""
        try:
            log.info(f"[ollama] Pulling model '{self.model_name}'...")
            resp = httpx.post(
                f"{self.ollama_host}/api/pull",
                json={"name": self.model_name, "stream": False},
                timeout=600,
            )
            if resp.status_code == 200:
                log.info(f"[ollama] Pull complete for '{self.model_name}'")
            else:
                log.error(f"[ollama] Pull failed: {resp.status_code}")
        except Exception as e:
            log.error(f"[ollama] Pull error: {e}")

    @staticmethod
    def list_models(ollama_host: str = "http://localhost:11434") -> list[dict]:
        """List all models available in Ollama."""
        try:
            resp = httpx.get(f"{ollama_host}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except Exception:
            pass
        return []

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Protocol

from swiftlet.config_store import EngineConfig

@dataclass
class BackendHandle:
    """Represents a running backend instance."""
    config: EngineConfig
    port: int
    started_at: float
    process: subprocess.Popen | None = None

class BackendLauncher(Protocol):
    def launch(self, config: EngineConfig, port: int) -> BackendHandle:
        """
        Launch the backend with the given configuration on the specified port.
        Must block until the server is healthy and responding to requests, or raise an error.
        """
        ...

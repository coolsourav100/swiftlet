"""
TOML configuration loader for swiftlet.
Reads swiftlet.toml (if present) and provides defaults that CLI flags override.
"""

from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for 3.10
    except ImportError:
        tomllib = None  # type: ignore


DEFAULT_CONFIG = {
    "model": {
        "path": "",
        "ctx_size": 8192,
    },
    "server": {
        "backend": "llamacpp",
        "llama_server": "llama-server",
        "pool_size": 1,
        "threads": 8,
        "startup_timeout": 300,
    },
    "learning": {
        "algorithm": "eps-greedy",
        "store_path": "swiftlet_learned_config.json",
    },
    "logging": {
        "level": "INFO",
        "log_dir": "",
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a TOML file, falling back to defaults.
    
    Search order:
    1. Explicit path (if given)
    2. ./swiftlet.toml (current directory)
    3. ~/.config/swiftlet/swiftlet.toml
    """
    if tomllib is None:
        return _deep_copy(DEFAULT_CONFIG)

    paths_to_try = []
    if config_path:
        paths_to_try.append(Path(config_path))
    paths_to_try.append(Path.cwd() / "swiftlet.toml")
    paths_to_try.append(Path.home() / ".config" / "swiftlet" / "swiftlet.toml")

    config = _deep_copy(DEFAULT_CONFIG)

    for p in paths_to_try:
        if p.exists():
            with open(p, "rb") as f:
                user_config = tomllib.load(f)
            _deep_merge(config, user_config)
            config["_loaded_from"] = str(p)
            break

    return config


def _deep_copy(d: dict) -> dict:
    return {k: _deep_copy(v) if isinstance(v, dict) else v for k, v in d.items()}


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base, modifying base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def generate_default_toml() -> str:
    """Generate a commented swiftlet.toml template."""
    return '''# swiftlet.toml — Configuration file for Swiftlet
# CLI flags override any values set here.

[model]
# path = "~/.ollama/models/qwen3-30b-a3b.gguf"
ctx_size = 8192

[server]
backend = "llamacpp"      # "llamacpp" or "mlx" or "ollama"
# llama_server = "/Applications/Ollama.app/Contents/Resources/llama-server"
pool_size = 1
threads = 8
startup_timeout = 300

[learning]
algorithm = "bayesian"     # "eps-greedy" or "bayesian"
store_path = "swiftlet_learned_config.json"

[logging]
level = "INFO"             # DEBUG, INFO, WARNING, ERROR
# log_dir = "~/.swiftlet/logs"

[web]
search = false             # Enable privacy-safe web search (DuckDuckGo)
'''

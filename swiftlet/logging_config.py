"""
Centralized logging for swiftlet.
Replaces raw print() calls with structured Python logging.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> logging.Logger:
    """
    Configure and return the root swiftlet logger.
    
    - Console output: colored, concise format
    - File output (optional): full timestamps, written to log_dir/swiftlet.log
    """
    logger = logging.getLogger("swiftlet")
    
    if logger.handlers:
        return logger  # Already configured
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Console handler — concise output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(_ColorFormatter("  %(message)s"))
    logger.addHandler(console)

    # File handler — detailed output
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "swiftlet.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    return logger


class _ColorFormatter(logging.Formatter):
    """Adds ANSI color codes to log level names for terminal output."""
    
    COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        # Add level prefix for non-INFO messages
        if record.levelno != logging.INFO:
            record.msg = f"{color}[{record.levelname}]{self.RESET} {record.msg}"
        return super().format(record)


def get_logger(name: str = "swiftlet") -> logging.Logger:
    """Get a child logger under the swiftlet namespace."""
    return logging.getLogger(f"swiftlet.{name}")

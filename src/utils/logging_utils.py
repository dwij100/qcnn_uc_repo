"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    """Configure console and file logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
        force=True,
    )

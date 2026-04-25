from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_logger(name: str = "qcnn_uc", log_file: Optional[str | Path] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(path) for h in logger.handlers):
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"))
            logger.addHandler(fh)

    return logger

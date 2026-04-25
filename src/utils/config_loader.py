from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the project root by walking upward until config/config.yaml exists."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "config" / "config.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing config/config.yaml")


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    root = find_project_root()
    path = Path(config_path) if config_path else root / "config" / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(root)
    return cfg


def project_path(cfg: Dict[str, Any], *parts: str) -> Path:
    return Path(cfg["_project_root"]).joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
